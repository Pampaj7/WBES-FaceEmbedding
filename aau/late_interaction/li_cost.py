#!/usr/bin/env python3
"""Costo per confronto dell'interazione tardiva, sullo stesso nodo del confronto vettoriale e dell'ICP.

    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 AAU_NV=--nv aau/competitors/gw_percettivi/gp_run.sh \
        aau/late_interaction/li_cost.py --work <tmp>                               (li.sbatch, passo cost; L40S)

Campione = quello di ``aau/costo_confronto/cc.py sample`` (HIFI3D, coppie original-down8k, original-up60k,
down8k-up60k, ``N_PAIRS`` per tipo, e le mesh cronometrate), scritto da ``li.sbatch`` in ``OUT/costo/cc``; nello stesso
job ``cc.py geo --metric icp`` (ICP + Chamfer in mm per coppia, CPU a un thread) e ``cc.py cmp`` (d_F e d_P fra
vettori, ``fact_paired.distances``, a blocchi di tutte le coppie di N mesh, CPU a un thread), con le loro definizioni.

Qui, seme s1234, GPU: per mesh la forward coi token (``li_tokens.Embedder``: la stessa forward del vettore, piu' la
lettura delle feature per vertice) e il sottocampione FPS con le masse di Voronoi; per coppia ISOLATA (una alla volta,
``torch.cuda.synchronize``) ogni variante (``li_dist.chamfer_pair``; ``cost_batch`` + ``chamfer_batch`` /
``sinkhorn_batch`` con un blocco di una coppia), epsilon del seme da ``WORK/dist/s1234/run.json``. Un giro di
riscaldamento escluso + ``--reps`` giri; per elemento la mediana dei giri, poi mediana [p25, p75] fra gli elementi
(come ``cc.report``). Memoria per mesh: byte dei token (float32) contro il vettore [s, u]. Gli operatori della mesh
(identici per vettore e token) non si ricalcolano: sono in ``costo_confronto`` (``cc.py ops``). Uscite
``OUT/costo/costo.json`` (tabella per ``results.md``) e ``OUT/costo/tempi.csv``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import li_common as lc

CC_OUT = lc.OUT / "costo" / "cc"


def fmt(s: float) -> str:
    if s >= 1:
        return f"{s:.2f} s"
    if s >= 1e-3:
        return f"{s * 1e3:.2f} ms"
    return f"{s * 1e6:.2f} us"


def stats(x) -> dict:
    x = np.asarray(x, np.float64)
    return {"n": len(x), "median": float(np.median(x)), "p25": float(np.percentile(x, 25)),
            "p75": float(np.percentile(x, 75))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--reps", type=int, default=3)
    a = p.parse_args()
    import li_dist as ld
    import li_tokens as lt
    S = json.loads((CC_OUT / "parts" / "sample.json").read_text())
    sets = lc.load_sets()
    st = sets["hifi3d"]
    E = lt.Embedder("s1234", [st["table"]])
    torch = E.torch
    eps = json.loads((lc.WORK / "dist" / "s1234" / "run.json").read_text())["eps"]
    ch = json.loads((lc.OUT / "pilot_choice.json").read_text())["choice"]
    ds = E.base.GTReadyDataset(st["ops"])
    pos = {Path(f).name: k for k, f in enumerate(ds.files)}
    meshes = sorted({int(x) for pr in S["pairs"] for x in pr[:2]})
    name = {k: Path(S["path"][str(k)]).name for k in meshes}
    timed = set(int(k) for k in S["timed"])
    rows, tok = [], {}
    for r in range(a.reps + 1):                                          # per mesh: forward coi token, sottocampione
        for k in (meshes if r == 0 else sorted(timed)):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            z, Z, w, X = E(ds, pos[name[k]])
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            idx, m = lt.subsample(X, w, lc.N_SUB)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            rows += [{"level": "mesh", "item": name[k], "rep": r, "phase": ph, "s": s}
                     for ph, s in (("forward_e_token", t1 - t0), ("sottocampione", t2 - t1))]
            if r == 0:
                tok[k] = (Z, w, Z[idx], m, Z.numel() * 4, z.size * 4)
    mu = torch.cat([tok[k][0] for k in meshes]).mean(0, keepdim=True)  # centratura come li_tokens.embed_set
    tok = {k: (Z - mu, w, Sx - mu, m, nb, nz) for k, (Z, w, Sx, m, nb, nz) in tok.items()}
    for r in range(a.reps + 1):                                          # per coppia isolata
        for pa, pb, typ in S["pairs"]:
            A, B = tok[int(pa)], tok[int(pb)]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ld.chamfer_pair(A[0], A[1], B[0], B[1])
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            C = ld.cost_batch(A[2][None], B[2][None])
            ld.chamfer_batch(C, A[3][None], B[3][None]).cpu()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            C = ld.cost_batch(A[2][None], B[2][None])
            ld.sinkhorn_batch(C, A[3][None], B[3][None], eps)[0].cpu()
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            rows += [{"level": "coppia", "item": f"{pa}|{pb}", "group": typ, "rep": r, "phase": v, "s": s}
                     for v, s in zip(lc.VARIANTS, (t1 - t0, t2 - t1, t3 - t2))]
    import pandas as pd
    df = pd.DataFrame(rows)
    (lc.OUT / "costo").mkdir(parents=True, exist_ok=True)
    df.to_csv(lc.OUT / "costo" / "tempi.csv", index=False)
    it = df[df["rep"] > 0].groupby(["level", "item", "phase"], as_index=False)["s"].median()
    out = {"variante_dichiarata": ch, "eps": eps, "n_sub": lc.N_SUB, "gpu": torch.cuda.get_device_name(0),
           "per_coppia": {v: stats(it[(it.level == "coppia") & (it.phase == v)]["s"]) for v in lc.VARIANTS},
           "per_mesh": {ph: stats(it[(it.level == "mesh") & (it.phase == ph)]["s"])
                        for ph in ("forward_e_token", "sottocampione")},
           "byte_per_mesh": {"token": stats([t[4] for t in tok.values()]), "vettore": stats([t[5] for t in tok.values()]),
                             "sottocampione": lc.N_SUB * 256 * 4}, **lc.stamp()}
    cc = {}                                                              # ICP e vettori dalle parti di cc.py, stesso job
    icp = pd.DataFrame(json.loads((CC_OUT / "parts" / "icp.json").read_text())["rows"])
    icp = icp[(icp.level == "coppia") & (icp.rep > 0)].groupby(["item", "phase"], as_index=False)["s"].median()
    cc["icp_per_coppia"] = stats(icp.groupby("item")["s"].sum())
    cm = json.loads((CC_OUT / "parts" / "cmp.json").read_text())
    cmd = pd.DataFrame(cm["rows"])
    for N in cm["n_cmp"]:
        b = cmd[(cmd.group == f"N={N}") & (cmd.rep > 0) & (cmd.phase == "confronti")]["s"]
        cc[f"vettore_per_coppia_N{N}"] = float(np.median(b)) / (N * (N - 1) / 2)
    c1 = cmd[(cmd.phase == "chiamata_1_coppia") & (cmd.rep > 0)]["s"]
    cc["vettore_chiamata_1_coppia"] = float(np.median(c1))
    cc["env_icp"] = json.loads((CC_OUT / "parts" / "icp.json").read_text())["env"]
    out["cc"] = cc
    pc = out["per_coppia"]
    md = [f"Nodo `{out['host']}`, job {out['job']}, GPU {out['gpu']}; CPU {cc['env_icp'].get('cpu', '-')} a un thread "
          "per l'ICP e i vettori. Campione di `cc.py sample` (HIFI3D, 3 tipi di coppie, "
          f"{int(pc[ch]['n'])} coppie); mediana [p25, p75] fra le coppie della mediana di {a.reps} giri.", "",
          "| metodo | dispositivo | per coppia |", "|---|---|---|"]
    for v in lc.VARIANTS:
        md.append(f"| interazione tardiva, {v}{' **(dichiarata)**' if v == ch else ''} | GPU, coppia isolata | "
                  f"{fmt(pc[v]['median'])} [{fmt(pc[v]['p25'])}, {fmt(pc[v]['p75'])}] |")
    md += [f"| vettori [s, u] (d_F e d_P), a blocchi di tutte le coppie di N mesh | CPU 1 thread | " +
           ", ".join(f"N = {N}: {fmt(cc[f'vettore_per_coppia_N{N}'])}" for N in cm["n_cmp"]) + " |",
           f"| vettori, una chiamata di `fact_paired.distances` per una coppia (rilegge il checkpoint) | CPU 1 thread | "
           f"{fmt(cc['vettore_chiamata_1_coppia'])} |",
           f"| ICP + Chamfer in mm | CPU 1 thread | {fmt(cc['icp_per_coppia']['median'])} "
           f"[{fmt(cc['icp_per_coppia']['p25'])}, {fmt(cc['icp_per_coppia']['p75'])}] |", "",
           f"Per mesh (una volta, GPU): forward coi token {fmt(out['per_mesh']['forward_e_token']['median'])}, "
           f"sottocampione FPS + Voronoi {fmt(out['per_mesh']['sottocampione']['median'])}. Memoria per mesh: token "
           f"{out['byte_per_mesh']['token']['median'] / 2 ** 20:.1f} MiB (mediana, float32) contro "
           f"{out['byte_per_mesh']['vettore']['median']:.0f} byte del vettore; sottocampione "
           f"{out['byte_per_mesh']['sottocampione'] / 2 ** 20:.2f} MiB."]
    out["md"] = md
    lc.atomic_json(lc.OUT / "costo" / "costo.json", out)
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
