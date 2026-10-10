#!/usr/bin/env python3
"""Calibrazione della scala di d_P sugli held-out SINTETICI del training (factorized_protocol emendamento 4, sez. 1).

    v3_work/trainer/ablations/c3f/calib_heldout.sbatch             (tutto: stage, operatori, embedding, calib)
    aau/run.sh v3_work/trainer/tools/fact_calib.py stage --in-dir <tmp>/in --table <ops>/scale_table.npz
    aau/run.sh v3_work/trainer/tools/fact_calib.py ckpt <chiave>    (stampa il checkpoint)
    aau/run.sh v3_work/trainer/tools/fact_calib.py calib
    v3_work/trainer/ablations/c3f/calib_heldout_bl.sbatch             (emendamento 5: stage, ICP e NICP cs, calib-bl)
    aau/outlineB/run_o3d.sh v3_work/trainer/tools/fact_calib.py bl --in-dir <tmp>/in --table <tmp>/scale_table.npz
    aau/outlineB/run_o3d.sh v3_work/trainer/tools/fact_calib.py calib-bl

Held-out = ``heldout`` dello split (uguale per C3M, split_scale_all.json, e C3F, c3f/split.json: si verifica), 100
soggetti per dominio (bfm, ict, gnm; ``rng(1234)``), etichette down8k, noisy, original, remesh, up60k delle sorgenti
della spec di C3M, geometria GREZZA (come build_scale_table.py) e tabella di scala area_mm2 = u_d^2 area(grezza).
Operatori ed embedding come i domini di test (areanorm k_eig 128, eval_v3 + zs_embed, WBES_V3_FACTORIZED_OUT=full).
Coppie: stesso dominio, soggetti diversi, etichette diverse. d_P,modello = ||u_i - u_j|| x dp_per_unit
(eval_factorized.dp_from_ckpt), d_P,GT = GT-SR di training (c3f/gt_sr.npz, ritaglio identico di gt_sr_bfm_ict_gnm)
x dP_per_unit. c = mediana(d_P,GT) / mediana(d_P,modello); sensibilita' c_LS = sum(g m) / sum(m^2); c per dominio.
Uscita: aau/runs/evidence/trainer_v3/factorized_calibration.csv (una riga per checkpoint con embedding held-out).

Emendamento 5 (``bl``, ``calib-bl``): ICP + Chamfer e NICP per coppia in modo cs (``aau/baselines_mm/blmm.pair_metrics``)
sulle STESSE mesh e coppie (X = indice minore nell'ordine dei nomi, seme = indice della coppia; NICP su 6000 coppie
``rng(1234)``), coordinate di ``blmm.work_coords`` in modo cs con L_d e CS_ref,d delle original held-out del dominio
(definizione di blmm_scalars.py) e frame di E12; d_P = distanza x L_d / CS_ref,d (a centroid size 1).
k = mediana(d_P,GT) / mediana(d_P,baseline), k_LS come c. Uscita: factorized_calibration_bl.csv (icp_cs, nicp_cs e
icp_cs_sub, ICP sul sottoinsieme del NICP), coppie in factorized/calib_heldout_bl.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
TRAINER = THIS.parent
REPO = TRAINER.parents[1]
for _p in (TRAINER, THIS, REPO / "aau/data_scale"):
    sys.path.insert(0, str(_p))
sys.path.append(str(REPO / "aau/baselines_mm"))        # blmm (emendamento 5), in coda: nessun ``common`` ombreggiato

EV = REPO / "aau/runs/evidence/trainer_v3"
RUNS = EV / "ablations/c3f_runs"
C3F = EV / "ablations/c3f"
SPEC = EV / "factorized/c3m/spec.json"                 # sorgenti di tutte le mesh (BFM, ICT, GNM)
SPLITS = (REPO / "aau/data_scale/split_scale_all.json", C3F / "split.json")
GT_SR = C3F / "gt_sr.npz"
OUT_EMB = EV / "factorized/calib_heldout"
OUT_CSV = EV / "factorized_calibration.csv"
LABELS = ("down8k", "noisy", "original", "remesh", "up60k")
DOMS = ("bfm", "ict", "gnm")
N_PER_DOM, SEED = 100, 1234
# chiave -> (run dir, epoca); C3F all'ultimo checkpoint (21.096 passi), C3M ai due checkpoint valutati
# emendamento 5: ICP e NICP per coppia cs sugli held-out, k delle composizioni
OUT_BL = EV / "factorized/calib_heldout_bl"
OUT_BL_CSV = EV / "factorized_calibration_bl.csv"
OPS_TABLE = REPO / "datasets/V3_OPS_CACHE/heldout_calib/scale_table.npz"   # tabella dello stage di calib_heldout.sbatch
FRAMES_E12 = REPO / "aau/runs/evidence/e12/frames.json"
N_NICP = 6000
BL_STEPS = {"fast": "rigid_icp_chamfer", "nicp": "nicp_p2tri"}
CKPTS = {f"{a}_s{s}": (RUNS / (a + ("" if s == 1234 else f"_s{s}")), "072")
         for a in ("factorized", "factorized2", "dual") for s in (1234, 2345)}
CKPTS.update({f"factorizedc3m_e{e}": (EV / "factorized/c3m/runs", e) for e in ("123", "205")})


def checkpoint(key: str) -> Path:
    root, e = CKPTS[key]
    hits = sorted(root.glob(f"v3_*/checkpoints/epoch{e}_ema.pth"))
    if not hits:
        raise SystemExit(f"{key}: nessun epoch{e}_ema.pth in {root}")
    return hits[0]


def heldout_subjects() -> dict:
    """{dominio: soggetti scelti} dagli held-out (identici nei due split)."""
    from common import domain_of
    held = [sorted(json.loads(p.read_text())["heldout"]) for p in SPLITS]
    if held[0] != held[1]:
        raise SystemExit(f"held-out diversi fra {SPLITS[0]} e {SPLITS[1]}")
    out = {}
    for d in DOMS:
        ids = sorted(s for s in held[0] if domain_of(s) == d)
        pick = ids if len(ids) <= N_PER_DOM else np.random.default_rng(SEED).choice(ids, N_PER_DOM, replace=False)
        out[d] = sorted(str(s) for s in pick)
    return out


def stage(in_dir: Path, table: Path) -> None:
    """Mesh grezze {V, F} in ``in_dir`` (nomi della spec) e la tabella di scala (formato build_scale_table)."""
    import build_scale_table as bst
    import data_v3 as dv
    import global_v3
    from cache_budget import load_index, read_member
    from common import domain_of, split_name
    subj = heldout_subjects()
    want = {s for v in subj.values() for s in v}
    spec = json.loads(SPEC.read_text())
    sources = dv.collect_sources(spec)
    names = sorted(n for n in sources if split_name(n)[0] in want and split_name(n)[1] in LABELS)
    got = {(split_name(n)[0]) for n in names}
    if got != want:
        raise SystemExit(f"soggetti senza mesh nelle sorgenti: {sorted(want - got)[:5]}")
    tf = {**json.loads(bst.FRAMES.read_text())["domains"], **global_v3.EXTRA_FRAMES}
    idx = load_index(Path(spec["tar_index"]))
    pos = {str(n): i for i, n in enumerate(idx["names"])}
    handles: dict = {}
    in_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for n in names:
            kind, src = sources[n]
            if kind == "tar":
                with np.load(io.BytesIO(read_member(idx, pos[n], handles))) as z:
                    V, F = bst._vf(z)
            else:
                raw, _ = bst.raw_of_view(Path(src))
                with np.load(raw) as z:
                    V, F = bst._vf(z)
            np.savez(in_dir / n, V=V, F=F)
            d = domain_of(split_name(n)[0])
            rows.append((n, d, float(tf[d]["u"]) ** 2 * bst.total_area(V, F), bst.total_area(V, F),
                         float(np.abs(V - V.mean(0)).max())))
    finally:
        for fh in handles.values():
            fh.close()
    table.parent.mkdir(parents=True, exist_ok=True)
    np.savez(table, names=np.asarray([r[0] for r in rows]), domain=np.asarray([r[1] for r in rows]),
             area_mm2=np.asarray([r[2] for r in rows]), area_raw=np.asarray([r[3] for r in rows]),
             maxabs_raw=np.asarray([r[4] for r in rows]), check=np.full(len(rows), np.nan))
    table.with_suffix(".json").write_text(json.dumps(
        {"definition": "fact_calib.py stage: area_mm2 = u_d^2 * area della geometria grezza (build_scale_table.py)",
         "spec": str(SPEC), "labels": LABELS, "seed": SEED, "n_per_domain": N_PER_DOM, "subjects": subj,
         "n_meshes": len(rows), "by_domain": {d: sum(r[1] == d for r in rows) for d in DOMS}}, indent=1) + "\n")
    print(f"[calib] {len(rows)} mesh di {len(want)} soggetti in {in_dir}, tabella {table}", flush=True)


def calib_one(key: str, emb: Path, G: np.ndarray, gpos: dict, dpu_gt: float) -> dict:
    import eval_factorized as ef
    from common import domain_of
    with np.load(emb, allow_pickle=True) as z:
        Z = np.asarray(z["Z"], np.float64)
        subj = [str(s) for s in z["subjects"]]
        topo = np.asarray([str(t) for t in z["topologies"]])
        ckpt = Path(str(z["checkpoint"]))
    if ckpt.resolve() != checkpoint(key).resolve():
        raise SystemExit(f"{key}: embedding di {ckpt}, atteso {checkpoint(key)}")
    import torch
    head = torch.load(ckpt, map_location="cpu", weights_only=False)["args"].get("head", "embed")
    u = Z[:, Z.shape[1] // 2:] if head == "dual" else Z[:, 1:]
    dpu = ef.dp_from_ckpt(ckpt)
    dom = np.asarray([domain_of(s) for s in subj])
    g = np.asarray([gpos[s] for s in subj])
    i, j = np.triu_indices(len(Z), 1)
    keep = (dom[i] == dom[j]) & (g[i] != g[j]) & (topo[i] != topo[j])
    i, j = i[keep], j[keep]
    m = np.linalg.norm(u[i] - u[j], axis=1) * dpu
    t = G[g[i], g[j]].astype(np.float64) * dpu_gt
    out = {"key": key, "head": head, "checkpoint": str(ckpt), "dp_per_unit": dpu, "n_meshes": len(Z),
           "n_pairs": int(len(i)), "median_dP_gt": float(np.median(t)), "median_dP_model": float(np.median(m)),
           "c_median": float(np.median(t) / np.median(m)), "c_ls": float((t * m).sum() / (m * m).sum())}
    for d in DOMS:
        k = dom[i] == d
        out[f"c_median_{d}"] = float(np.median(t[k]) / np.median(m[k])) if k.any() else float("nan")
    return out


def calib() -> None:
    with np.load(GT_SR, allow_pickle=True) as z:
        G = np.asarray(z["D_orig"], np.float32)
        gpos = {str(n): k for k, n in enumerate(z["names"])}
    dpu_gt = float(json.loads(GT_SR.with_suffix(".json").read_text())["dP_per_unit"])
    rows = []
    for key in CKPTS:
        emb = OUT_EMB / key / "embeddings.npz"
        if emb.exists():
            r = calib_one(key, emb, G, gpos, dpu_gt)
            rows.append(r)
            print(f"[calib] {key}: c {r['c_median']:.4f} (LS {r['c_ls']:.4f}; bfm {r['c_median_bfm']:.3f} ict "
                  f"{r['c_median_ict']:.3f} gnm {r['c_median_gnm']:.3f}), {r['n_pairs']} coppie", flush=True)
        else:
            print(f"[calib] {key}: embedding held-out assenti ({emb})", flush=True)
    if rows:
        with open(OUT_CSV, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"[calib] {len(rows)} righe -> {OUT_CSV}", flush=True)


def load() -> dict:
    """{chiave: riga} di factorized_calibration.csv (vuoto se assente); chiave come CKPTS."""
    if not OUT_CSV.exists():
        return {}
    return {r["key"]: {k: (v if k in ("key", "head", "checkpoint") else float(v)) for k, v in r.items()}
            for r in csv.DictReader(open(OUT_CSV))}


# ------------------------------------------------------------------------------- emendamento 5: baseline cs

def heldout_pairs(names: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(i, j, dominio) delle coppie di calib_one sui nomi ordinati della tabella: i < j, stesso dominio, soggetti
    ed etichette diversi."""
    with np.load(OPS_TABLE) as z:
        dmap = dict(zip([str(x) for x in z["names"]], [str(x) for x in z["domain"]]))
    sid = np.asarray([n[:-4].split("_GTready_", 1)[0] for n in names])
    lab = np.asarray([n[:-4].split("_GTready_", 1)[1] for n in names])
    dom = np.asarray([dmap[n] for n in names])
    i, j = np.triu_indices(len(names), 1)
    keep = (dom[i] == dom[j]) & (sid[i] != sid[j]) & (lab[i] != lab[j])
    return i[keep], j[keep], dom


_C: dict = {}          # nome -> (coordinate cs, offset); ereditato dai worker (fork)


def mesh_mm(path: Path, dom: str) -> tuple[np.ndarray, np.ndarray]:
    """(V in mm nel frame di E12 del dominio, F) di una mesh messa in scena da ``stage`` (come blmm.to_mm)."""
    import blmm
    V, F = blmm.load_raw(path)
    f = json.loads(FRAMES_E12.read_text())["domains"][dom]
    return f["u"] * V @ np.asarray(f["R"]).T + np.asarray(f["t"]), F


def _scalars(task):
    import blmm
    return blmm.mesh_scalars(*mesh_mm(Path(task[0]), task[1]))


def _bl_chunk(task):
    import blmm
    step, items = task
    return [blmm.pair_metrics(_C[a], _C[b], q, "cs", step)[BL_STEPS[step]] for a, b, q in items]


def baselines(in_dir: Path, table: Path, workers: int, steps: list) -> None:
    """ICP + Chamfer (passo fast, tutte le coppie) e NICP per coppia (passo nicp, N_NICP coppie) in modo cs sugli
    held-out messi in scena da ``stage``; scalari per mesh e L_d, CS_ref,d in OUT_BL."""
    import multiprocessing as mp
    import time
    import blmm
    with np.load(table) as z:
        names = [str(x) for x in z["names"]]
    with np.load(OPS_TABLE) as z:
        if sorted(str(x) for x in z["names"]) != sorted(names):
            raise SystemExit(f"{table}: mesh diverse da quelle di {OPS_TABLE}")
    order = np.argsort(names)
    names = [names[k] for k in order]
    i, j, dom = heldout_pairs(names)
    OUT_BL.mkdir(parents=True, exist_ok=True)
    sc_path = OUT_BL / "scalars.npz"
    if sc_path.exists():
        with np.load(sc_path) as z:
            if [str(x) for x in z["names"]] != names:
                raise SystemExit(f"{sc_path}: nomi diversi")
            sc = {k: z[k] for k in z.files if k not in ("names", "domain")}
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            res = pool.map(_scalars, [(str(in_dir / n), d) for n, d in zip(names, dom)], chunksize=2)
        sc = {k: np.asarray([r[k] for r in res]) for k in res[0]}
        blmm.atomic_savez(sc_path, names=np.asarray(names), domain=dom, **sc)
    orig = np.asarray([n.endswith("_GTready_original.npz") for n in names])
    prm = {d: {"L": float(np.median(sc["maxabs"][orig & (dom == d)])),
               "cs_ref": float(np.median(sc["cs"][orig & (dom == d)])), "n_ref": int((orig & (dom == d)).sum())}
           for d in DOMS}
    (OUT_BL / "params.json").write_text(json.dumps(
        {"definition": "blmm_scalars.py sulle original held-out del dominio: L = mediana di maxabs, cs_ref = mediana "
                       "di cs (mm, frame di E12)", "frames": str(FRAMES_E12), "domains": prm}, indent=1) + "\n")
    print(f"[calib-bl] {len(names)} mesh, {len(i)} coppie, parametri {prm}", flush=True)
    for k, (n, d) in enumerate(zip(names, dom)):
        Vmm, _ = mesh_mm(in_dir / n, d)
        # blmm.work_coords, modo cs
        _C[n] = ((Vmm - Vmm.mean(0)) / prm[d]["L"] * (prm[d]["cs_ref"] / sc["cs"][k]), np.zeros(3))
    with mp.get_context("fork").Pool(workers) as pool:
        for step in steps:
            q = np.arange(len(i)) if step == "fast" else \
                np.sort(np.random.default_rng(SEED).choice(len(i), N_NICP, replace=False))
            items = [(names[i[k]], names[j[k]], int(k)) for k in q]
            chunk = 16 if step == "fast" else 4
            t0 = time.time()
            res = pool.map(_bl_chunk, [(step, items[s:s + chunk]) for s in range(0, len(items), chunk)], chunksize=1)
            x = np.asarray([v for block in res for v in block], np.float64)
            unit = np.asarray([prm[d]["L"] / prm[d]["cs_ref"] for d in dom[i[q]]])
            blmm.atomic_savez(OUT_BL / f"{step}.npz", names=np.asarray(names), q=q, i=i[q], j=j[q], dP=x * unit,
                              metric=BL_STEPS[step], n_failed=int((~np.isfinite(x)).sum()))
            print(f"[calib-bl] {step}: {len(q)} coppie in {time.time() - t0:.0f}s, fallite {int((~np.isfinite(x)).sum())}",
                  flush=True)


def k_of(t: np.ndarray, m: np.ndarray, dom: np.ndarray) -> dict:
    """k come c: rapporto delle mediane (primaria), minimi quadrati senza intercetta, per dominio."""
    ok = np.isfinite(m)
    t, m, dom = t[ok], m[ok], dom[ok]
    out = {"n_pairs": int(len(m)), "n_failed": int((~ok).sum()), "median_dP_gt": float(np.median(t)),
           "median_dP_bl": float(np.median(m)), "k_median": float(np.median(t) / np.median(m)),
           "k_ls": float((t * m).sum() / (m * m).sum())}
    for d in DOMS:
        out[f"k_median_{d}"] = float(np.median(t[dom == d]) / np.median(m[dom == d])) if (dom == d).any() else float("nan")
    return out


def calib_bl() -> None:
    with np.load(GT_SR, allow_pickle=True) as z:
        G = np.asarray(z["D_orig"], np.float32)
        gpos = {str(n): k for k, n in enumerate(z["names"])}
    dpu_gt = float(json.loads(GT_SR.with_suffix(".json").read_text())["dP_per_unit"])
    rows, fast = [], None
    for step, key in (("fast", "icp_cs"), ("nicp", "nicp_cs")):
        p = OUT_BL / f"{step}.npz"
        if not p.exists():
            print(f"[calib-bl] {key}: {p} assente", flush=True)
            continue
        with np.load(p) as z:
            names, q, i, j, m = [str(x) for x in z["names"]], z["q"], z["i"], z["j"], np.asarray(z["dP"], np.float64)
        g = np.asarray([gpos[n[:-4].split("_GTready_", 1)[0]] for n in names])
        dom = heldout_pairs(names)[2]
        t = G[g[i], g[j]].astype(np.float64) * dpu_gt
        rows.append({"key": key, "metric": BL_STEPS[step], **k_of(t, m, dom[i])})
        if step == "fast":
            fast = dict(zip(q.tolist(), m))
        elif fast is not None:
            rows.append({"key": "icp_cs_sub", "metric": BL_STEPS["fast"],
                         **k_of(t, np.asarray([fast[k] for k in q.tolist()]), dom[i])})
    for r in rows:
        print(f"[calib-bl] {r['key']}: k {r['k_median']:.4f} (LS {r['k_ls']:.4f}; bfm {r['k_median_bfm']:.3f} ict "
              f"{r['k_median_ict']:.3f} gnm {r['k_median_gnm']:.3f}), {r['n_pairs']} coppie, fallite {r['n_failed']}",
              flush=True)
    if rows:
        with open(OUT_BL_CSV, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"[calib-bl] {len(rows)} righe -> {OUT_BL_CSV}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--in-dir", type=Path, required=True)
    s.add_argument("--table", type=Path, required=True)
    c = sub.add_parser("ckpt")
    c.add_argument("key", choices=list(CKPTS))
    sub.add_parser("calib")
    sub.add_parser("keys")
    b = sub.add_parser("bl")
    b.add_argument("--in-dir", type=Path, required=True)
    b.add_argument("--table", type=Path, required=True)
    b.add_argument("--workers", type=int, default=32)
    b.add_argument("--steps", default="fast,nicp")
    sub.add_parser("calib-bl")
    a = ap.parse_args()
    if a.cmd == "stage":
        stage(a.in_dir, a.table)
    elif a.cmd == "bl":
        baselines(a.in_dir, a.table, a.workers, a.steps.split(","))
    elif a.cmd == "calib-bl":
        calib_bl()
    elif a.cmd == "ckpt":
        print(checkpoint(a.key))
    elif a.cmd == "keys":
        print(" ".join(CKPTS))
    else:
        calib()


if __name__ == "__main__":
    main()
