#!/usr/bin/env python3
"""Calibrazione della scala di d_P sugli held-out SINTETICI del training (factorized_protocol emendamento 4, sez. 1).

    v3_work/trainer/ablations/c3f/calib_heldout.sbatch             (tutto: stage, operatori, embedding, calib)
    aau/run.sh v3_work/trainer/tools/fact_calib.py stage --in-dir <tmp>/in --table <ops>/scale_table.npz
    aau/run.sh v3_work/trainer/tools/fact_calib.py ckpt <chiave>    (stampa il checkpoint)
    aau/run.sh v3_work/trainer/tools/fact_calib.py calib

Held-out = ``heldout`` dello split (uguale per C3M, split_scale_all.json, e C3F, c3f/split.json: si verifica), 100
soggetti per dominio (bfm, ict, gnm; ``rng(1234)``), etichette down8k, noisy, original, remesh, up60k delle sorgenti
della spec di C3M, geometria GREZZA (come build_scale_table.py) e tabella di scala area_mm2 = u_d^2 area(grezza).
Operatori ed embedding come i domini di test (areanorm k_eig 128, eval_v3 + zs_embed, WBES_V3_FACTORIZED_OUT=full).
Coppie: stesso dominio, soggetti diversi, etichette diverse. d_P,modello = ||u_i - u_j|| x dp_per_unit
(eval_factorized.dp_from_ckpt), d_P,GT = GT-SR di training (c3f/gt_sr.npz, ritaglio identico di gt_sr_bfm_ict_gnm)
x dP_per_unit. c = mediana(d_P,GT) / mediana(d_P,modello); sensibilita' c_LS = sum(g m) / sum(m^2); c per dominio.
Uscita: aau/runs/evidence/trainer_v3/factorized_calibration.csv (una riga per checkpoint con embedding held-out).
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
    a = ap.parse_args()
    if a.cmd == "stage":
        stage(a.in_dir, a.table)
    elif a.cmd == "ckpt":
        print(checkpoint(a.key))
    elif a.cmd == "keys":
        print(" ".join(CKPTS))
    else:
        calib()


if __name__ == "__main__":
    main()
