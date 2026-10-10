#!/usr/bin/env python3
"""Delta appaiati dei bracci di factorized_protocol.md contro le baseline in mm, sulle STESSE righe e repliche.

    v3_work/unified_gt/run.sh v3_work/trainer/tools/fact_paired.py [--workers 32]

Righe, GT e seme per (dominio, gruppo): quelli di aau/baselines_mm/blmm_eval.py (importato: righe di E12, colonne
delle baseline, GT FR / SR / maxabs, seme della differenza pubblicata e108 - Chamfer eval), gruppo primario del
dominio (HIFI3D e dev FaceScape ``nocrop_cross``, FaceVerse ``mesh_pair_nocrop``). Si aggiungono le colonne dei bracci
dagli embedding del passo ``form`` (ultimo checkpoint EMA, 21.096 passi): d_F e d_P per i fattorizzati, ||z|| per
ctrlfr (d_F e d_P anche per ``dual``: z_F e u). Delta (braccio - baseline) contro: ICP + Chamfer in mm, NICP su
template in mm, taglia stimata dalla mesh (centroid size robusta, non oracolo), NICP per coppia in modo cs, e108;
con FR e SR. Maschera COMUNE per dominio: le righe con tutte le distanze finite (NICP per coppia fallisce su alcune
coppie con up60k), quindi i punti delle baseline possono differire di poco da quelli pubblicati. IC 95% percentile,
P(delta <= 0). Uscita: aau/runs/evidence/trainer_v3/factorized_paired.csv.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve().parent
TRAINER = THIS.parent
REPO = TRAINER.parents[1]
sys.path.insert(0, str(REPO / "aau/baselines_mm"))

import blmm_eval as be  # noqa: E402  (sola lettura)

# NIENTE moduli del trainer: blmm_eval importa aau/baselines/common.py, che ha lo stesso nome di
# v3_work/trainer/common.py. Distanze e percorsi come factorized_v3.model_distances e fact_summary (copiati).
EVAL = REPO / "aau/runs/evidence/trainer_v3/ablations/c3f_eval"
ARMS, SEEDS = ("factorized", "factorized2", "ctrlfr", "dual"), (1234, 2345)
FORM_DIR = {"hifi": "form_hifi", "devfs": "form_devfs", "fv": "form_fv_expr"}


def embeddings(dom: str, arm: str, seed: int, e: str):
    v = arm + ("" if seed == 1234 else f"s{seed}")
    hits = sorted((EVAL / FORM_DIR[dom]).glob(f"data_*/scale_v3{v}fulle{e}*/zs_zeroshot/embeddings.npz"))
    return hits[0] if hits else None


def distances(Z, i, j, ckpt: Path) -> dict:
    """factorized_v3.model_distances: fattorizzati -> d_F e d_P; dual -> z_F e u; altrimenti z."""
    import json
    import torch
    args = torch.load(ckpt, map_location="cpu", weights_only=False)["args"]
    head = args.get("head", "embed")
    if head in ("factorized", "factorized2"):
        side = json.loads(Path(args["dist_npz"]).with_suffix(".json").read_text())
        dpu = float(side.get("dp_per_unit", side.get("dP_per_unit"))) / float(args.get("gt_scale", 1.0))
        S = np.exp(Z[:, 0])
        dP = np.linalg.norm(Z[i, 1:] - Z[j, 1:], axis=1) * dpu
        return {"form": np.sqrt((S[i] - S[j]) ** 2 + S[i] * S[j] * dP ** 2), "shape": dP}
    if head == "dual":
        L = Z.shape[1] // 2
        return {"zf": np.linalg.norm(Z[i, :L] - Z[j, :L], axis=1), "u": np.linalg.norm(Z[i, L:] - Z[j, L:], axis=1)}
    return {"z": np.linalg.norm(Z[i] - Z[j], axis=1)}

VIEWS = {"hifi3d": ("hifi", "nocrop_cross"), "faceverse": ("fv", "mesh_pair_nocrop"), "facescape": ("devfs", "nocrop_cross")}
BASELINES = {"mm_rigid_icp_chamfer": "ICP + Chamfer in mm", "mm_nicp_template": "NICP su template in mm",
             "est_cs": "taglia stimata", "cs_nicp_p2tri": "NICP per coppia (cs)", "scale_e108": "e108"}
GTS = ("fr", "sr")
_J: dict = {}


def rows_for(view: str):
    """Righe del gruppo primario con le colonne delle baseline e delle GT (come blmm_eval.frames_for, topologie tenute)."""
    if view == "hifi3d":
        fr = be.e12m.hifi_frames()[0]
        base_df = fr["all_cross"][0]
    elif view == "faceverse":
        fr = be.e12m.fv_frames()
        base_df = fr["mesh_pair_nocrop"][0]
    else:
        fr = be.e12m.fs_frames()
        base_df = fr["all_cross"][0]
    group = VIEWS[view][1]
    idx = be.zes.Index(be.blmm.subjects(view))
    pub = [m for m in list(be.PUBLISHED) + list(be.REFS) if m in base_df]
    D, _ = be.view_distances(view, idx, set(pub))
    df = be.add_gts(be.add_columns(base_df, D, idx), be.GT_SET[view])
    df = df.rename(columns={k: f"{v[0]}_{v[1]}" for k, v in be.PUBLISHED.items() if k in df})
    df = df[df["topology_a"].ne("crop") & df["topology_b"].ne("crop")].reset_index(drop=True)
    if len(df) != len(fr[group][0]):
        raise SystemExit(f"{view}: {len(df)} righe invece delle {len(fr[group][0])} di E12")
    return df, idx, fr[group][2]


def model_columns(view: str, idx) -> dict:
    """{colonna: D (n, n) sulle chiavi di idx} per ogni braccio e seme, ultimo checkpoint."""
    dom = VIEWS[view][0]
    out = {}
    for arm in ARMS:
        for seed in SEEDS:
            p = embeddings(dom, arm, seed, "072")
            if p is None:
                continue
            with np.load(p, allow_pickle=True) as z:
                Z = np.asarray(z["Z"], np.float64)
                keys = list(zip([str(x) for x in z["subjects"]], [str(x) for x in z["topologies"]]))
                ckpt = Path(str(z["checkpoint"]))
            pos = {k: r for r, k in enumerate(keys)}
            Z = Z[[pos[k] for k in idx.keys]]
            n = len(Z)
            i, j = (a.ravel() for a in np.meshgrid(np.arange(n), np.arange(n), indexing="ij"))
            for k, d in distances(Z, i, j, ckpt).items():
                out[f"{arm}_s{seed}|{k}"] = d.reshape(n, n)
    return out


def _rep(k: int) -> dict:
    from scipy.stats import spearmanr
    J = _J
    c = J["counts"][k]
    wt = c[J["sa"]].astype(np.int64) * c[J["sb"]]
    keep = wt > 0
    w = wt[keep]
    out = {}
    for g in GTS:
        gt = np.repeat(J["cols"][f"gt_{g}"][keep], w)
        for m in J["methods"]:
            out[(g, m)] = spearmanr(gt, np.repeat(J["cols"][m][keep], w)).correlation
    return out


def main() -> None:
    global _J
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    recs = []
    for view in VIEWS:
        df, idx, seed = rows_for(view)
        M = model_columns(view, idx)
        if not M:
            print(f"[paired] {view}: nessun embedding dei bracci", flush=True)
            continue
        df = be.add_columns(df, M, idx)
        bl = [b for b in BASELINES if b in df]
        methods = list(M) + bl
        cols = {m: df[m].to_numpy(np.float64) for m in methods}
        cols.update({f"gt_{g}": df[f"gt_{g}"].to_numpy(np.float64) for g in GTS})
        subjects = np.array(sorted(set(df["subject_a"]) | set(df["subject_b"])))
        s2i = {s: i for i, s in enumerate(subjects)}
        sa, sb = df["subject_a"].map(s2i).to_numpy(), df["subject_b"].map(s2i).to_numpy()
        mask = (sa != sb) & np.all([np.isfinite(v) for v in cols.values()], axis=0)
        rng = np.random.default_rng(seed)
        counts = [np.ones(len(subjects), dtype=np.int64)] + \
            [np.bincount(rng.integers(0, len(subjects), len(subjects)), minlength=len(subjects)) for _ in range(a.n_boot)]
        _J = {"counts": counts, "sa": sa[mask], "sb": sb[mask], "cols": {k: v[mask] for k, v in cols.items()},
              "methods": methods}
        with mp.get_context("fork").Pool(a.workers) as pool:
            reps = pool.map(_rep, range(a.n_boot + 1), chunksize=4)
        V = {key: np.array([r[key] for r in reps]) for key in reps[0]}
        for g in GTS:
            for m in M:
                for b in bl:
                    d = V[(g, m)] - V[(g, b)]
                    lo, hi = np.percentile(d[1:], [2.5, 97.5])
                    recs.append({"domain": view, "group": VIEWS[view][1], "gt": g, "arm": m.split("|")[0],
                                 "distance": m.split("|")[1], "baseline": BASELINES[b], "arm_point": V[(g, m)][0],
                                 "baseline_point": V[(g, b)][0], "delta": d[0], "ci_low": lo, "ci_high": hi,
                                 "p_le0": float((d[1:] <= 0).mean()), "n_rows": int(mask.sum()), "seed": int(seed)})
        print(f"[paired] {view}: {len(M)} colonne dei bracci, {int(mask.sum())} righe", flush=True)
    out = REPO / "aau/runs/evidence/trainer_v3/factorized_paired.csv"
    pd.DataFrame(recs).to_csv(out, index=False)
    print(f"[paired] {len(recs)} delta -> {out}", flush=True)


if __name__ == "__main__":
    main()
