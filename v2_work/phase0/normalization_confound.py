#!/usr/bin/env python
"""Fase 1a: quantify how much of the v1 cross-topology Chamfer degradation is a
normalization confound.

For the same pair sets as v1 (table1_pairlevel_exact), recompute raw symmetric
Chamfer under two per-mesh normalizations:
  - maxabs : center on vertex mean, divide by max|coord|   (v1 benchmark)
  - area   : center on area-weighted centroid, divide by sqrt(total area)

Report Spearman vs GT for both, per topology pair and overall.
Subset of subjects to keep CPU cost sane (--n-subjects).

Output: v2_work/phase0/normalization_confound.csv (+ printed table)
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
PAIR_TABLE_ROOT = REPO_ROOT / "paper_artifacts" / "bootstrap_ci" / "table1_pairlevel_exact"
MESH_ROOT = REPO_ROOT / "datasets" / "REMESH" / "npz_data_topo_500"


def load_mesh(path: Path):
    with np.load(path) as d:
        V = (d["verts"] if "verts" in d else d["V"]).astype(np.float64)
        F = (d["faces"] if "faces" in d else d["F"]).astype(np.int64)
    return V, F


def normalize_maxabs(V, F):
    Vn = V - V.mean(axis=0, keepdims=True)
    s = np.max(np.abs(Vn))
    return Vn / max(s, 1e-9)


def normalize_area(V, F):
    tri = V[F]                                   # (M,3,3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(n, axis=1)      # (M,)
    cent = tri.mean(axis=1)                      # (M,3)
    total = max(float(areas.sum()), 1e-12)
    c = (cent * areas[:, None]).sum(axis=0) / total
    Vn = V - c[None, :]
    return Vn / np.sqrt(total)


def subsample(V, n, seed):
    if len(V) <= n:
        return V
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(V), size=n, replace=False)
    return V[np.sort(idx)]


def chamfer(X: np.ndarray, Y: np.ndarray) -> float:
    Xt = torch.from_numpy(np.ascontiguousarray(X)).float()
    Yt = torch.from_numpy(np.ascontiguousarray(Y)).float()
    d2 = torch.cdist(Xt, Yt) ** 2
    return float(d2.min(dim=1).values.mean() + d2.min(dim=0).values.mean())


def spearman(x, y):
    from scipy.stats import spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[m], y[m]).statistic) if m.sum() >= 3 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subjects", type=int, default=30)
    ap.add_argument("--n-points", type=int, default=4096)
    ap.add_argument("--out", default=str(THIS_DIR / "normalization_confound.csv"))
    args = ap.parse_args()

    # normalized mesh cache: (mesh_name, norm) -> points
    cache: dict[tuple[str, str], np.ndarray] = {}

    def points(name: str, norm: str) -> np.ndarray:
        key = (name, norm)
        if key not in cache:
            V, F = load_mesh(MESH_ROOT / f"{name}.npz")
            Vn = normalize_maxabs(V, F) if norm == "maxabs" else normalize_area(V, F)
            cache[key] = subsample(Vn, args.n_points, seed=hash(name) % 2**31)
        return cache[key]

    rows_out = []
    agg = {"gt": [], "maxabs": [], "area": [], "latent": []}
    t0 = time.time()
    for d in sorted(PAIR_TABLE_ROOT.iterdir()):
        f = d / "pair_metrics.csv"
        if not f.exists():
            continue
        with open(f, newline="") as fh:
            rows = list(csv.DictReader(fh))
        subj = sorted({r["subject_a"] for r in rows} | {r["subject_b"] for r in rows})[: args.n_subjects]
        keep = set(subj)
        rows = [r for r in rows if r["subject_a"] in keep and r["subject_b"] in keep]

        gt, ch_max, ch_area, lat = [], [], [], []
        for r in rows:
            na = f"{r['subject_a']}_GTready_{r['topology_a']}"
            nb = f"{r['subject_b']}_GTready_{r['topology_b']}"
            gt.append(float(r["gt_distance"]))
            lat.append(float(r["latent_distance"]))
            ch_max.append(chamfer(points(na, "maxabs"), points(nb, "maxabs")))
            ch_area.append(chamfer(points(na, "area"), points(nb, "area")))

        row = {
            "pair_label": d.name,
            "n_pairs": len(rows),
            "spearman_chamfer_maxabs": spearman(gt, ch_max),
            "spearman_chamfer_area": spearman(gt, ch_area),
            "spearman_latent": spearman(gt, lat),
        }
        rows_out.append(row)
        agg["gt"] += gt; agg["maxabs"] += ch_max; agg["area"] += ch_area; agg["latent"] += lat
        print(f"{d.name:24s} n={len(rows):5d} maxabs={row['spearman_chamfer_maxabs']:+.3f} "
              f"area={row['spearman_chamfer_area']:+.3f} latent={row['spearman_latent']:+.3f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    overall = {
        "pair_label": "OVERALL",
        "n_pairs": len(agg["gt"]),
        "spearman_chamfer_maxabs": spearman(agg["gt"], agg["maxabs"]),
        "spearman_chamfer_area": spearman(agg["gt"], agg["area"]),
        "spearman_latent": spearman(agg["gt"], agg["latent"]),
    }
    rows_out.insert(0, overall)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    print("\nOVERALL:", overall)


if __name__ == "__main__":
    main()
