#!/usr/bin/env python3
"""
STEP 3 — Per-cell identity analysis.

Input:
  - grid_cache/
  - grid_identity_results.csv (solo per lista soggetti)

Output:
  - cell_identity_scores.csv
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from itertools import combinations

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

GRID_CACHE = Path("grid_cache")
OUT_CSV = Path("cell_identity_scores.csv")

EPS = 1e-8
VARIANTS = ["original", "remesh", "crop", "noisy"]

# -------------------------------------------------
# UTILS
# -------------------------------------------------

def load_grid(subject_id, variant):
    path = GRID_CACHE / subject_id / f"{variant}.npz"
    if not path.exists():
        return None, None
    d = np.load(path)
    G = torch.from_numpy(d["G"]).float()
    M = torch.from_numpy(d["M"]).bool()
    return G, M

# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():
    subjects = sorted([p.name for p in GRID_CACHE.iterdir() if p.is_dir()])
    print(f"📦 Subjects: {len(subjects)}")

    # infer K, D
    ex_subj = subjects[0]
    ex = np.load(next((GRID_CACHE / ex_subj).glob("*.npz")))
    K, D = ex["G"].shape
    print(f"🧊 Grid cells: {K} | Latent dim: {D}")

    # storage
    intra_vals = [[] for _ in range(K)]
    inter_vals = [[] for _ in range(K)]

    # -------------------------------
    # INTRA-SUBJECT (topology)
    # -------------------------------
    print("🔁 Computing INTRA-subject variability...")
    for sid in tqdm(subjects):
        grids = {}
        masks = {}

        for v in VARIANTS:
            G, M = load_grid(sid, v)
            if G is not None:
                grids[v] = G
                masks[v] = M

        if len(grids) < 2:
            continue

        for v1, v2 in combinations(grids.keys(), 2):
            G1, M1 = grids[v1], masks[v1]
            G2, M2 = grids[v2], masks[v2]

            valid = M1 & M2
            if valid.sum() == 0:
                continue

            diff = (G1[valid] - G2[valid]).norm(dim=1) ** 2
            idxs = valid.nonzero(as_tuple=True)[0]

            for i, c in enumerate(idxs):
                intra_vals[c].append(diff[i].item())

    # -------------------------------
    # INTER-SUBJECT (identity)
    # -------------------------------
    print("🔀 Computing INTER-subject variability...")
    for sid_A, sid_B in tqdm(list(combinations(subjects, 2))):
        GA, MA = load_grid(sid_A, "original")
        GB, MB = load_grid(sid_B, "original")

        if GA is None or GB is None:
            continue

        valid = MA & MB
        if valid.sum() == 0:
            continue

        diff = (GA[valid] - GB[valid]).norm(dim=1) ** 2
        idxs = valid.nonzero(as_tuple=True)[0]

        for i, c in enumerate(idxs):
            inter_vals[c].append(diff[i].item())

    # -------------------------------
    # AGGREGATE
    # -------------------------------
    rows = []
    for c in range(K):
        intra = np.array(intra_vals[c])
        inter = np.array(inter_vals[c])

        if len(intra) < 5 or len(inter) < 5:
            continue

        var_intra = intra.mean()
        var_inter = inter.mean()
        score = var_inter / (var_intra + EPS)

        rows.append([
            c,
            var_intra,
            var_inter,
            score,
            len(intra),
            len(inter)
        ])

    df = pd.DataFrame(
        rows,
        columns=[
            "cell_idx",
            "var_intra",
            "var_inter",
            "score",
            "n_intra",
            "n_inter"
        ]
    )

    df.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Saved per-cell analysis to {OUT_CSV.resolve()}")

    # quick summary
    print("\n🔝 Top-10 discriminative cells:")
    print(df.sort_values("score", ascending=False).head(10))

# -------------------------------------------------
# ENTRY
# -------------------------------------------------

if __name__ == "__main__":
    main()
