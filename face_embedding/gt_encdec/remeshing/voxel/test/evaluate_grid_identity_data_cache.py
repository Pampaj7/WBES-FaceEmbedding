#!/usr/bin/env python3
"""
STEP 2 — Dataset-wide evaluation using precomputed grid embeddings.

Input:
  grid_cache/
    subject/
      original.npz
      remesh.npz
      crop.npz
      noisy.npz

Output:
  grid_identity_results.csv
"""

import csv
import random
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

GRID_CACHE = Path("grid_cache")
OUT_CSV = Path("grid_identity_results.csv")

K_NEG = 5                  # negatives per subject
EPS = 1e-8
VARIANTS = ["original", "remesh", "crop", "noisy"]

random.seed(1234)

# -------------------------------------------------
# UTILS
# -------------------------------------------------

def load_grid(subject_id, variant):
    path = GRID_CACHE / subject_id / f"{variant}.npz"
    if not path.exists():
        raise FileNotFoundError
    d = np.load(path)
    G = torch.from_numpy(d["G"]).float()
    M = torch.from_numpy(d["M"]).bool()
    return G, M


def identity_distance(GA, MA, GB, MB):
    valid = MA & MB
    if valid.sum() == 0:
        return None, 0
    diff = GA[valid] - GB[valid]
    return diff.norm(dim=1).mean().item(), int(valid.sum())

# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():
    subjects = sorted([p.name for p in GRID_CACHE.iterdir() if p.is_dir()])
    print(f"📦 Subjects: {len(subjects)}")

    rows = []

    for sid_A in tqdm(subjects, desc="Subjects"):
        # ---------- INTRA ----------
        try:
            G_Ao, M_Ao = load_grid(sid_A, "original")
            G_Ar, M_Ar = load_grid(sid_A, "remesh")
        except FileNotFoundError:
            continue

        d_intra, n_cells = identity_distance(G_Ao, M_Ao, G_Ar, M_Ar)
        if d_intra is None:
            continue

        rows.append([
            sid_A, sid_A,
            "original", "remesh",
            d_intra,
            1.0,
            n_cells
        ])

        # ---------- INTER ----------
        negatives = random.sample(
            [s for s in subjects if s != sid_A],
            min(K_NEG, len(subjects) - 1)
        )

        for sid_B in negatives:
            for var_A, var_B in [
                ("original", "original"),
                ("remesh",   "original"),
                ("original", "remesh"),
                ("crop",     "original"),
                ("original", "crop"),
            ]:
                try:
                    G_A, M_A = load_grid(sid_A, var_A)
                    G_B, M_B = load_grid(sid_B, var_B)
                except FileNotFoundError:
                    continue

                d_inter, n_cells = identity_distance(G_A, M_A, G_B, M_B)
                if d_inter is None:
                    continue

                rows.append([
                    sid_A, sid_B,
                    var_A, var_B,
                    d_inter,
                    d_inter / (d_intra + EPS),
                    n_cells
                ])

    # ---------- SAVE CSV ----------
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "subject_A",
            "subject_B",
            "variant_A",
            "variant_B",
            "distance",
            "ratio",
            "n_cells_common"
        ])
        writer.writerows(rows)

    print(f"\n✅ Saved results to {OUT_CSV.resolve()}")
    print(f"📊 Total comparisons: {len(rows)}")

# -------------------------------------------------
# ENTRY
# -------------------------------------------------

if __name__ == "__main__":
    main()
