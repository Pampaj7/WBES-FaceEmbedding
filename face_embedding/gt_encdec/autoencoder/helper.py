# build_gt_distance_npz_normalized.py
# Compute normalized pairwise Euclidean distances for GT meshes
# Uses same normalization as GTReadyDatasetNPZ
# Author: Leonardo Pampaloni — 2025

import os
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

GT_DIR = "../../../datasets/GT_ready/npz_data"
OUT = "latent_analysis/dist_matrices_fields/D_orig_gt_normalized.npz"
BLOCK = 200       # number of meshes per block (controls memory)
DTYPE = np.float32

def load_normalized_verts(path):
    """Load verts from .npz, apply centering and scale normalization."""
    d = np.load(path)
    V = d["verts"].astype(DTYPE)
    V = V - V.mean(axis=0, keepdims=True)
    scale = np.max(np.abs(V))
    if scale > 1e-6:
        V = V / scale
    else:
        V = np.zeros_like(V)
    return V.flatten()

def main():
    names = sorted([f for f in os.listdir(GT_DIR) if f.endswith(".npz")])
    N = len(names)
    print(f"📦 Found {N} meshes — computing NxN distance matrix with normalization (block size {BLOCK}).")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp_path = OUT.replace(".npz", "_tmp.dat")

    # Initialize on-disk memmap matrix
    D = np.memmap(tmp_path, dtype=DTYPE, mode="w+", shape=(N, N))
    D[:] = 0

    # Block-wise computation to avoid RAM overflow
    for i0 in tqdm(range(0, N, BLOCK), desc="Outer blocks"):
        i1 = min(N, i0 + BLOCK)
        X_block = [load_normalized_verts(os.path.join(GT_DIR, names[i])) for i in range(i0, i1)]
        X_block = np.stack(X_block, dtype=DTYPE)

        for j0 in range(i0, N, BLOCK):
            j1 = min(N, j0 + BLOCK)
            Y_block = [load_normalized_verts(os.path.join(GT_DIR, names[j])) for j in range(j0, j1)]
            Y_block = np.stack(Y_block, dtype=DTYPE)

            # Compute pairwise Euclidean distances between normalized meshes
            D_block = cdist(X_block, Y_block, metric="euclidean").astype(DTYPE)
            D[i0:i1, j0:j1] = D_block
            D[j0:j1, i0:i1] = D_block.T
            D.flush()
            del Y_block, D_block

        del X_block

    # Compress to .npz for later use
    print("💾 Saving compressed NPZ (this may take a few minutes)...")
    np.savez_compressed(OUT, D_orig=np.array(D), names=np.array(names))
    del D
    os.remove(tmp_path)
    print(f"✅ Saved {OUT} | D shape={(N, N)})")

if __name__ == "__main__":
    main()
