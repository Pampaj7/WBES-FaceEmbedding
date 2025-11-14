#!/usr/bin/env python3
"""
Compute pairwise GT distance matrix using the SAME normalisation
applied in GTReadyDatasetNPZ:
- subtract mean
- divide by global max |coord|
- then L2 per-vertex mean squared distance
"""

import os
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import torch

# =======================
# Config
# =======================
DATA_DIR = "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/GT_ready/npz_data_cropped_23470_with_ops"

# Usa 26 core → abbastanza parallelo senza saturare la RAM
N_PROC = min(max(mp.cpu_count() - 4, 1), 26)

# Cartella di output
OUT_DIR = "./gt_distance_matrix"
os.makedirs(OUT_DIR, exist_ok=True)

# File finale
OUT_PATH = os.path.join(OUT_DIR, "normalized_matrix_distances.npz")

# ============================================================
# GLOBAL STATE (VISIBILE NEI WORKER PER FORK)
# ============================================================
V_LIST = None
N_MESHES = 0


# ============================================================
# HELPERS
# ============================================================
def load_mesh(path: str) -> np.ndarray:
    """Load verts [N,3] from NPZ."""
    data = np.load(path, allow_pickle=False)
    V = data["verts"].astype(np.float32)
    data.close()
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"Invalid verts shape {V.shape} in {path}")
    return V


def init_worker(v_list, n_meshes):
    """
    Inizializza i worker con un riferimento globale a V_LIST.
    Su Linux (fork) questo è copy-on-write: niente copia reale in RAM.
    """
    global V_LIST, N_MESHES
    V_LIST = v_list
    N_MESHES = n_meshes


def face_distance(pair):
    """
    Compute per-vertex mean L2 distance between two meshes.

    pair = (i, j) sono SOLO indici.
    I dati veri stanno in V_LIST, condivisa tra i processi (via fork).
    """
    i, j = pair
    Vi = V_LIST[i]
    Vj = V_LIST[j]

    diff = Vi - Vj
    # L2 per vertice, poi media sui vertici
    d = np.sqrt((diff * diff).sum(axis=1)).mean().astype(np.float32)
    return i, j, d


def task_generator(n):
    """
    Generatore di tutte le coppie (i, j) con i < j.
    Non costruisce una lista gigante in RAM, ma streama le coppie.
    """
    for i in range(n):
        for j in range(i + 1, n):
            yield (i, j)


# ============================================================
# MAIN
# ============================================================
def main():
    global V_LIST, N_MESHES

    # -------------------------
    # 1) Carica tutte le mesh
    # -------------------------
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".npz")])
    if not files:
        raise RuntimeError(f"No .npz files found in {DATA_DIR}")

    print(f"Found {len(files)} meshes")

    names = []
    V_list_local = []

    for fname in tqdm(files, desc="Loading", ncols=100):
        path = os.path.join(DATA_DIR, fname)
        V = load_mesh(path)
        V_list_local.append(V)
        names.append(fname.replace(".npz", ""))

    V_LIST = V_list_local
    N_MESHES = len(V_LIST)
    n = N_MESHES

    print(f"Meshes loaded: {n}")
    print(f"Using {N_PROC} worker processes.")

    # -------------------------
    # 2) Alloca matrice D
    # -------------------------
    print("Allocating distance matrix...")
    D = np.zeros((n, n), dtype=np.float32)

    # -------------------------
    # 3) Prepara generatore di task
    # -------------------------
    total_pairs = n * (n - 1) // 2
    gen = task_generator(n)

    # -------------------------
    # 4) Pool globale, nessun nested Pool
    # -------------------------
    print("Computing pairwise distances...")
    with mp.Pool(
        processes=N_PROC,
        initializer=init_worker,
        initargs=(V_LIST, N_MESHES),
    ) as pool:
        for i, j, d in tqdm(
            pool.imap_unordered(face_distance, gen, chunksize=64),
            total=total_pairs,
            ncols=100,
            desc="Pairs",
        ):
            D[i, j] = d
            D[j, i] = d

    # -------------------------
    # 5) Normalizzazione
    # -------------------------
    mask_pos = D > 0
    if not np.any(mask_pos):
        raise RuntimeError("Distance matrix is all zeros, something went wrong.")

    max_val = D[mask_pos].max()
    D_norm = D / max_val
    print(f"Normalized by factor: {max_val:.6f}")

    # -------------------------
    # 6) Salvataggio
    # -------------------------
    np.savez(OUT_PATH, D_orig=D_norm.astype(np.float32), names=np.array(names))
    print(f"\n✅ Saved GT distance matrix to: {OUT_PATH}")
    print(f"   Shape: {D_norm.shape}")


if __name__ == "__main__":
    # Importante per multiprocessing su Windows/macOS,
    # innocuo ma ok anche su Linux/HPC.
    main()