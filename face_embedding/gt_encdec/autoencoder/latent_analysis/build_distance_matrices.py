#!/usr/bin/env python3
import os, sys, torch, numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

# === CONFIG PATHS ===
GT_DIR  = "../../../datasets/GT_ready/npz_data"
LAT_DIR = "results_diffusionAE/latents_full"
OUT_DIR = "results_diffusionAE/dist_matrices_fields"
os.makedirs(OUT_DIR, exist_ok=True)

# === COSTRUISCI LISTA FILE COMUNI ===
common = sorted([
    f for f in os.listdir(LAT_DIR)
    if f.endswith(".npz") and os.path.exists(os.path.join(GT_DIR, f))
])
print(f"📦 Trovati {len(common)} file comuni tra GT e LATENTI")

# === FUNZIONI DI CARICAMENTO ===
def load_gt(path):
    d = np.load(path)
    return d["verts"].astype(np.float32)  # (n_verts, 3)

def load_latent(path):
    d = np.load(path)
    Zf = d["Z_field"].astype(np.float32)   # (n_verts, 256)
    Zg = d["Z_global"].astype(np.float32)  # (256,)
    return Zf, Zg

# === CARICAMENTO DATI ===
verts_all = []
Zfield_all = []
Zglobal_all = []

for fn in tqdm(common, desc="Loading data"):
    V = load_gt(os.path.join(GT_DIR, fn))
    Zf, Zg = load_latent(os.path.join(LAT_DIR, fn))
    verts_all.append(V.flatten())
    Zfield_all.append(Zf)
    Zglobal_all.append(Zg)

verts_all = np.stack(verts_all)       # (N, n_verts*3)
Zglobal_all = np.stack(Zglobal_all)   # (N, 256)

# === NORMALIZZA LATENTI PER-VERTICE ===
print("⚙️ Normalizzo campi latenti per-vertex...")
for k in range(len(Zfield_all)):
    F = Zfield_all[k]
    norms = np.linalg.norm(F, axis=1, keepdims=True) + 1e-9
    Zfield_all[k] = F / norms

Zfield_all = np.stack(Zfield_all)  # (N, n_verts, 256)

# === FUNZIONE VELOCE PER DISTANZE PER-VERTEX ===
def pairwise_field_distance_fast(fields, chunk_verts: int = 4096):
    """
    Calcola D_lat_field (NxN) come 1 - mean_vertex(dot(F_i[v], F_j[v])).
    fields: (N, V, D) float32, normalizzati per riga (per-vertex).
    Usa chunking sui vertici + BLAS per parallelizzare.
    """
    N, V, D = fields.shape
    Dmat = np.zeros((N, N), dtype=np.float32)
    V_float = float(V)

    for i in tqdm(range(N), desc="Field distances (BLAS+chunks)"):
        acc = np.zeros((N,), dtype=np.float64)
        for s in range(0, V, chunk_verts):
            e = min(V, s + chunk_verts)
            Fi_chunk = fields[i, s:e, :].reshape(-1)       # (cv*D,)
            All_chunk_T = fields[:, s:e, :].reshape(N, -1).T  # (cv*D, N)
            acc += Fi_chunk @ All_chunk_T                  # BLAS multi-thread qui

        cos_mean = acc / V_float
        dist_row = (1.0 - cos_mean).astype(np.float32)
        Dmat[i, :] = dist_row
        Dmat[:, i] = dist_row
        Dmat[i, i] = 0.0

    return Dmat

# === CALCOLO MATRICE ===
print("🔢 Calcolo matrici pairwise...")
D_orig = squareform(pdist(verts_all, metric="euclidean"))  # distanza geometrica GT
D_lat_mean = squareform(pdist(Zglobal_all, metric="cosine"))  # distanza tra latenti globali
D_lat_field = pairwise_field_distance_fast(Zfield_all, chunk_verts=4096)  # distanza per-vertex

# === SALVA I RISULTATI ===
np.savez_compressed(
    os.path.join(OUT_DIR, "distance_matrices_fields.npz"),
    D_orig=D_orig,
    D_lat_mean=D_lat_mean,
    D_lat_field=D_lat_field,
    names=common
)

print(f"\n✅ Salvate matrici in {OUT_DIR}/distance_matrices_fields.npz")
print(f"   D_orig shape:      {D_orig.shape}")
print(f"   D_lat_mean shape:  {D_lat_mean.shape}")
print(f"   D_lat_field shape: {D_lat_field.shape}")
print("\n💡 Suggerimento: esporta OMP_NUM_THREADS e MKL_NUM_THREADS per usare tutti i core.")
print("   Esempio:")
print("   export OMP_NUM_THREADS=32; export MKL_NUM_THREADS=32; python3 compute_distances_fast.py")
