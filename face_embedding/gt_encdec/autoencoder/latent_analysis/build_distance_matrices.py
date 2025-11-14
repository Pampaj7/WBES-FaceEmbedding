#!/usr/bin/env python3
import os, sys, numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
import traceback

# === CONFIG PATHS ===
GT_DIR  = "../../../../datasets/GT_ready/npz_data"
LAT_DIR = "../results_diffusionAE_latentaware/latents_full"
OUT_DIR = "../results_diffusionAE_latentaware/dist_matrices_fields"
os.makedirs(OUT_DIR, exist_ok=True)

# === COSTRUISCI LISTA FILE COMUNI ===
all_lat = sorted([f for f in os.listdir(LAT_DIR) if f.endswith(".npz")])
common = [f for f in all_lat if os.path.exists(os.path.join(GT_DIR, f))]
print(f"📦 Trovati {len(common)} file comuni tra GT e LATENTI (in {LAT_DIR})")

# === FUNZIONI DI CARICAMENTO E FALLBACK KEYS ===
def load_gt(path):
    d = np.load(path)
    if "verts" not in d.files:
        raise KeyError("GT .npz missing 'verts' array")
    return d["verts"].astype(np.float32)  # (n_verts, 3)

def load_latent_fallback(path):
    d = np.load(path)
    keys = set(d.files)

    # Z_field: prefer RAW, poi NORM, poi legacy 'Z_field'
    if "Z_field_raw" in keys:
        Zf = d["Z_field_raw"].astype(np.float32)
    elif "Z_field_norm" in keys:
        Zf = d["Z_field_norm"].astype(np.float32)
    elif "Z_field" in keys:
        Zf = d["Z_field"].astype(np.float32)
    else:
        raise KeyError(f"No Z_field* key found in {os.path.basename(path)}")

    # Z_global: prefer RAW, poi NORM, poi legacy 'Z_global'
    if "Z_global_raw" in keys:
        Zg = d["Z_global_raw"].astype(np.float32)
    elif "Z_global_norm" in keys:
        Zg = d["Z_global_norm"].astype(np.float32)
    elif "Z_global" in keys:
        Zg = d["Z_global"].astype(np.float32)
    else:
        raise KeyError(f"No Z_global* key found in {os.path.basename(path)}")

    return Zf, Zg

# --- Caricamento e filtraggio robusto ---
verts_list = []
Zfield_list = []
Zglobal_list = []
valid_files = []
skipped = []

expected_V = None
for fn in tqdm(common, desc="Loading data"):
    gt_path = os.path.join(GT_DIR, fn)
    lat_path = os.path.join(LAT_DIR, fn)
    try:
        V = load_gt(gt_path)            # (n_verts, 3)
    except Exception as e:
        skipped.append((fn, f"GT load error: {e}"))
        continue

    try:
        Zf, Zg = load_latent_fallback(lat_path)  # Zf: (Vf, D), Zg: (D,)
    except Exception as e:
        skipped.append((fn, f"latent load error: {e}"))
        continue

    Vf = int(Zf.shape[0])
    if expected_V is None:
        expected_V = Vf
    if Vf != expected_V:
        skipped.append((fn, f"V mismatch: Vf={Vf} expected={expected_V}"))
        continue

    verts_list.append(V.flatten())
    Zfield_list.append(Zf)
    Zglobal_list.append(Zg)
    valid_files.append(fn)

# report
print(f"\n✅ Valid files after filtering: {len(valid_files)} / {len(common)}")
if skipped:
    print("⚠️ Skipped examples (first 20):")
    for s in skipped[:20]:
        print("   ", s)

if len(valid_files) == 0:
    raise RuntimeError("No valid files found after filtering — inspect skipped list above.")

# stack
verts_all = np.stack(verts_list)       # (N, n_verts*3)
Zglobal_all = np.stack(Zglobal_list)   # (N, D)
Zfield_all = np.stack(Zfield_list)     # (N, V, D)

# === NORMALIZZA LATENTI PER-VERTICE (sicuro) ===
print("\n⚙️ Normalizzo campi latenti per-vertex...")
N, V, D = Zfield_all.shape
for k in range(N):
    F = Zfield_all[k]
    norms = np.linalg.norm(F, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Zfield_all[k] = F / (norms + 1e-9)
# ora Zfield_all shape: (N, V, D)

# === FUNZIONE VELOCE PER DISTANZE PER-VERTEX ===
def pairwise_field_distance_fast(fields, chunk_verts: int = 4096):
    N, V, D = fields.shape
    Dmat = np.zeros((N, N), dtype=np.float32)
    V_float = float(V)
    for i in tqdm(range(N), desc="Field distances (BLAS+chunks)"):
        acc = np.zeros((N,), dtype=np.float64)
        for s in range(0, V, chunk_verts):
            e = min(V, s + chunk_verts)
            Fi_chunk = fields[i, s:e, :].reshape(-1)           # (cv*D,)
            All_chunk_T = fields[:, s:e, :].reshape(N, -1).T  # (cv*D, N)
            acc += Fi_chunk @ All_chunk_T
        cos_mean = acc / V_float
        dist_row = (1.0 - cos_mean).astype(np.float32)
        Dmat[i, :] = dist_row
        Dmat[:, i] = dist_row
        Dmat[i, i] = 0.0
    return Dmat

# === CALCOLO MATRICI PAIRWISE ===
print("\n🔢 Calcolo matrici pairwise...")
D_orig = squareform(pdist(verts_all, metric="euclidean"))  # distanza geometrica GT

# --- Normalizzazione dataset-wide (MDS-style) sui latenti globali ---
Z_all = Zglobal_all.astype(np.float64)  # N x D (raw means)
Zc = Z_all - Z_all.mean(axis=0, keepdims=True)
global_scale = np.linalg.norm(Zc) / np.sqrt(max(Zc.shape[0] - 1, 1))
Z_norm_global = Zc / (global_scale + 1e-8)
np.save(os.path.join(OUT_DIR, "Z_global_norm_dataset.npy"), Z_norm_global.astype(np.float32))

# distanza euclidea su latenti normalizzati (coerente con MDS-style)
D_lat_mean = squareform(pdist(Z_norm_global.astype(np.float32), metric="euclidean"))

# versione cosine (per confronto col metodo precedente)
D_lat_mean_cosine = squareform(pdist(Z_all.astype(np.float32), metric="cosine"))
np.save(os.path.join(OUT_DIR, "D_lat_mean_cosine.npy"), D_lat_mean_cosine.astype(np.float32))

# distanza per-vertex (come prima)
D_lat_field = pairwise_field_distance_fast(Zfield_all, chunk_verts=4096)

# --- statistiche rapide
mask = np.triu_indices_from(D_lat_mean, k=1)
print("\nD_lat_mean (eucl) stats: mean=%.4f std=%.4f min=%.4f max=%.4f" % (
    D_lat_mean[mask].mean(), D_lat_mean[mask].std(), D_lat_mean.min(), D_lat_mean.max()))
print("D_lat_mean_cosine stats: mean=%.4f std=%.4f" % (
    D_lat_mean_cosine[mask].mean(), D_lat_mean_cosine[mask].std()))

# === SALVA I RISULTATI ===
np.savez_compressed(
    os.path.join(OUT_DIR, "distance_matrices_fields.npz"),
    D_orig=D_orig,
    D_lat_mean=D_lat_mean,
    D_lat_mean_cosine=D_lat_mean_cosine,
    D_lat_field=D_lat_field,
    names=np.array(valid_files)
)
print(f"\n✅ Salvate matrici in {OUT_DIR}/distance_matrices_fields.npz")
print(f"   D_orig shape:      {D_orig.shape}")
print(f"   D_lat_mean shape:  {D_lat_mean.shape}")
print(f"   D_lat_field shape: {D_lat_field.shape}")

print("\n💡 Suggerimento: esporta OMP_NUM_THREADS e MKL_NUM_THREADS per usare tutti i core.")
print("   Esempio:")
print("   export OMP_NUM_THREADS=32; export MKL_NUM_THREADS=32; python3 build_distance_matrices.py")
