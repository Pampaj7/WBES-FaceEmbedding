import sys

if "/equilibrium/lpampaloni/diffusion-net/src" not in sys.path:
    sys.path.append("/equilibrium/lpampaloni/diffusion-net/src")

import os
import sys
import torch
import numpy as np
import trimesh

# === PATHS ===
DATA_DIR = "../../../datasets/GT_ready/"
OPS_DIR = os.path.join(DATA_DIR, "operators")
CKPT_PATH = "results_diffusionAE/diffusionAE_epoch20.pth"  # cambia se serve
OUT_DIR = "debug_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# === Add DiffusionNet path ===
if "/home/pampaj/diffusion-net/src" not in sys.path:
    sys.path.append("/home/pampaj/diffusion-net/src")

from diffusion_autoencoder import DiffusionAutoencoder
from dataset_gtready import GTReadyDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧩 Using device: {device}")

# === Load model ===
model = DiffusionAutoencoder(latent_dim=32).to(device)
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.eval()

# === Dataset (solo un campione) ===
dataset = GTReadyDataset(DATA_DIR, ops_dir=OPS_DIR, device=device)
sample = dataset[0]
print(f"🔹 Sample loaded: {sample['name']}")

# === Print operator stats ===
def safe_stats(name, t):
    if t.is_sparse:
        vals = t.coalesce().values()
        print(f"{name:>8s}: shape={tuple(t.shape)} (sparse) | "
              f"min={vals.min():.3e}, max={vals.max():.3e}, mean={vals.mean():.3e}, std={vals.std():.3e}")
    else:
        print(f"{name:>8s}: shape={tuple(t.shape)} | "
              f"min={t.min():.3e}, max={t.max():.3e}, mean={t.mean():.3e}, std={t.std():.3e}")

print("\n=== INPUT STATS ===")
for key in ["verts", "mass", "L", "evals", "evecs"]:
    t = sample[key]
    if isinstance(t, torch.Tensor):
        safe_stats(key, t)
print("===================\n")

# === Forward pass senza AMP ===
with torch.no_grad():
    V = sample["verts"].to(device)
    mass = sample["mass"].to(device)
    L = sample["L"].to(device)
    evals = sample["evals"].to(device)
    evecs = sample["evecs"].to(device)
    gradX = sample.get("gradX")
    gradY = sample.get("gradY")

    V_rec, _ = model(V, mass, L, evals, evecs, gradX, gradY)

print("=== OUTPUT STATS ===")
print(f"V_rec shape: {tuple(V_rec.shape)}")
print(f"min={V_rec.min():.3e}, max={V_rec.max():.3e}, mean={V_rec.mean():.3e}, std={V_rec.std():.3e}")

# === Check for NaN / Inf ===
if torch.isnan(V_rec).any() or torch.isinf(V_rec).any():
    print("⚠️  WARNING: NaN or Inf detected in reconstruction output!")
else:
    print("✅ No NaN/Inf detected.")

# === Save reconstruction and original for comparison ===
verts_rec = V_rec.cpu().numpy()
verts_in = sample["verts"].cpu().numpy()
faces_np = sample["faces"].cpu().numpy()

np.save(os.path.join(OUT_DIR, "verts_input.npy"), verts_in)
np.save(os.path.join(OUT_DIR, "verts_reconstructed.npy"), verts_rec)
np.save(os.path.join(OUT_DIR, "faces.npy"), faces_np)

mesh_rec = trimesh.Trimesh(vertices=verts_rec, faces=faces_np, process=False)
mesh_in = trimesh.Trimesh(vertices=verts_in, faces=faces_np, process=False)
mesh_rec.export(os.path.join(OUT_DIR, "reconstructed.obj"))
mesh_in.export(os.path.join(OUT_DIR, "original.obj"))

print(f"💾 Saved meshes to: {OUT_DIR}")
print("👉 Load them in Meshlab or Blender to check alignment visually.")
