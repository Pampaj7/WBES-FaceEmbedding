#!/usr/bin/env python3
import os
import numpy as np
import torch
from tqdm import tqdm
import os, sys

# === Add DiffusionNet path ===
for p in [
    "/equilibrium/lpampaloni/diffusion-net/src",
    "/home/pampaj/diffusion-net/src",
    "/seidenas/users/lpampaloni/diffusion-net/src",
]:
    if p not in sys.path:
        sys.path.append(p)

# aggiungi la cartella dell'autoencoder al path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AUTOENC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # una directory sopra

if AUTOENC_DIR not in sys.path:
    sys.path.append(AUTOENC_DIR)

from diffusion_autoencoder import DiffusionAutoencoder

from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset
from diffusion_autoencoder import DiffusionEncoderOnly
from helper import patch_dataset_with_get_by_name, collate_skip

# ==========================================================
# CONFIG
# ==========================================================
DATA_DIR = "../../../../datasets/GT_ready/npz_data_cropped_23470_with_ops/"
CKPT = "../encoder_only/encoder_only_epoch50.pth"    # <--- metti qui il tuo file
OUT_LATENTS = "../encoder_only/latents_stage1.npz"

LATENT_DIM = 256
WIDTH = 128
N_BLOCKS = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Extracting latents using encoder on {device}")


# ==========================================================
# LOAD DATASET
# ==========================================================
dataset = GTReadyDataset(DATA_DIR)
dataset = patch_dataset_with_get_by_name(dataset)

print(f"📂 Loaded dataset with {len(dataset)} meshes")


# ==========================================================
# LOAD ENCODER MODEL
# ==========================================================
model = DiffusionEncoderOnly(
    latent_dim=LATENT_DIM,
    width=WIDTH,
    n_blocks=N_BLOCKS,
    dropout=0.1,
).to(device)

print(f"🔧 Loading encoder checkpoint: {CKPT}")
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()


# ==========================================================
# EXTRACT LATENTS
# ==========================================================
names = []
Z_list = []

with torch.no_grad():
    for sample in tqdm(dataset, desc="Extracting Z"):
        
        base = sample["name"][:-4] if sample["name"].endswith(".npz") else sample["name"]
        
        V = sample["verts"].to(device)
        mass = sample["mass"].to(device)
        evals = sample["evals"].to(device)
        evecs = sample["evecs"].to(device)
        faces = sample["faces"].to(device)
        L = sample["L"].to(device)
        gradX = sample["gradX"].to(device)
        gradY = sample["gradY"].to(device)

        Z = model(V, mass, L, evals, evecs, faces, gradX, gradY)
        Z_list.append(Z.cpu().numpy())
        names.append(base)

Z_array = np.vstack(Z_list)

# ==========================================================
# SAVE LATENTS
# ==========================================================
np.savez(
    OUT_LATENTS,
    names=np.array(names),
    Z=Z_array,
)

print(f"\n💾 Saved latent vectors → {OUT_LATENTS}")
print(f"   Z shape: {Z_array.shape}")
