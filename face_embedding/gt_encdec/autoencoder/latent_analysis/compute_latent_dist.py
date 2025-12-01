#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ============================================================
# FIX PYTHONPATH
# ============================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

for p in [
    "/equilibrium/lpampaloni/diffusion-net/src",
    "/home/pampaj/diffusion-net/src",
    "/seidenas/users/lpampaloni/diffusion-net/src",
]:
    if p not in sys.path:
        sys.path.append(p)

# ============================================================
# IMPORTS
# ============================================================
from helper import patch_dataset_with_get_by_name
from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset
from diffusion_autoencoder import DiffusionAutoencoder, DiffusionEncoderOnly


def collate_skip(batch):
    return [s for s in batch if s is not None]


# ============================================================
# CONFIG (CAMBIA SOLO QUESTE 3 COSE)
# ============================================================
IS_ENCODER_ONLY = True                       # <--- SWITCH QUI

MODEL_PATH = (
    "../encoder_only/encoder_only_epoch50.pth"
    if IS_ENCODER_ONLY
    else "../test_safe_latent/diffusionAE_epoch40.pth"
)

OUT_FILE = (
    "../encoder_only/latent_distances_encoderonly.npz"
    if IS_ENCODER_ONLY
    else "../test_safe_latent/latent_distances_autoencoder.npz"
)

DATA_DIR = "../../../../datasets/GT_ready/npz_data_cropped_23470_with_ops/"

BATCH_SIZE = 16
N_FILES = 5000                               # <--- sottocampione veloce
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# PAIRWISE DIST (velocissima)
# ============================================================
def pairwise_distances(X):
    XX = np.sum(X * X, axis=1, keepdims=True)
    dist2 = XX + XX.T - 2 * (X @ X.T)
    dist2[dist2 < 0] = 0
    return np.sqrt(dist2)


# ============================================================
# LOAD DATASET
# ============================================================
dataset = GTReadyDataset(DATA_DIR)
dataset = patch_dataset_with_get_by_name(dataset)

dataset.files = dataset.files[:N_FILES]   # <--- sottocampione veloce

print("Dataset length:", len(dataset))

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_skip,
    num_workers=12,     # velocizza ma non esplode la RAM
)


# ============================================================
# LOAD MODEL (dipende dal flag)
# ============================================================
if IS_ENCODER_ONLY:
    print("🧬 Using ENCODER-ONLY model")
    model = DiffusionEncoderOnly(
        latent_dim=256, width=128, n_blocks=4
    ).to(DEVICE)
else:
    print("🧬 Using FULL AUTOENCODER")
    model = DiffusionAutoencoder(
        latent_dim=256, width=128, n_blocks=4
    ).to(DEVICE)

print(model)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# ============================================================
# EXTRACT LATENTS
# ============================================================
print("🔍 Extracting latent codes...")

Z_all = []
names = []

with torch.no_grad():
    for batch in tqdm(loader):
        for sample in batch:

            V     = sample["verts"].to(DEVICE)
            mass  = sample["mass"].to(DEVICE)
            L     = sample["L"].to(DEVICE)
            evals = sample["evals"].to(DEVICE)
            evecs = sample["evecs"].to(DEVICE)
            faces = sample["faces"].to(DEVICE)
            gradX = sample["gradX"].to(DEVICE)
            gradY = sample["gradY"].to(DEVICE)

            if IS_ENCODER_ONLY:
                Z = model(V, mass, L, evals, evecs, faces, gradX, gradY)
            else:
                _, Z = model(V, mass, L, evals, evecs, faces, gradX, gradY)

            Z_all.append(Z.squeeze().cpu().numpy())
            names.append(sample["name"])

print("Collected:", len(Z_all))

if len(Z_all) == 0:
    raise RuntimeError("❌ No latent vectors extracted — dataset or loader broken")


Z_all = np.vstack(Z_all)


# ============================================================
# LOAD GT DISTANCES (stesso subset)
# ============================================================
DIST_GT_PATH = "gt_distance_matrix/normalized_matrix_distances.npz"
D_gt_pack = np.load(DIST_GT_PATH)
D_orig_full = D_gt_pack["D_orig"]
D_orig = D_orig_full[:N_FILES, :N_FILES]   # allinea ai 1000 campioni


# ============================================================
# COMPUTE D_lat & SAVE
# ============================================================
print("📐 Computing pairwise lat-dist...")
D_lat = pairwise_distances(Z_all)

np.savez(
    OUT_FILE,
    D_orig=D_orig,
    D_lat=D_lat,
    names=np.array(names),
)

print("✅ Saved:", OUT_FILE)
