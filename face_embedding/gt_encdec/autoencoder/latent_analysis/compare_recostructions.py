#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch

# ================================================
# PATH FIX
# ================================================
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

from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset
from diffusion_autoencoder import DiffusionAutoencoder


# ================================================
# CONFIG
# ================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_DIR = "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/GT_ready/npz_data"
CHECKPOINT = "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/gt_encdec/autoencoder/results_diffusionAE/diffusionAE_5000_epoch45.pth"

N_SAMPLES = 50

OUT_CSV = "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/gt_encdec/autoencoder/results_diffusionAE/mean_errors_53k.csv"


# ================================================
# MAIN
# ================================================
def main():

    print(f"Using {DEVICE}")

    # Load model
    print("Loading model…")
    model = DiffusionAutoencoder().to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    model.eval()

    # Load dataset
    print("Loading dataset…")
    dataset = GTReadyDataset(DATASET_DIR)
    print(f"Dataset size = {len(dataset)}")

    mean_errors = []

    for idx in range(N_SAMPLES):
        print(f"[{idx+1}/{N_SAMPLES}] evaluating…")

        sample = dataset[idx]

        V_gt = sample["verts"].to(DEVICE)

        # inference
        with torch.no_grad():
            V_rec, _ = model(
                V_gt,
                sample["mass"].to(DEVICE),
                sample["L"].to(DEVICE),
                sample["evals"].to(DEVICE),
                sample["evecs"].to(DEVICE),
                faces=sample["faces"].to(DEVICE),
                gradX=sample["gradX"].to(DEVICE),
                gradY=sample["gradY"].to(DEVICE),
            )

        # L2 per-vertex
        err = torch.norm(V_gt - V_rec, dim=1)

        mean_err = err.mean().item()
        mean_errors.append(mean_err)

        print(f"  → mean L2 error: {mean_err:.6f}")

    # save CSV
    np.savetxt(OUT_CSV, np.array(mean_errors), delimiter=",")
    print(f"\nSaved mean error table → {OUT_CSV}")

    # summary
    print("\n=== SUMMARY OVER ALL SUBJECTS ===")
    print("Mean:", float(np.mean(mean_errors)))
    print("Median:", float(np.median(mean_errors)))
    print("Std:", float(np.std(mean_errors)))


if __name__ == "__main__":
    main()
