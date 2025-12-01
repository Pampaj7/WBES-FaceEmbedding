#!/usr/bin/env python3
import os
import torch
import numpy as np
import sys

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
from dataset_gtready import GTReadyDatasetNPZ
from diffusion_autoencoder import DiffusionAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
DATA_DIR = "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/GT_ready/npz_data_cropped_23470_with_ops"
CKPT_PATH = "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/gt_encdec/autoencoder/test_safe_latent/diffusionAE_epoch50.pth"
OUT_DIR = "../test_safe_latent/latent_sensitivity_output"
os.makedirs(OUT_DIR, exist_ok=True)

METRICS_FILE = os.path.join(OUT_DIR, "metrics.txt")

LATENT_DIM = 256
WIDTH = 128
N_BLOCKS = 4
K_SPEC = 16


def chamfer_distance(points_a, points_b, n_samples=5000):
    pa = points_a.float()
    pb = points_b.float()

    idx_a = torch.randperm(pa.shape[0])[:n_samples]
    idx_b = torch.randperm(pb.shape[0])[:n_samples]

    pa = pa[idx_a]
    pb = pb[idx_b]

    diff1 = pa.unsqueeze(1) - pb.unsqueeze(0)
    dist1 = (diff1 ** 2).sum(dim=2)
    min1 = dist1.min(dim=1).values

    diff2 = pb.unsqueeze(1) - pa.unsqueeze(0)
    dist2 = (diff2 ** 2).sum(dim=2)
    min2 = dist2.min(dim=1).values

    chamfer = min1.mean() + min2.mean()
    return chamfer.item()


# ---------------------------------------------------------------------
# Utility: save OBJ
# ---------------------------------------------------------------------
def save_obj(v, f, path):
    with open(path, "w") as o:
        for x in v:
            o.write(f"v {x[0]:.6f} {x[1]:.6f} {x[2]:.6f}\n")
        for tri in f:
            o.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("Loading AE model ...")
    model = DiffusionAutoencoder(
        latent_dim=LATENT_DIM,
        width=WIDTH,
        n_blocks=N_BLOCKS,
        k_spec=K_SPEC
    ).to(device)

    state = torch.load(CKPT_PATH, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)

    model.eval()
    encoder = model.encoder
    bottleneck = model.vertex_bottleneck
    decoder = model.decoder

    for p in model.parameters():
        p.requires_grad = False

    print("Loading dataset …")
    dataset = GTReadyDatasetNPZ(DATA_DIR)
    sample = dataset[0]

    V = sample["verts"].to(device)
    mass = sample["mass"].to(device)
    L = sample["L"].to(device)
    evals = sample["evals"].to(device)
    evecs = sample["evecs"].to(device)
    faces = sample["faces"].to(device)
    gradX = sample["gradX"].to(device)
    gradY = sample["gradY"].to(device)

    # Reset metrics file
    with open(METRICS_FILE, "w") as f:
        f.write("")  # empty file

    print("Extracting Z_per_vertex …")

    with torch.no_grad():
        Zpv = encoder(
            V, mass, L, evals, evecs,
            faces=faces, gradX=gradX, gradY=gradY
        )
        Zpv = bottleneck(Zpv)

        S = model._take_or_pad_evecs(evecs, K_SPEC)

        Z_in = torch.cat([Zpv, S], dim=1)

        V_rec_base = decoder(
            Z_in, mass, L, evals, evecs,
            faces=faces, gradX=gradX, gradY=gradY
        )

    base_path = os.path.join(OUT_DIR, "base.obj")
    save_obj(V_rec_base.cpu().numpy(), faces.cpu().numpy(), base_path)

    # Base metrics: chamfer 0, cos_id=1
    with open(METRICS_FILE, "a") as f:
        f.write(f"base.obj 0.000000 1.000000\n")

    # ================================================================
    # TEST A: Local noise
    # ================================================================
    print("\n=== Test A: Per-vertex noise ===")
    sigmas_local = [0.01, 0.05, 0.2]

    for s in sigmas_local:
        with torch.no_grad():
            eps = torch.randn_like(Zpv) * s
            Zpv_pert = Zpv + eps

            Z_in_pert = torch.cat([Zpv_pert, S], dim=1)
            V_rec = decoder(
                Z_in_pert, mass, L, evals, evecs,
                faces=faces, gradX=gradX, gradY=gradY
            )

            chamf = chamfer_distance(V_rec.cpu(), V.cpu())
            cos_id = 1.0  # local noise does not define identity shift

        fname = f"local_noise_{s}.obj"
        out_path = os.path.join(OUT_DIR, fname)
        save_obj(V_rec.cpu().numpy(), faces.cpu().numpy(), out_path)

        with open(METRICS_FILE, "a") as f:
            f.write(f"{fname} {chamf:.6f} {cos_id:.6f}\n")

        print(f"sigma={s} | chamfer={chamf:.6f} | saved={out_path}")

    # ================================================================
    # TEST B: Global shift
    # ================================================================
    print("\n=== Test B: Global shift ===")
    sigmas_global = [0.1, 0.5, 2.0]

    direction = torch.randn(1, LATENT_DIM).to(device)

    for s in sigmas_global:
        with torch.no_grad():
            eps_global = direction * s
            Zpv_pert = Zpv + eps_global

            Z_in_pert = torch.cat([Zpv_pert, S], dim=1)
            V_rec = decoder(
                Z_in_pert, mass, L, evals, evecs,
                faces=faces, gradX=gradX, gradY=gradY
            )

            chamf = chamfer_distance(V_rec.cpu(), V.cpu())

            Zg_orig = Zpv.mean(dim=0)
            Zg_pert = Zpv_pert.mean(dim=0)
            cos_id = torch.nn.functional.cosine_similarity(
                Zg_orig, Zg_pert, dim=0).item()

        fname = f"global_shift_{s}.obj"
        out_path = os.path.join(OUT_DIR, fname)
        save_obj(V_rec.cpu().numpy(), faces.cpu().numpy(), out_path)

        with open(METRICS_FILE, "a") as f:
            f.write(f"{fname} {chamf:.6f} {cos_id:.6f}\n")

        print(f"sigma={s} | chamfer={chamf:.6f} | cos_id={cos_id:.4f} | saved={out_path}")


    # ================================================================
        
    # TEST C: Regional perturbation (Nose area)
    # ================================================================
    print("\n=== Test C: Nose-region perturbation ===")

    # --- Detect nose tip automatically ---
    with torch.no_grad():
        nose_tip_idx = torch.argmin(V[:, 2])  # punta del naso
        nose_tip = V[nose_tip_idx]
        dist = torch.norm(V - nose_tip, dim=1)
        nose_mask = dist < 0.02   # raggio 2 cm circa

    sigmas_nose = [0.05, 0.1, 0.5]

    for s in sigmas_nose:
        with torch.no_grad():
            Zpv_pert = Zpv.clone()
            noise = torch.randn_like(Zpv_pert[nose_mask]) * s
            Zpv_pert[nose_mask] += noise

            Z_in_pert = torch.cat([Zpv_pert, S], dim=1)

            V_rec = decoder(
                Z_in_pert, mass, L, evals, evecs,
                faces=faces, gradX=gradX, gradY=gradY
            )

            chamf = chamfer_distance(V_rec.cpu(), V.cpu())

            Zg_orig = Zpv.mean(dim=0)
            Zg_pert = Zpv_pert.mean(dim=0)
            cos_id = torch.nn.functional.cosine_similarity(
                Zg_orig, Zg_pert, dim=0
            ).item()

        fname = f"nose_perturb_{s}.obj"
        out_path = os.path.join(OUT_DIR, fname)
        save_obj(V_rec.cpu().numpy(), faces.cpu().numpy(), out_path)

        with open(METRICS_FILE, "a") as f:
            f.write(f"{fname} {chamf:.6f} {cos_id:.6f}\n")

        print(f"sigma={s} | chamfer={chamf:.6f} | cos_id={cos_id:.4f} | saved={out_path}")


if __name__ == "__main__":
    main()
