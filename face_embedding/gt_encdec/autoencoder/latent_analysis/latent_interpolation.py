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

# -------------------------------------------------------------
# Settings
# -------------------------------------------------------------
DATA_DIR = "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/GT_ready/npz_data_cropped_23470_with_ops"
CKPT_PATH = "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/gt_encdec/autoencoder/test_safe_latent/diffusionAE_epoch50.pth"

OUT_DIR = "../test_safe_latent/latent_interpolation_output"
os.makedirs(OUT_DIR, exist_ok=True)

METRICS_FILE = os.path.join(OUT_DIR, "metrics.txt")

LATENT_DIM = 256
WIDTH = 128
N_BLOCKS = 4
K_SPEC = 16


# ============================================================
# Chamfer distance
# ============================================================
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


# ============================================================
# Save OBJ
# ============================================================
def save_obj(v, f, path):
    with open(path, "w") as o:
        for x in v:
            o.write(f"v {x[0]:.6f} {x[1]:.6f} {x[2]:.6f}\n")
        for tri in f:
            o.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


# ============================================================
# Main
# ============================================================
def main():
    print("Loading model...")
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

    print("Loading dataset...")
    dataset = GTReadyDatasetNPZ(DATA_DIR)

    # Select two subjects A and B
    sampleA = dataset[0]
    sampleB = dataset[1]

    # Unpack tensors
    def unpack(sample):
        return (
            sample["verts"].to(device),
            sample["mass"].to(device),
            sample["L"].to(device),
            sample["evals"].to(device),
            sample["evecs"].to(device),
            sample["faces"].to(device),
            sample["gradX"].to(device),
            sample["gradY"].to(device),
        )

    (VA, massA, LA, evalsA, evecsA, facesA, gradXA, gradYA) = unpack(sampleA)
    (VB, massB, LB, evalsB, evecsB, facesB, gradXB, gradYB) = unpack(sampleB)

    print("Extracting Zpv_A and Zpv_B...")
    with torch.no_grad():
        Zpv_A = bottleneck(encoder(VA, massA, LA, evalsA, evecsA,
                                   faces=facesA, gradX=gradXA, gradY=gradYA))
        Zpv_B = bottleneck(encoder(VB, massB, LB, evalsB, evecsB,
                                   faces=facesB, gradX=gradXB, gradY=gradYB))

        SA = model._take_or_pad_evecs(evecsA, K_SPEC)
        SB = model._take_or_pad_evecs(evecsB, K_SPEC)

    # Reset metrics file
    with open(METRICS_FILE, "w") as f:
        f.write("")

    print("\n=== Test D: Latent interpolation (A <-> B) ===")

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    for α in alphas:
        with torch.no_grad():
            
            Zpv_alpha = (1 - α) * Zpv_A + α * Zpv_B

            S_alpha = (1 - α) * SA + α * SB

            Z_in_alpha = torch.cat([Zpv_alpha, S_alpha], dim=1)

            V_rec = decoder(
                Z_in_alpha,
                massA, LA, evalsA, evecsA,
                faces=facesA, gradX=gradXA, gradY=gradYA
            )

            # Metrics
            chamA = chamfer_distance(V_rec.cpu(), VA.cpu())
            chamB = chamfer_distance(V_rec.cpu(), VB.cpu())

            Zg_alpha = Zpv_alpha.mean(dim=0)
            cosA = torch.nn.functional.cosine_similarity(Zg_alpha, Zpv_A.mean(dim=0), dim=0).item()
            cosB = torch.nn.functional.cosine_similarity(Zg_alpha, Zpv_B.mean(dim=0), dim=0).item()

        fname = f"interp_{α:.2f}.obj"
        out_path = os.path.join(OUT_DIR, fname)
        save_obj(V_rec.cpu().numpy(), facesA.cpu().numpy(), out_path)

        with open(METRICS_FILE, "a") as f:
            f.write(f"{fname} {chamA:.6f} {chamB:.6f} {cosA:.6f} {cosB:.6f}\n")

        print(f"α={α:.2f} | cham(A)={chamA:.4f} | cham(B)={chamB:.4f} | cosA={cosA:.4f} | cosB={cosB:.4f}")


if __name__ == "__main__":
    main()
