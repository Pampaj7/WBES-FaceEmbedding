#!/usr/bin/env python3
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random

# ============================================================
# IMPORTS
# ============================================================

from basic_test_grid import SpatialGrid
import sys

sys.path.append(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/gt_encdec/autoencoder"
)

from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset
from diffusion_autoencoder import DiffusionEncoderOnly

try:
    import diffusion_net
    DiffusionNet = diffusion_net.layers.DiffusionNet
except Exception:
    from diffusion_net import DiffusionNet

# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GRID_SIZE   = 8
LATENT_DIM  = 256
WIDTH       = 128
N_BLOCKS    = 4
DROPOUT     = 0.1

K_NEG = 5
SEED  = 42

DATA_CANON = Path(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/REMESH/data_CANONICAL"
)

DATA_OPS = Path(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/REMESH/npz_data_topo_500_withops"
)

ENCODER_CKPT = Path(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/gt_encdec/remeshing/"
    "cross_topo_model/encoder_stage1_multitopo_second_try/encoder_stage1_epoch50.pth"
)

OUT_CSV = Path("identity_confusion_test_with_crop.csv")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# UTILS
# ============================================================

def load_sparse_tensor(indices, values, shape):
    indices = torch.from_numpy(indices).long().to(DEVICE)
    values  = torch.from_numpy(values).float().to(DEVICE)
    shape   = tuple(shape.tolist())
    return torch.sparse_coo_tensor(indices, values, size=shape)


def load_subject_variant(subject_id, variant):
    canon = np.load(DATA_CANON / f"{subject_id}_{variant}.npz")
    ops   = np.load(DATA_OPS   / f"{subject_id}_{variant}.npz")

    return {
        "verts": torch.from_numpy(canon["V"]).float().to(DEVICE),
        "faces": torch.from_numpy(canon["F"]).long().to(DEVICE),
        "mass":  torch.from_numpy(ops["mass"]).float().to(DEVICE),
        "L": load_sparse_tensor(
            ops["L_indices"], ops["L_values"], ops["L_shape"]
        ),
        "evals": torch.from_numpy(ops["evals"]).float().to(DEVICE),
        "evecs": torch.from_numpy(ops["evecs"]).float().to(DEVICE),
        "gradX": load_sparse_tensor(
            ops["gradX_indices"], ops["gradX_values"], ops["gradX_shape"]
        ),
        "gradY": load_sparse_tensor(
            ops["gradY_indices"], ops["gradY_values"], ops["gradY_shape"]
        ),
    }


@torch.no_grad()
def encode_per_vertex(model, sample):
    Z_field, _ = model(
        sample["verts"],
        sample["mass"],
        sample["L"],
        sample["evals"],
        sample["evecs"],
        sample["faces"],
        sample["gradX"],
        sample["gradY"],
        return_per_vertex=True,
        add_noise=False
    )
    return Z_field.squeeze(0)


def grid_distance(grid, A, B):
    G_A, M_A = grid(A["verts"], A["Z"])
    G_B, M_B = grid(B["verts"], B["Z"])

    valid = M_A & M_B
    if valid.sum() == 0:
        return float("inf")

    return (G_A[valid] - G_B[valid]).norm(dim=1).mean().item()


def chamfer_distance(V1, V2):
    d12 = torch.cdist(V1, V2).min(dim=1)[0].mean()
    d21 = torch.cdist(V2, V1).min(dim=1)[0].mean()
    return (d12 + d21).item()

# ============================================================
# MAIN
# ============================================================

def main():
    print("🚀 Identity Preservation under Partial Geometry (Remesh + Crop)")

    encoder = DiffusionEncoderOnly(
        latent_dim=LATENT_DIM,
        width=WIDTH,
        n_blocks=N_BLOCKS,
        dropout=DROPOUT,
    ).to(DEVICE)

    encoder.load_state_dict(torch.load(ENCODER_CKPT, map_location=DEVICE))
    encoder.eval()

    grid = SpatialGrid(grid_size=GRID_SIZE)

    subjects = sorted(
        p.stem.replace("_original", "")
        for p in DATA_CANON.glob("*_original.npz")
    )

    results = []

    for sid_A in tqdm(subjects, desc="Subjects A"):
        try:
            A_orig = load_subject_variant(sid_A, "original")
            A_rem  = load_subject_variant(sid_A, "remesh")
            A_crop = load_subject_variant(sid_A, "crop")
        except FileNotFoundError:
            continue

        A_orig["Z"] = encode_per_vertex(encoder, A_orig)
        A_rem["Z"]  = encode_per_vertex(encoder, A_rem)
        A_crop["Z"] = encode_per_vertex(encoder, A_crop)

        d_self = {
            "remesh": {
                "grid": grid_distance(grid, A_orig, A_rem),
                "cham": chamfer_distance(A_orig["verts"], A_rem["verts"]),
            },
            "crop": {
                "grid": grid_distance(grid, A_orig, A_crop),
                "cham": chamfer_distance(A_orig["verts"], A_crop["verts"]),
            }
        }

        negs = random.sample(
            [s for s in subjects if s != sid_A],
            K_NEG
        )

        for sid_B in negs:
            for vb in ["remesh", "crop"]:
                try:
                    B = load_subject_variant(sid_B, vb)
                except FileNotFoundError:
                    continue

                B["Z"] = encode_per_vertex(encoder, B)

                d_grid_inter = grid_distance(grid, A_orig, B)
                d_cham_inter = chamfer_distance(
                    A_orig["verts"], B["verts"]
                )

                results.append({
                    "subject_A": sid_A,
                    "subject_B": sid_B,
                    "variant_B": vb,

                    "d_grid_self": d_self[vb]["grid"],
                    "d_grid_inter": d_grid_inter,

                    "d_chamfer_self": d_self[vb]["cham"],
                    "d_chamfer_inter": d_cham_inter,

                    "grid_confused": d_self[vb]["grid"] > d_grid_inter,
                    "chamfer_confused": d_self[vb]["cham"] > d_cham_inter,
                })

    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)

    print(f"\n✅ Saved results to {OUT_CSV.resolve()}")

    print("\n📊 CONFUSION RATES")
    print("Chamfer:", df["chamfer_confused"].mean())
    print("Grid   :", df["grid_confused"].mean())

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
