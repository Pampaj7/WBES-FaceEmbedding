#!/usr/bin/env python3
"""
STEP 1 — Precompute grid-based embeddings for the entire dataset.

For each (subject, variant):
- run frozen encoder
- pool per-vertex latents into spatial grid
- save (G, M) to disk

Output structure:
grid_cache/
  id0000_GTready/
    original.npz
    remesh.npz
    crop.npz
    noisy.npz
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
from basic_test_grid import SpatialGrid

# -------------------------------------------------
# PATHS
# -------------------------------------------------

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

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GRID_SIZE = 8
LATENT_DIM = 256
WIDTH = 128
N_BLOCKS = 4
DROPOUT = 0.1

VARIANTS = ["original", "remesh", "crop", "noisy"]

DATA_CANON = Path(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/REMESH/data_CANONICAL"
)
DATA_OPS = Path(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/REMESH/npz_data_topo_500_withops"
)

ENCODER_CKPT = Path(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/gt_encdec/remeshing/"
    "cross_topo_model/encoder_stage1_multitopo_second_try/"
    "encoder_stage1_epoch50.pth"
)

OUT_DIR = Path("grid_cache")
OUT_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# DATA LOADING
# -------------------------------------------------

def load_sparse_tensor(indices, values, shape, device):
    indices = torch.from_numpy(indices).long().to(device)
    values  = torch.from_numpy(values).float().to(device)
    shape   = tuple(shape.tolist())
    return torch.sparse_coo_tensor(indices, values, size=shape)


def load_subject_variant(subject_id, variant):
    canon_path = DATA_CANON / f"{subject_id}_{variant}.npz"
    ops_path   = DATA_OPS   / f"{subject_id}_{variant}.npz"

    canon = np.load(canon_path)
    ops   = np.load(ops_path)

    sample = {
        "verts": torch.from_numpy(canon["V"]).float().to(DEVICE),
        "faces": torch.from_numpy(canon["F"]).long().to(DEVICE),

        "mass": torch.from_numpy(ops["mass"]).float().to(DEVICE),

        "L": load_sparse_tensor(
            ops["L_indices"], ops["L_values"], ops["L_shape"], DEVICE
        ),
        "evals": torch.from_numpy(ops["evals"]).float().to(DEVICE),
        "evecs": torch.from_numpy(ops["evecs"]).float().to(DEVICE),

        "gradX": load_sparse_tensor(
            ops["gradX_indices"], ops["gradX_values"], ops["gradX_shape"], DEVICE
        ),
        "gradY": load_sparse_tensor(
            ops["gradY_indices"], ops["gradY_values"], ops["gradY_shape"], DEVICE
        ),
    }
    return sample

# -------------------------------------------------
# ENCODER
# -------------------------------------------------

@torch.no_grad()
def encode_per_vertex(model, sample):
    """
    Returns:
        Z_field ∈ R^{N × D}
    """
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

# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():
    print("🚀 STEP 1 — Precomputing grid embeddings")

    # ----------------------------
    # Load encoder
    # ----------------------------
    encoder = DiffusionEncoderOnly(
        latent_dim=LATENT_DIM,
        width=WIDTH,
        n_blocks=N_BLOCKS,
        dropout=DROPOUT,
    ).to(DEVICE)

    encoder.load_state_dict(torch.load(ENCODER_CKPT, map_location=DEVICE))
    encoder.eval()

    print("✅ Encoder loaded")

    # ----------------------------
    # Grid
    # ----------------------------
    grid = SpatialGrid(grid_size=GRID_SIZE)

    # ----------------------------
    # Subjects
    # ----------------------------
    subjects = sorted(
        p.stem.replace("_original", "")
        for p in DATA_CANON.glob("*_original.npz")
    )

    print(f"📦 Found {len(subjects)} subjects")

    # ----------------------------
    # Loop
    # ----------------------------
    for sid in tqdm(subjects, desc="Subjects"):
        subj_out = OUT_DIR / sid
        subj_out.mkdir(exist_ok=True)

        for variant in VARIANTS:
            out_path = subj_out / f"{variant}.npz"
            if out_path.exists():
                continue  # cache hit

            try:
                sample = load_subject_variant(sid, variant)
            except FileNotFoundError:
                continue

            Z = encode_per_vertex(encoder, sample)
            G, M = grid(sample["verts"], Z)

            np.savez(
                out_path,
                G=G.detach().cpu().numpy().astype(np.float32),
                M=M.detach().cpu().numpy().astype(np.bool_)
            )

    print("\n✅ DONE — Grid cache ready.")
    print(f"📁 Saved to: {OUT_DIR.resolve()}")

# -------------------------------------------------
# ENTRY
# -------------------------------------------------

if __name__ == "__main__":
    main()
