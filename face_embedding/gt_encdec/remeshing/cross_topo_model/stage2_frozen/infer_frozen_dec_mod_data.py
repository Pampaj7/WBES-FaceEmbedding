#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np
import torch

# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODER_CKPT = "../encoder_stage1_multitopo_second_try/encoder_stage1_epoch50.pth"
DECODER_CKPT = "stage2_decoder_epoch50.pth"

DATA_DIR = Path(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/"
    "REMESH/npz_data_topo_500_withops"
)

SUBJECT_ID = "id0000"

OUT_DIR = Path("outputs/stage2_frozen_inference")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LATENT_DIM = 256
WIDTH = 128
N_BLOCKS = 4
K_SPEC = 16

# ============================================================
# IMPORT MODELS
# ============================================================
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from latent_loss import stress_loss, scale_loss  # scale_loss non serve se fai slope
from latent_loss import stress_loss, smooth_loss  # smooth_loss usa L e Z_field
from geometric_loss import GeometricLoss
from helper import patch_dataset_with_get_by_name, collate_skip

# ======

# ============================================================
# LOAD ENCODER (STAGE 1)
# ============================================================

print("📂 Loading frozen encoder (Stage 1)")
encoder = DiffusionEncoderOnly(
    latent_dim=LATENT_DIM,
    width=WIDTH,
    n_blocks=N_BLOCKS,
    dropout=0.0,
).to(DEVICE)

encoder.load_state_dict(torch.load(ENCODER_CKPT, map_location=DEVICE))
encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False

# ============================================================
# LOAD DECODER (STAGE 2)
# ============================================================

print("📂 Loading decoder (Stage 2)")
decoder = DiffusionNet(
    C_in=LATENT_DIM + K_SPEC,
    C_out=3,
    C_width=WIDTH,
    N_block=N_BLOCKS,
    with_gradient_features=True,
    dropout=0.0,
).to(DEVICE)

decoder.load_state_dict(torch.load(DECODER_CKPT, map_location=DEVICE))
decoder.eval()

# ============================================================
# UTILS
# ============================================================

def take_or_pad_evecs(evecs: torch.Tensor, k: int) -> torch.Tensor:
    """Ensure exactly k spectral components."""
    if evecs.shape[1] >= k:
        return evecs[:, :k]
    pad = torch.zeros(
        evecs.shape[0], k - evecs.shape[1],
        device=evecs.device, dtype=evecs.dtype
    )
    return torch.cat([evecs, pad], dim=1)


def save_obj(path: Path, V: np.ndarray, F: torch.Tensor) -> None:
    """Save mesh to OBJ."""
    with open(path, "w") as f:
        for v in V:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in F.cpu().numpy():
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def load_coo_sparse(data: np.lib.npyio.NpzFile, prefix: str, device: torch.device) -> torch.Tensor:
    """
    Rebuild a torch sparse COO tensor from fields:
      {prefix}_indices, {prefix}_values, {prefix}_shape
    """
    idx = data[f"{prefix}_indices"]
    val = data[f"{prefix}_values"]
    shp = data[f"{prefix}_shape"]

    idx = torch.as_tensor(idx, device=device)
    val = torch.as_tensor(val, device=device)

    # support both (nnz,2) and (2,nnz)
    if idx.ndim != 2:
        raise ValueError(f"{prefix}_indices has unexpected ndim={idx.ndim}")

    if idx.shape[0] != 2 and idx.shape[1] == 2:
        idx = idx.t()

    if idx.shape[0] != 2:
        raise ValueError(f"{prefix}_indices has unexpected shape {tuple(idx.shape)}")

    shp = tuple(int(x) for x in np.array(shp).tolist())
    return torch.sparse_coo_tensor(idx, val, size=shp, device=device).coalesce()

# ============================================================
# SELECT SUBJECT FILES
# ============================================================

npz_files = sorted(DATA_DIR.glob(f"{SUBJECT_ID}_GTready_*.npz"))

print(f"\nFound {len(npz_files)} files for subject {SUBJECT_ID}")
for p in npz_files:
    print(" ", p.name)

if len(npz_files) == 0:
    raise RuntimeError(f"No .npz files found for subject {SUBJECT_ID} in {DATA_DIR}")

# ============================================================
# INFERENCE LOOP
# ============================================================

latents = {}

for npz_path in npz_files:
    name = npz_path.stem
    print(f"\n▶ Processing {name}")

    data = np.load(npz_path)

    # geometry
    V = torch.tensor(data["verts"], device=DEVICE)                 # (N,3)
    faces = torch.tensor(data["faces"], device=DEVICE, dtype=torch.long)

    # spectral stuff (dense)
    mass  = torch.tensor(data["mass"], device=DEVICE)
    evals = torch.tensor(data["evals"], device=DEVICE)
    evecs = torch.tensor(data["evecs"], device=DEVICE)

    # operators (sparse COO in your files)
    L     = load_coo_sparse(data, "L", DEVICE)
    gradX = load_coo_sparse(data, "gradX", DEVICE)
    gradY = load_coo_sparse(data, "gradY", DEVICE)

    with torch.no_grad():
        Z_field, Z_global = encoder(
            V, mass, L, evals, evecs,
            faces, gradX, gradY,
            return_per_vertex=True,
            add_noise=False,
        )  # Z_field: (N,D)  Z_global: (1,D)

        S = take_or_pad_evecs(evecs, K_SPEC)       # (N,K)
        Z_in = torch.cat([Z_field, S], dim=1)      # (N,D+K)

        V_rec = decoder(
            Z_in, mass, L, evals, evecs,
            faces=faces, gradX=gradX, gradY=gradY
        )  # (N,3)

    latents[name] = Z_global.squeeze(0).detach().cpu()

    out = OUT_DIR / f"{name}_rec.obj"
    save_obj(out, V_rec.detach().cpu().numpy(), faces)
    print(f"💾 saved {out}")

# ============================================================
# LATENT DISTANCES (SANITY CHECK)
# ============================================================

print("\n🔎 Intra-subject latent distances:")
keys = list(latents.keys())

for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        d = torch.norm(latents[keys[i]] - latents[keys[j]]).item()
        print(f"  {keys[i]} vs {keys[j]} : {d:.4f}")
