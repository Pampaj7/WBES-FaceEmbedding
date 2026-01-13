#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path

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

GRID_SIZE = 8
LATENT_DIM = 256
WIDTH = 128
N_BLOCKS = 4
DROPOUT = 0.1

# Dataset paths
DATA_CANON = Path(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/REMESH/data_CANONICAL"
)
DATA_OPS = Path(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/REMESH/npz_data_topo_500_withops"
)

# Encoder checkpoint
ENCODER_CKPT = Path(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/gt_encdec/remeshing/cross_topo_model/encoder_stage1_multitopo_second_try/encoder_stage1_epoch50.pth"
)

# ============================================================
# DATA LOADING
# ============================================================

def load_sparse_tensor(indices, values, shape, device):
    indices = torch.from_numpy(indices).long().to(device)
    values  = torch.from_numpy(values).float().to(device)
    shape   = tuple(shape.tolist())
    return torch.sparse_coo_tensor(indices, values, size=shape)


def load_subject_variant(subject_id, variant):
    canon = np.load(DATA_CANON / f"{subject_id}_{variant}.npz")
    ops   = np.load(DATA_OPS   / f"{subject_id}_{variant}.npz")

    sample = {
        # Canonical geometry
        "verts": torch.from_numpy(canon["V"]).float().to(DEVICE),
        "faces": torch.from_numpy(canon["F"]).long().to(DEVICE),

        # Intrinsic operators (sparse)
        "mass":  torch.from_numpy(ops["mass"]).float().to(DEVICE),

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


# ============================================================
# ENCODER WRAPPER
# ============================================================

@torch.no_grad()
def encode_per_vertex(model, sample):
    """
    Restituisce Z_field ∈ R^{N×D}
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

# ============================================================
# DISTANCE
# ============================================================

def identity_distance(grid_A, mask_A, grid_B, mask_B):
    """
    Mean L2 over cells present in BOTH.
    """
    valid = mask_A & mask_B
    if valid.sum() == 0:
        return torch.tensor(float("inf"), device=grid_A.device)

    diff = grid_A[valid] - grid_B[valid]
    return diff.norm(dim=1).mean()

def compute_pair_distance(encoder, grid, sid_A, var_A, sid_B, var_B):
    """
    Calcola la distanza grid-based tra (sid_A, var_A) e (sid_B, var_B)
    """
    A = load_subject_variant(sid_A, var_A)
    B = load_subject_variant(sid_B, var_B)

    Z_A = encode_per_vertex(encoder, A)
    Z_B = encode_per_vertex(encoder, B)

    G_A, M_A = grid(A["verts"], Z_A)
    G_B, M_B = grid(B["verts"], Z_B)

    d = identity_distance(G_A, M_A, G_B, M_B)
    return d.item(), int((M_A & M_B).sum())

# ============================================================
# MAIN
# ============================================================

def main():
    print("🚀 Running grid-based identity probe (INFERENCE ONLY)")

    # --------------------------------------------------------
    # Load encoder
    # --------------------------------------------------------
    encoder = DiffusionEncoderOnly(
        latent_dim=LATENT_DIM,
        width=WIDTH,
        n_blocks=N_BLOCKS,
        dropout=DROPOUT,
    ).to(DEVICE)

    encoder.load_state_dict(torch.load(ENCODER_CKPT, map_location=DEVICE))
    encoder.eval()

    print("✅ Encoder loaded")

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------
    grid = SpatialGrid(grid_size=GRID_SIZE)

    # --------------------------------------------------------
    # Subjects
    # --------------------------------------------------------
    subjects = sorted(
        p.stem.replace("_original", "")
        for p in DATA_CANON.glob("*_original.npz")
    )
    assert len(subjects) >= 2, "Need at least two subjects"

    sid_A = subjects[5]
    sid_B = subjects[7]

    print(f"🧑 Subject A: {sid_A}")
    print(f"🧑 Subject B: {sid_B}")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    A_orig = load_subject_variant(sid_A, "original")
    A_rem  = load_subject_variant(sid_A, "remesh")
    B_orig = load_subject_variant(sid_B, "original")
    # --- vertices for visualization (numpy) ---
    V_A = A_orig["verts"].detach().cpu().numpy()
    V_B = B_orig["verts"].detach().cpu().numpy()

    # --------------------------------------------------------
    # Encode
    # --------------------------------------------------------
    Z_A_orig = encode_per_vertex(encoder, A_orig)
    Z_A_rem  = encode_per_vertex(encoder, A_rem)
    Z_B_orig = encode_per_vertex(encoder, B_orig)

    # --------------------------------------------------------
    # Grid pooling
    # --------------------------------------------------------
    G_A_orig, M_A_orig = grid(A_orig["verts"], Z_A_orig)
    G_A_rem,  M_A_rem  = grid(A_rem["verts"],  Z_A_rem)
    G_B_orig, M_B_orig = grid(B_orig["verts"], Z_B_orig)
    from plot_grid_cell_errors_plotly import main as plot_cells

    plot_cells(
        G_A_orig, M_A_orig,
        G_B_orig, M_B_orig,
        V_A, V_B
    )


    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------
    print("\n📦 Active cells:")
    print("A original:", int(M_A_orig.sum()))
    print("A remesh  :", int(M_A_rem.sum()))
    print("B original:", int(M_B_orig.sum()))

    # --------------------------------------------------------
    # Distances
    # --------------------------------------------------------
    intra = identity_distance(G_A_orig, M_A_orig, G_A_rem, M_A_rem)
    inter = identity_distance(G_A_orig, M_A_orig, G_B_orig, M_B_orig)

    print("\n📊 RESULTS")
    print(f"Intra-subject (A orig vs remesh): {intra.item():.4f}")
    print(f"Inter-subject (A vs B):           {inter.item():.4f}")
    print(f"Ratio (inter / intra):            {(inter / intra).item():.2f}")

    # --------------------------------------------------------
    # Cross-topology stress test
    # --------------------------------------------------------
    print("\n🧪 CROSS-TOPOLOGY INTER-SUBJECT TESTS")

    tests = [
        ("original", "original"),
        ("remesh",   "original"),
        ("original", "remesh"),
        ("crop",     "original"),
        ("original", "crop"),
        ("remesh",   "crop"),
        ("crop",     "remesh"),
    ]

    # riferimento intra-identità
    d_ref = intra.item()

    for va, vb in tests:
        try:
            d, n_cells = compute_pair_distance(
                encoder, grid,
                sid_A, va,
                sid_B, vb
            )
            ratio = d / (d_ref + 1e-9)
            print(
                f"A_{va:8s} vs B_{vb:8s} | "
                f"d = {d:7.4f} | "
                f"ratio = {ratio:5.2f} | "
                f"cells = {n_cells:3d}"
            )
        except FileNotFoundError:
            print(f"A_{va} vs B_{vb} | ❌ missing variant")

    print("\n✅ Done.")

# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()
