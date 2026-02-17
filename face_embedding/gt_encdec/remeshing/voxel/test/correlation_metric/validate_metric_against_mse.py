#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from pathlib import Path
import sys
import matplotlib.pyplot as plt
sys.path.append(str(Path(__file__).resolve().parents[1]))

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

MAX_SUBJECTS = 30   # ⚠️ per evitare O(N²) ingestibile
VARIANT = "original"

DATA_CANON = Path(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/REMESH/data_CANONICAL"
)
DATA_OPS = Path(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/REMESH/npz_data_topo_500_withops"
)

ENCODER_CKPT = Path(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/gt_encdec/remeshing/cross_topo_model/encoder_stage1_multitopo_second_try/encoder_stage1_epoch50.pth"
)

# ============================================================
# PERTURBATION CONFIG
# ============================================================

APPLY_PERTURBATION = False   # <-- switch ON / OFF

PERTURB_MAX_ANGLE_DEG = 30.0
PERTURB_SCALE_RANGE = (0.7, 1.3)


# ============================================================
# DATA LOADING (riuso diretto)
# ============================================================

def load_sparse_tensor(indices, values, shape, device):
    indices = torch.from_numpy(indices).long().to(device)
    values  = torch.from_numpy(values).float().to(device)
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

# ============================================================
# METRICS
# ============================================================


def random_similarity_transform(
    V,
    max_angle_deg=30.0,
    scale_range=(0.7, 1.3),
):
    """
    Applica una rotazione + scala + traslazione più forte ai vertici.
    V: (N, 3) torch tensor
    """
    device = V.device

    # --- random rotation ---
    angle = torch.empty(1, device=device).uniform_(
        -max_angle_deg, max_angle_deg
    ) * np.pi / 180.0

    axis = torch.randn(3, device=device)
    axis = axis / axis.norm()

    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], device=device)

    R = (
        torch.eye(3, device=device)
        + torch.sin(angle) * K
        + (1 - torch.cos(angle)) * (K @ K)
    )

    # --- scale (isotropic, ma più forte) ---
    s = torch.empty(1, device=device).uniform_(*scale_range)

    # --- translation (KEY PART) ---
    # scala proporzionale al bounding box → realistico
    bbox_size = V.max(dim=0).values - V.min(dim=0).values
    t = 0.2 * bbox_size * torch.randn(3, device=device)

    return (V @ R.T) * s + t


@torch.no_grad()
def vertex_mse(A, B):
    diff = A["verts"] - B["verts"]
    return (diff ** 2).sum(dim=1).mean().item()

@torch.no_grad()
def vertex_rmse(A, B):
    """
    MODIFICA:
    Usiamo RMSE invece di MSE.
    Ora la metrica è in unità di distanza (coerente con Chamfer).
    """
    diff = A["verts"] - B["verts"]
    return torch.sqrt((diff ** 2).sum(dim=1).mean()).item()


@torch.no_grad()
def chamfer_distance(A, B):
    VA = A["verts"]
    VB = B["verts"]
    d1 = torch.cdist(VA, VB).min(dim=1).values.mean()
    d2 = torch.cdist(VB, VA).min(dim=1).values.mean()
    return (d1 + d2).item()

@torch.no_grad()
def chamfer_rms(A, B):
    """
    MODIFICA:
    Chamfer bidirezionale simmetrica in forma RMS.

        sqrt( 0.5 * ( E[d(A→B)^2] + E[d(B→A)^2] ) )

    Coerente dimensionalmente con RMSE.
    """
    VA = A["verts"]
    VB = B["verts"]

    d1 = torch.cdist(VA, VB).min(dim=1).values
    d2 = torch.cdist(VB, VA).min(dim=1).values

    return torch.sqrt(
        0.5 * (d1.pow(2).mean() + d2.pow(2).mean())
    ).item()


@torch.no_grad()
def encode_per_vertex(model, sample):
    Z, _ = model(
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
    return Z.squeeze(0)

def grid_distance(grid, encoder, A, B):
    ZA = encode_per_vertex(encoder, A)
    ZB = encode_per_vertex(encoder, B)
    GA, MA = grid(A["verts"], ZA)
    GB, MB = grid(B["verts"], ZB)

    valid = MA & MB
    if valid.sum() == 0:
        return np.nan

    diff = GA[valid] - GB[valid]
    return diff.norm(dim=1).mean().item()

def grid_distance_rms(grid, encoder, A, B):
    """
    MODIFICA:
    Anche la distanza grid-based diventa RMS.
    Prima era media delle norme → ora è sqrt(mean(||diff||^2)).
    """

    ZA = encode_per_vertex(encoder, A)
    ZB = encode_per_vertex(encoder, B)

    GA, MA = grid(A["verts"], ZA)
    GB, MB = grid(B["verts"], ZB)

    valid = MA & MB
    if valid.sum() == 0:
        return np.nan

    diff = GA[valid] - GB[valid]

    return torch.sqrt(diff.pow(2).sum(dim=1).mean()).item()


# ============================================================
# ORDERING AGREEMENT
# ============================================================

def ordering_agreement(ref, test):
    assert len(ref) == len(test)
    n = len(ref)
    total = 0
    correct = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if (ref[i] < ref[j]) == (test[i] < test[j]):
                correct += 1
    return correct / total

# ============================================================
# MAIN
# ============================================================

def main():
    print("🚀 Metric validation vs vertex-wise MSE")

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
    )[:MAX_SUBJECTS]

    print(f"🧑 Using {len(subjects)} subjects")

    samples = {
        sid: load_subject_variant(sid, VARIANT)
        for sid in subjects
    }

    mse_vals = []
    chamfer_vals = []
    grid_vals = []

    pairs = list(combinations(subjects, 2))

    for sid_a, sid_b in tqdm(pairs):
        A = samples[sid_a]
        B = samples[sid_b]

        if APPLY_PERTURBATION:
            B = dict(B)  # shallow copy
            B["verts"] = random_similarity_transform(
                B["verts"],
                max_angle_deg=PERTURB_MAX_ANGLE_DEG,
                scale_range=PERTURB_SCALE_RANGE,
            )

        # --------------------------------------------------------
        # MODIFICA: tutte le metriche ora sono in forma RMS
        # --------------------------------------------------------

        mse_vals.append(vertex_rmse(A, B))
        chamfer_vals.append(chamfer_rms(A, B))
        grid_vals.append(grid_distance_rms(grid, encoder, A, B))

    mse_vals = np.array(mse_vals)
    chamfer_vals = np.array(chamfer_vals)
    grid_vals = np.array(grid_vals)

    # ========================================================
    # STATS
    # ========================================================

    print("\n📊 Correlations (vs MSE)")

    print("Chamfer:")
    print("  Pearson :", pearsonr(mse_vals, chamfer_vals)[0])
    print("  Spearman:", spearmanr(mse_vals, chamfer_vals)[0])
    print("  Ordering agreement:",
          ordering_agreement(mse_vals, chamfer_vals))

    print("\nGrid:")
    print("  Pearson :", pearsonr(mse_vals, grid_vals)[0])
    print("  Spearman:", spearmanr(mse_vals, grid_vals)[0])
    print("  Ordering agreement:",
          ordering_agreement(mse_vals, grid_vals))

    # ========================================================
    # PLOTS
    # ========================================================

    # ========================================================
    # PLOTS (properly normalized, y=x diagonal, metrics inside)
    # ========================================================

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)

    # Global limits for true diagonal
    lim_min = min(
        mse_vals.min(),
        chamfer_vals.min(),
        grid_vals.min()
    )
    lim_max = max(
        mse_vals.max(),
        chamfer_vals.max(),
        grid_vals.max()
    )

    # ------------------------------------------------
    # MSE vs Chamfer
    # ------------------------------------------------
    axes[0].scatter(
        mse_vals,
        chamfer_vals,
        s=22,
        alpha=0.6
    )

    axes[0].plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        linestyle="--",
        linewidth=1
    )

    axes[0].set_title("RMSE vs Chamfer (RMS)")
    axes[0].set_xlabel("Vertex-wise RMSE")
    axes[0].set_ylabel("Distance")
    axes[0].set_xlim(lim_min, lim_max)
    axes[0].set_ylim(lim_min, lim_max)
    axes[0].grid(True, alpha=0.3)

    axes[0].text(
        0.05, 0.95,
        f"Pearson   = {pearsonr(mse_vals, chamfer_vals)[0]:.3f}\n"
        f"Spearman  = {spearmanr(mse_vals, chamfer_vals)[0]:.3f}\n"
        f"Agreement = {ordering_agreement(mse_vals, chamfer_vals):.3f}",
        transform=axes[0].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.85)
    )

    # ------------------------------------------------
    # MSE vs Grid
    # ------------------------------------------------
    axes[1].scatter(
        mse_vals,
        grid_vals,
        s=22,
        alpha=0.6
    )

    axes[1].plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        linestyle="--",
        linewidth=1
    )

    axes[1].set_title("RMSE vs Grid-based distance (RMS)")
    axes[1].set_xlabel("Vertex-wise RMSE")
    axes[1].set_xlim(lim_min, lim_max)
    axes[1].set_ylim(lim_min, lim_max)
    axes[1].grid(True, alpha=0.3)

    axes[1].text(
        0.05, 0.95,
        f"Pearson   = {pearsonr(mse_vals, grid_vals)[0]:.3f}\n"
        f"Spearman  = {spearmanr(mse_vals, grid_vals)[0]:.3f}\n"
        f"Agreement = {ordering_agreement(mse_vals, grid_vals):.3f}",
        transform=axes[1].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.85)
    )

    plt.tight_layout()
    suffix = "perturbed" if APPLY_PERTURBATION else "aligned"
    plt.savefig(f"mse_vs_chamfer_vs_grid_{VARIANT}_perturbation:{APPLY_PERTURBATION}_{suffix}_gridSize:{GRID_SIZE}.png", dpi=200)
    plt.close()


# ============================================================

if __name__ == "__main__":
    main()
