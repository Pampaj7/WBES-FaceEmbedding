#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from basic_test_grid import SpatialGrid

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

GRID_SIZE = 16
LATENT_DIM = 256
WIDTH = 128
N_BLOCKS = 4
DROPOUT = 0.1

MAX_SUBJECTS = 30

VARIANT_REF = "original"
VARIANT_QUERY = "crop"

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

@torch.no_grad()
def vertex_mse(A, B):
    return ((A["verts"] - B["verts"]) ** 2).sum(dim=1).mean().item()

@torch.no_grad()
def chamfer_distance(A, B):
    VA, VB = A["verts"], B["verts"]
    d1 = torch.cdist(VA, VB).min(dim=1).values.mean()
    d2 = torch.cdist(VB, VA).min(dim=1).values.mean()
    return (d1 + d2).item()

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
    return (GA[valid] - GB[valid]).norm(dim=1).mean().item()

# ============================================================
# RANKING UTILS
# ============================================================

def ordering_agreement(ref, test):
    n = len(ref)
    total = correct = 0
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
    print("🚀 Cross-topology ranking validation")

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

    print(f"🧑 Subjects: {len(subjects)}")

    # preload
    orig = {s: load_subject_variant(s, VARIANT_REF) for s in subjects}
    rem  = {s: load_subject_variant(s, VARIANT_QUERY) for s in subjects}

    spearman_ch, spearman_gr = [], []
    kendall_ch, kendall_gr = [], []
    agree_ch, agree_gr = [], []

    for sid in tqdm(subjects, desc="Per-subject ranking"):
        # reference ordering (MSE on original)
        ref_dists = []
        test_ids = []

        for other in subjects:
            if other == sid:
                continue
            ref_dists.append(vertex_mse(orig[sid], orig[other]))
            test_ids.append(other)

        ref_dists = np.array(ref_dists)

        # cross-topology distances
        chamfer_d = []
        grid_d = []

        for other in test_ids:
            chamfer_d.append(chamfer_distance(orig[sid], rem[other]))
            grid_d.append(grid_distance(grid, encoder, orig[sid], rem[other]))

        chamfer_d = np.array(chamfer_d)
        grid_d = np.array(grid_d)

        # ranking comparisons
        spearman_ch.append(spearmanr(ref_dists, chamfer_d)[0])
        spearman_gr.append(spearmanr(ref_dists, grid_d)[0])

        kendall_ch.append(kendalltau(ref_dists, chamfer_d)[0])
        kendall_gr.append(kendalltau(ref_dists, grid_d)[0])

        agree_ch.append(ordering_agreement(ref_dists, chamfer_d))
        agree_gr.append(ordering_agreement(ref_dists, grid_d))

    print("\n📊 RESULTS (mean ± std)")
    print("Chamfer:")
    print(f"  Spearman: {np.mean(spearman_ch):.3f} ± {np.std(spearman_ch):.3f}")
    print(f"  Kendall : {np.mean(kendall_ch):.3f} ± {np.std(kendall_ch):.3f}")
    print(f"  Agreement: {np.mean(agree_ch):.3f}")

    print("\nGrid:")
    print(f"  Spearman: {np.mean(spearman_gr):.3f} ± {np.std(spearman_gr):.3f}")
    print(f"  Kendall : {np.mean(kendall_gr):.3f} ± {np.std(kendall_gr):.3f}")
    print(f"  Agreement: {np.mean(agree_gr):.3f}")

    print("\n✅ Done.")

# ============================================================

if __name__ == "__main__":
    main()
