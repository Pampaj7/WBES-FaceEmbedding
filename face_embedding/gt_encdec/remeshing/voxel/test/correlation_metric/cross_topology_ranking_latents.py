
import torch
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATENT_DIM = 256
WIDTH = 128
N_BLOCKS = 4
DROPOUT = 0.1

MAX_SUBJECTS = 30

VARIANT_REF = "original"
VARIANT_QUERY = "remesh"   # crop / remesh / etc.

# ---------------- PERTURBATION ----------------

APPLY_PERTURBATION = False   # <<< SWITCH
PERTURB_MAX_ANGLE_DEG = 30.0
PERTURB_SCALE_RANGE = (0.7, 1.3)
PERTURB_TRANSLATION_FACTOR = 0.2

# ------------------------------------------------------------

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

# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------


def load_sparse_tensor(indices, values, shape, device):
    indices = torch.from_numpy(indices).long().to(device)
    values = torch.from_numpy(values).float().to(device)
    shape = tuple(shape.tolist())
    return torch.sparse_coo_tensor(indices, values, size=shape)


def load_subject_variant(subject_id, variant):
    canon = np.load(DATA_CANON / f"{subject_id}_{variant}.npz")
    ops = np.load(DATA_OPS / f"{subject_id}_{variant}.npz")

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

# ------------------------------------------------------------
# RANDOM SIMILARITY TRANSFORM (STRONG, CONTROLLED)
# ------------------------------------------------------------


def random_similarity_transform(
    V,
    max_angle_deg,
    scale_range,
    translation_factor,
):
    device = V.device

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

    s = torch.empty(1, device=device).uniform_(*scale_range)

    bbox = V.max(dim=0).values - V.min(dim=0).values
    t = translation_factor * bbox * torch.randn(3, device=device)

    return (V @ R.T) * s + t

# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------


@torch.no_grad()
def vertex_mse(A, B):
    return ((A["verts"] - B["verts"]) ** 2).sum(dim=1).mean().item()

@torch.no_grad()
def vertex_rmse(A, B):
    """
    MODIFICA:
    Usiamo RMSE invece di MSE.
    Ora è dimensionalmente coerente con Chamfer RMS.
    """
    diff = A["verts"] - B["verts"]
    return torch.sqrt((diff ** 2).sum(dim=1).mean()).item()

@torch.no_grad()
def chamfer_distance(A, B):
    VA, VB = A["verts"], B["verts"]
    d1 = torch.cdist(VA, VB).min(dim=1).values.mean()
    d2 = torch.cdist(VB, VA).min(dim=1).values.mean()
    return (d1 + d2).item()

@torch.no_grad()
def vertex_rmse(A, B):
    """
    MODIFICA:
    Usiamo RMSE invece di MSE.
    Ora è dimensionalmente coerente con Chamfer RMS.
    """
    diff = A["verts"] - B["verts"]
    return torch.sqrt((diff ** 2).sum(dim=1).mean()).item()


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


@torch.no_grad()
def latent_chamfer_distance(Z_A, Z_B, chunk_size=2048):
    def min_dist(X, Y):
        out = []
        for i in range(0, X.shape[0], chunk_size):
            d = torch.cdist(X[i:i+chunk_size], Y)
            out.append(d.min(dim=1).values)
        return torch.cat(out)

    return (min_dist(Z_A, Z_B).mean() +
            min_dist(Z_B, Z_A).mean()).item()

@torch.no_grad()
def latent_chamfer_rms(Z_A, Z_B, chunk_size=2048):
    """
    MODIFICA:
    Anche la Chamfer nei latenti ora è RMS coerente.
    """

    def min_dist(X, Y):
        out = []
        for i in range(0, X.shape[0], chunk_size):
            d = torch.cdist(X[i:i+chunk_size], Y)
            out.append(d.min(dim=1).values)
        return torch.cat(out)

    d1 = min_dist(Z_A, Z_B)
    d2 = min_dist(Z_B, Z_A)

    return torch.sqrt(
        0.5 * (d1.pow(2).mean() + d2.pow(2).mean())
    ).item()

@torch.no_grad()
def chamfer_rms(A, B):
    """
    MODIFICA:
    Chamfer bidirezionale simmetrica in forma RMS.
    sqrt( 0.5 * (E[d(A→B)^2] + E[d(B→A)^2]) )
    """
    VA, VB = A["verts"], B["verts"]

    d1 = torch.cdist(VA, VB).min(dim=1).values
    d2 = torch.cdist(VB, VA).min(dim=1).values

    return torch.sqrt(
        0.5 * (d1.pow(2).mean() + d2.pow(2).mean())
    ).item()
    
# ------------------------------------------------------------
# ORDERING AGREEMENT
# ------------------------------------------------------------


def ordering_agreement(ref, test):
    n = len(ref)
    tot = cor = 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += 1
            if (ref[i] < ref[j]) == (test[i] < test[j]):
                cor += 1
    return cor / tot

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------


def main():
    print("🚀 Cross-topology correlation plots")
    print(f"Perturbation: {APPLY_PERTURBATION}")

    encoder = DiffusionEncoderOnly(
        latent_dim=LATENT_DIM,
        width=WIDTH,
        n_blocks=N_BLOCKS,
        dropout=DROPOUT,
    ).to(DEVICE)

    encoder.load_state_dict(torch.load(ENCODER_CKPT, map_location=DEVICE))
    encoder.eval()

    subjects = sorted(
        p.stem.replace("_original", "")
        for p in DATA_CANON.glob("*_original.npz")
    )[:MAX_SUBJECTS]

    ref = {s: load_subject_variant(s, VARIANT_REF) for s in subjects}
    qry = {s: load_subject_variant(s, VARIANT_QUERY) for s in subjects}

    mse_vals, chamfer_vals, latent_vals = [], [], []

    pairs = list(combinations(subjects, 2))

    for a, b in tqdm(pairs):
        A = ref[a]
        B = qry[b]

        if APPLY_PERTURBATION:
            B = dict(B)
            B["verts"] = random_similarity_transform(
                B["verts"],
                PERTURB_MAX_ANGLE_DEG,
                PERTURB_SCALE_RANGE,
                PERTURB_TRANSLATION_FACTOR,
            )

        mse_vals.append(vertex_rmse(ref[a], ref[b]))
        chamfer_vals.append(chamfer_rms(A, B))

        ZA = encode_per_vertex(encoder, A)
        ZB = encode_per_vertex(encoder, B)
        latent_vals.append(latent_chamfer_rms(ZA, ZB))

    mse_vals = np.array(mse_vals)
    chamfer_vals = np.array(chamfer_vals)
    latent_vals = np.array(latent_vals)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    stats = {
        "Chamfer": (
            pearsonr(mse_vals, chamfer_vals)[0],
            spearmanr(mse_vals, chamfer_vals)[0],
            ordering_agreement(mse_vals, chamfer_vals),
        ),
        "Latent": (
            pearsonr(mse_vals, latent_vals)[0],
            spearmanr(mse_vals, latent_vals)[0],
            ordering_agreement(mse_vals, latent_vals),
        )
    }

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)

    lim_min = min(mse_vals.min(), chamfer_vals.min(), latent_vals.min())
    lim_max = max(mse_vals.max(), chamfer_vals.max(), latent_vals.max())

    for ax, vals, name in zip(
        axes,
        [chamfer_vals, latent_vals],
        ["Chamfer", "Latent Chamfer"]
    ):
        ax.scatter(mse_vals, vals, s=20, alpha=0.6)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "--")
        ax.set_xlabel("Vertex-wise RMSE")
        ax.set_title(f"RMSE vs {name} (cross-topology)")
        ax.grid(True, alpha=0.3)

        p, s, a = stats["Chamfer" if "Chamfer" in name and name ==
                        "Chamfer" else "Latent"]
        ax.text(
            0.05, 0.95,
            f"Pearson   = {p:.3f}\n"
            f"Spearman  = {s:.3f}\n"
            f"Agreement = {a:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round", alpha=0.85)
        )

    axes[0].set_ylabel("Distance")

    plt.tight_layout()
    out = (
        f"crossTopo_{VARIANT_QUERY}"
        f"_perturb:{APPLY_PERTURBATION}.png"
    )
    plt.savefig(out, dpi=200)
    plt.close()

    print(f"💾 Saved plot to {out}")
    print("Mean RMSE:", mse_vals.mean())
    print("Mean Chamfer RMS:", chamfer_vals.mean())
    print("Mean Latent RMS:", latent_vals.mean())
    print("✅ Done.")


if __name__ == "__main__":
    main()
