#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# ============================================================
# PATHS
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

# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATENT_DIM = 256
WIDTH = 128
N_BLOCKS = 4
DROPOUT = 0.1

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


@torch.no_grad()
def latent_nn_3d_error(
    Z_src, V_src,
    Z_tgt, V_tgt,
    chunk_size=2048
):
    """
    Per ogni punto in src:
    - trova NN nello spazio latente in tgt
    - misura errore euclideo in 3D

    Ritorna:
    - errors: (N_src,)
    - nn_indices: (N_src,)
    """
    errors = []
    nn_indices = []

    for i in range(0, Z_src.shape[0], chunk_size):
        z = Z_src[i:i+chunk_size]           # (k, D)
        d = torch.cdist(z, Z_tgt, p=2)      # (k, N_tgt)
        nn = d.argmin(dim=1)                # (k,)

        v_src = V_src[i:i+chunk_size]       # (k, 3)
        v_tgt = V_tgt[nn]                   # (k, 3)

        err = (v_src - v_tgt).norm(dim=1)   # (k,)
        errors.append(err)
        nn_indices.append(nn)

    return torch.cat(errors), torch.cat(nn_indices)


def load_subject_variant(subject_id, variant):
    canon = np.load(DATA_CANON / f"{subject_id}_{variant}.npz")
    ops   = np.load(DATA_OPS   / f"{subject_id}_{variant}.npz")

    sample = {
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
    return sample

# ============================================================
# ENCODER
# ============================================================

@torch.no_grad()
def encode_per_vertex(model, sample):
    """
    Restituisce Z ∈ R^{N×D}
    """
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

# ============================================================
# LATENT CHAMFER
# ============================================================

@torch.no_grad()
def latent_chamfer(Z_A, Z_B, squared=True, chunk_size=2048):
    """
    Chamfer Distance tra due insiemi di latenti:
    Z_A: (N_A, D)
    Z_B: (N_B, D)
    """

    def min_dist(X, Y):
        mins = []
        for i in range(0, X.shape[0], chunk_size):
            x = X[i:i + chunk_size]
            d = torch.cdist(x, Y, p=2)
            if squared:
                d = d ** 2
            mins.append(d.min(dim=1).values)
        return torch.cat(mins)

    d_A = min_dist(Z_A, Z_B).mean()
    d_B = min_dist(Z_B, Z_A).mean()
    return d_A + d_B

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n🧬 LATENT CHAMFER PROBE (INFERENCE ONLY)\n")

    # --------------------------------------------------------
    # Encoder
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
    # Subjects
    # --------------------------------------------------------
    subjects = sorted(
        p.stem.replace("_original", "")
        for p in DATA_CANON.glob("*_original.npz")
    )
    assert len(subjects) >= 2

    sid_A = subjects[5]
    sid_B = subjects[7]

    print(f"🧑 Subject A: {sid_A}")
    print(f"🧑 Subject B: {sid_B}")

    # --------------------------------------------------------
    # Load samples
    # --------------------------------------------------------
    A_orig = load_subject_variant(sid_A, "original")
    A_rem  = load_subject_variant(sid_A, "remesh")
    B_orig = load_subject_variant(sid_B, "original")

    # --------------------------------------------------------
    # Encode
    # --------------------------------------------------------
    Z_A_orig = encode_per_vertex(encoder, A_orig)
    Z_A_rem  = encode_per_vertex(encoder, A_rem)
    Z_B_orig = encode_per_vertex(encoder, B_orig)

    print("📐 Latent shapes:")
    print("A original:", Z_A_orig.shape)
    print("A remesh  :", Z_A_rem.shape)
    print("B original:", Z_B_orig.shape)

    # --------------------------------------------------------
    # Distances
    # --------------------------------------------------------
    cd_intra = latent_chamfer(Z_A_orig, Z_A_rem)
    cd_inter = latent_chamfer(Z_A_orig, Z_B_orig)

    print("\n📊 RESULTS")
    print(f"Intra-subject (A orig vs remesh): {cd_intra.item():.4f}")
    print(f"Inter-subject (A vs B):           {cd_inter.item():.4f}")
    print(f"Ratio (inter / intra):            {(cd_inter / cd_intra).item():.2f}")


    # --------------------------------------------------------
    # Latent NN → index consistency test (inter-subject, same topology)
    # --------------------------------------------------------

    V_A = A_orig["verts"]
    V_B = B_orig["verts"]

    errors, nn_idx = latent_nn_3d_error(
        Z_A_orig, V_A,
        Z_B_orig, V_B
    )

    # --------------------------------------------------------
    # Heatmap errore 3D su mesh A
    # --------------------------------------------------------


    err_np = errors.detach().cpu().numpy()
    V_np   = V_A.detach().cpu().numpy()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    p = ax.scatter(
        V_np[:, 0],
        V_np[:, 1],
        V_np[:, 2],
        c=err_np,
        s=1,
        cmap="inferno"
    )
    ax.view_init(elev=90, azim=90)

    ax.set_title("Latent NN → 3D Error (Same Topology)")
    ax.set_axis_off()
    fig.colorbar(p, ax=ax, shrink=0.6, label="3D error")

    out_path = "latent_nn_3d_error_heatmap.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"💾 Saved heatmap to {out_path}")

    # ground-truth index = identity
    gt_idx = torch.arange(Z_A_orig.shape[0], device=DEVICE)

    index_error = (nn_idx != gt_idx).float()

    print("\n🔍 LATENT NN → INDEX CONSISTENCY (A orig → B orig)")
    print(f"Exact index match rate: {(1.0 - index_error.mean()).item() * 100:.2f}%")

    # 3D fallback (quanto sbaglia in spazio anche se l'indice non è esatto)
    print(f"Mean 3D error:   {errors.mean().item():.4f}")
    print(f"Median 3D error: {errors.median().item():.4f}")
    print(f"90% perc.:       {errors.quantile(0.9).item():.4f}")

    # --------------------------------------------------------
    # Istogramma distanza 3D index-wise (proxy geodetica)
    # --------------------------------------------------------

    gt_idx = torch.arange(V_A.shape[0], device=DEVICE)

    V_gt = V_A
    V_nn = V_B[nn_idx]

    geo_err = (V_gt - V_nn).norm(dim=1).detach().cpu().numpy()

    plt.figure(figsize=(6, 4))
    plt.hist(geo_err, bins=100, density=True)
    plt.xlabel("Distance on surface (3D proxy)")
    plt.ylabel("Density")
    plt.title("Distribution of NN-induced surface error")

    out_path = "latent_nn_surface_error_hist.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"💾 Saved histogram to {out_path}")

            
    # --------------------------------------------------------
    # Random baseline (same topology)
    # --------------------------------------------------------

    perm = torch.randperm(Z_B_orig.shape[0], device=DEVICE)
    Z_rand = Z_B_orig[perm]
    V_rand = V_B[perm]

    errors_rand, nn_idx_rand = latent_nn_3d_error(
        Z_A_orig, V_A,
        Z_rand,   V_rand
    )

    index_error_rand = (nn_idx_rand != gt_idx).float()

    print("\n🧪 RANDOM BASELINE (same topology)")
    print(f"Exact index match rate (rand): {(1.0 - index_error_rand.mean()).item() * 100:.2f}%")
    print(f"Mean random 3D error:          {errors_rand.mean().item():.4f}")

    # --------------------------------------------------------
    # Cross-topology stress test
    # --------------------------------------------------------
    print("\n🧪 CROSS-TOPOLOGY INTER-SUBJECT")

    tests = [
        ("original", "original"),
        ("remesh",   "original"),
        ("original", "remesh"),
        ("crop",     "original"),
        ("original", "crop"),
        ("remesh",   "crop"),
        ("crop",     "remesh"),
    ]

    ref = cd_intra.item()

    for va, vb in tests:
        try:
            A = load_subject_variant(sid_A, va)
            B = load_subject_variant(sid_B, vb)

            ZA = encode_per_vertex(encoder, A)
            ZB = encode_per_vertex(encoder, B)

            d = latent_chamfer(ZA, ZB).item()
            print(
                f"A_{va:8s} vs B_{vb:8s} | "
                f"d = {d:7.4f} | ratio = {d / (ref + 1e-9):5.2f}"
            )
        except FileNotFoundError:
            print(f"A_{va} vs B_{vb} | ❌ missing variant")

    print("\n✅ Done.")

# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()
