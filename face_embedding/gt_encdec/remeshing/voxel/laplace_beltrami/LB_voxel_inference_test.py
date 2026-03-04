#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from LB_voxelization import LBVoxelizer

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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_CANON = Path(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/REMESH/data_CANONICAL"
)

DATA_OPS = Path(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/REMESH/npz_data_topo_500_withops"
)
OUT_DIR = Path("lb_plots")
OUT_DIR.mkdir(exist_ok=True)

N_EVECS = 4
BINS = 4

# ============================================================
# UTILS
# ============================================================


def random_rotation():
    a, b = np.random.uniform(0, 2 * np.pi, size=2)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(a), -np.sin(a)],
                   [0, np.sin(a),  np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)],
                   [0,          1, 0],
                   [-np.sin(b), 0, np.cos(b)]])
    return Rx @ Ry


def transform_vertices(V, scale=1.7):
    R = random_rotation()
    Vt = (V.cpu().numpy() @ R.T) * scale
    return torch.from_numpy(Vt).float().to(V.device)


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

# ============================================================
# LB GRID (CORE)
# ============================================================


def compute_lb_voxel_id(evecs, n_evecs, bins):
    Phi = evecs[:, :n_evecs]
    Phi = (Phi - Phi.mean(0)) / (Phi.std(0) + 1e-8)

    coords = []
    for i in range(n_evecs):
        edges = torch.linspace(
            Phi[:, i].min(),
            Phi[:, i].max(),
            bins + 1,
            device=Phi.device,
        )[1:-1]
        coords.append(torch.bucketize(Phi[:, i], edges))

    coords = torch.stack(coords, dim=1)

    voxel_id = torch.zeros(
        coords.shape[0], dtype=torch.long, device=Phi.device
    )
    for i in range(n_evecs):
        voxel_id += coords[:, i] * (bins ** i)

    return voxel_id

# ============================================================
# PLOT
# ============================================================


def plot_mesh_regions(V, F, region_id, title, out_html):
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=V[:, 0],
                y=V[:, 1],
                z=V[:, 2],
                i=F[:, 0],
                j=F[:, 1],
                k=F[:, 2],
                intensity=region_id,
                colorscale="Turbo",
                showscale=True,
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(out_html)
    print(f"💾 Saved: {out_html}")

# ============================================================
# MAIN — L’ESPERIMENTO CHE VUOI TU
# ============================================================


def main():
    print("🧪 LB partition test: inter-subject, inter-topology, inter-pose")

    # --------------------------------------------------------
    # Subjects
    # --------------------------------------------------------
    sid_A = "id0000_GTready"   # original, pose 1
    sid_B = "id0001_GTready"   # remesh, pose 2

    sample_A = load_subject_variant(sid_A, "original")
    sample_B = load_subject_variant(sid_B, "remesh")

    # rotate + scale ONLY subject B
    sample_B_t = dict(sample_B)
    sample_B_t["verts"] = transform_vertices(sample_B["verts"])

    # --------------------------------------------------------
    # LB intrinsic regions
    # --------------------------------------------------------
    voxel_id_A = compute_lb_voxel_id(
        sample_A["evecs"], n_evecs=N_EVECS, bins=BINS
    )

    voxel_id_B = compute_lb_voxel_id(
        sample_B["evecs"], n_evecs=N_EVECS, bins=BINS
    )

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------
    plot_mesh_regions(
        sample_A["verts"].cpu().numpy(),
        sample_A["faces"].cpu().numpy(),
        voxel_id_A.cpu().numpy(),
        "LB regions – Subject A (original, pose 1)",
        OUT_DIR / "lb_subjectA_original_pose1.html",
    )

    plot_mesh_regions(
        sample_B_t["verts"].cpu().numpy(),
        sample_B["faces"].cpu().numpy(),
        voxel_id_B.cpu().numpy(),
        "LB regions – Subject B (remesh, pose 2)",
        OUT_DIR / "lb_subjectB_remesh_pose2.html",
    )

    print("✅ Done. Open the two HTML files and compare visually.")


if __name__ == "__main__":
    main()
