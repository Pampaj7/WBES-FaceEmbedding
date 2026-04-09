#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


DATASET_DIR = Path(__file__).resolve().parent / "REMESH" / "npz_data_topo_500_withops"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an interactive HTML preview of a remeshed face mesh."
    )
    parser.add_argument(
        "--subject",
        help="Subject stem without variant suffix, for example: id0000_GTready",
    )
    parser.add_argument(
        "--variant",
        default="remesh",
        choices=["original", "remesh", "crop", "noisy"],
        help="Variant to visualize.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output HTML path. Defaults to datasets/preview_mesh_<subject>_<variant>.html",
    )
    parser.add_argument(
        "--hide-wireframe",
        action="store_true",
        help="Hide the unique triangle edges overlay.",
    )
    return parser.parse_args()


def discover_first_subject(dataset_dir: Path) -> str:
    files = sorted(dataset_dir.glob("*_remesh.npz"))
    if not files:
        raise FileNotFoundError(f"No remesh files found in {dataset_dir}")
    return files[0].name.replace("_remesh.npz", "")


def load_mesh(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)

    if {"verts", "faces"} <= set(data.files):
        verts = np.asarray(data["verts"], dtype=np.float64)
        faces = np.asarray(data["faces"], dtype=np.int32)
    elif {"V", "F"} <= set(data.files):
        verts = np.asarray(data["V"], dtype=np.float64)
        faces = np.asarray(data["F"], dtype=np.int32)
    else:
        raise KeyError(f"Unsupported NPZ layout in {npz_path}: {data.files}")

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Faces in {npz_path} are not triangular: {faces.shape}")

    return verts, faces


def unique_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    return np.unique(np.sort(edges, axis=1), axis=0)


def boundary_and_non_manifold_edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique[counts == 1], unique[counts > 2]


def triangle_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def edge_trace(
    verts: np.ndarray,
    edges: np.ndarray,
    *,
    name: str,
    color: str,
    width: int,
) -> go.Scatter3d:
    if edges.size == 0:
        return go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode="lines",
            name=name,
            showlegend=True,
        )

    coords = verts[edges]
    x = np.full((coords.shape[0], 3), np.nan, dtype=np.float64)
    y = np.full((coords.shape[0], 3), np.nan, dtype=np.float64)
    z = np.full((coords.shape[0], 3), np.nan, dtype=np.float64)

    x[:, :2] = coords[:, :, 0]
    y[:, :2] = coords[:, :, 1]
    z[:, :2] = coords[:, :, 2]

    return go.Scatter3d(
        x=x.ravel(),
        y=y.ravel(),
        z=z.ravel(),
        mode="lines",
        line=dict(color=color, width=width),
        name=name,
        hoverinfo="skip",
    )


def mesh_trace(verts: np.ndarray, faces: np.ndarray, *, name: str) -> go.Mesh3d:
    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color="#9ec5fe",
        opacity=0.9,
        flatshading=True,
        lighting=dict(ambient=0.45, diffuse=0.7, specular=0.15, roughness=0.8),
        lightposition=dict(x=120, y=160, z=200),
        name=name,
        hovertemplate="v=(%{x:.4f}, %{y:.4f}, %{z:.4f})<extra></extra>",
    )


def summarize_mesh(verts: np.ndarray, faces: np.ndarray) -> dict[str, int | float]:
    areas = triangle_areas(verts, faces)
    bbox_diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    eps = max(np.median(areas) * 1e-8, bbox_diag * 1e-12, np.finfo(np.float64).eps)
    degenerate_faces = int(np.count_nonzero(areas <= eps))

    boundary_edges, non_manifold_edges = boundary_and_non_manifold_edges(faces)
    bad_index_faces = int(
        np.count_nonzero((faces < 0).any(axis=1) | (faces >= len(verts)).any(axis=1))
    )

    return {
        "verts": int(len(verts)),
        "faces": int(len(faces)),
        "unique_edges": int(len(unique_edges(faces))),
        "boundary_edges": int(len(boundary_edges)),
        "non_manifold_edges": int(len(non_manifold_edges)),
        "degenerate_faces": degenerate_faces,
        "bad_index_faces": bad_index_faces,
    }


def build_figure(subject: str, variant: str, verts: np.ndarray, faces: np.ndarray, show_wireframe: bool) -> go.Figure:
    summary = summarize_mesh(verts, faces)
    boundary_edges, non_manifold_edges = boundary_and_non_manifold_edges(faces)
    all_edges = unique_edges(faces)

    fig = go.Figure()
    fig.add_trace(mesh_trace(verts, faces, name=f"{variant} surface"))

    if show_wireframe:
        fig.add_trace(
            edge_trace(
                verts,
                all_edges,
                name="wireframe",
                color="rgba(40, 40, 40, 0.35)",
                width=2,
            )
        )

    fig.add_trace(
        edge_trace(
            verts,
            boundary_edges,
            name=f"boundary edges ({len(boundary_edges)})",
            color="#e63946",
            width=5,
        )
    )
    fig.add_trace(
        edge_trace(
            verts,
            non_manifold_edges,
            name=f"non-manifold edges ({len(non_manifold_edges)})",
            color="#ff9f1c",
            width=7,
        )
    )

    stats = (
        f"V={summary['verts']} | F={summary['faces']} | "
        f"boundary={summary['boundary_edges']} | "
        f"non-manifold={summary['non_manifold_edges']} | "
        f"degenerate={summary['degenerate_faces']} | "
        f"bad-index={summary['bad_index_faces']}"
    )

    fig.update_layout(
        title=f"{subject}_{variant}<br><sup>{stats}</sup>",
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.75)"),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=1.4, y=0.9, z=0.8)),
        ),
    )
    return fig


def main() -> None:
    args = parse_args()

    subject = args.subject or discover_first_subject(DATASET_DIR)
    mesh_path = DATASET_DIR / f"{subject}_{args.variant}.npz"
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    out_path = args.out or (Path(__file__).resolve().parent / f"preview_mesh_{subject}_{args.variant}.html")

    verts, faces = load_mesh(mesh_path)
    fig = build_figure(subject, args.variant, verts, faces, show_wireframe=not args.hide_wireframe)
    fig.write_html(out_path, include_plotlyjs="cdn")

    print(f"Loaded: {mesh_path}")
    print(f"Saved HTML to: {out_path}")


if __name__ == "__main__":
    main()
