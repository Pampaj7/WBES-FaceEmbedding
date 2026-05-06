#!/usr/bin/env python3
"""Visualize the source deformation used by NICP to estimate correspondences."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FACBENCH_DIR = REPO_ROOT / "faceBench"
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FACBENCH_DIR))
sys.path.insert(0, str(THIS_DIR))

import facebench as fb
from facebench.rigid_aligners.icp import prealign_by_bbox
from mesh_npz_utils import load_normalized_vertices_npz


def sample_points(V: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or len(V) <= max_points:
        return V
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(V), size=max_points, replace=False))
    return V[idx]


def normalize_for_view(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    all_pts = np.vstack(arrays)
    center = all_pts.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(all_pts - center, axis=1).max()
    return tuple((arr - center) / scale for arr in arrays)


def project(P: np.ndarray, axes: str, rotate_180: bool) -> np.ndarray:
    axis_map = {"x": 0, "y": 1, "z": 2}
    P2 = P[:, [axis_map[axes[0]], axis_map[axes[1]]]]
    return -P2 if rotate_180 else P2


def prepare_method(Xs: np.ndarray, Ys: np.ndarray, method: str) -> Tuple[np.ndarray, np.ndarray]:
    if method == "direct":
        X_eval = Xs
    elif method == "bbox":
        X_eval = prealign_by_bbox(Xs, Ys)
    elif method == "rigid":
        X_eval, _ = fb.icp_align(Xs, Ys, prealign="bbox")
    else:
        raise ValueError(f"unknown method: {method}")
    X_deformed = fb.nonrigid_icp_align(X_eval, Ys)
    return X_eval, X_deformed


def draw_overlay(ax, X_eval: np.ndarray, X_deformed: np.ndarray, Y: np.ndarray, axes: str, title: str, rotate_180: bool) -> None:
    Xv, Dv, Yv = normalize_for_view(X_eval, X_deformed, Y)
    X2 = project(Xv, axes, rotate_180)
    D2 = project(Dv, axes, rotate_180)
    Y2 = project(Yv, axes, rotate_180)
    ax.scatter(Y2[:, 0], Y2[:, 1], s=1.0, alpha=0.22, color="#ff7f0e", label="target")
    ax.scatter(X2[:, 0], X2[:, 1], s=1.0, alpha=0.18, color="#1f77b4", label="source before NICP")
    ax.scatter(D2[:, 0], D2[:, 1], s=1.0, alpha=0.18, color="#16a34a", label="source after NICP")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_displacement(ax, X_eval: np.ndarray, X_deformed: np.ndarray, Y: np.ndarray, axes: str, title: str, rotate_180: bool, n_lines: int, seed: int) -> None:
    Xv, Dv, Yv = normalize_for_view(X_eval, X_deformed, Y)
    X2 = project(Xv, axes, rotate_180)
    D2 = project(Dv, axes, rotate_180)
    Y2 = project(Yv, axes, rotate_180)
    disp = np.linalg.norm(X_deformed - X_eval, axis=1)
    rng = np.random.default_rng(seed)
    high = np.argsort(disp)[-max(1, n_lines // 3):]
    rest = np.setdiff1d(np.arange(len(X_eval)), high, assume_unique=False)
    rand = rng.choice(rest, size=min(len(rest), n_lines - len(high)), replace=False)
    line_idx = np.sort(np.concatenate([high, rand]))

    ax.scatter(Y2[:, 0], Y2[:, 1], s=1.0, alpha=0.12, color="#ff7f0e", label="target")
    ax.scatter(X2[:, 0], X2[:, 1], s=1.0, alpha=0.16, color="#1f77b4", label="before")
    ax.scatter(D2[:, 0], D2[:, 1], s=1.0, alpha=0.18, color="#16a34a", label="after")
    threshold = np.percentile(disp, 75)
    for i in line_idx:
        color = "#dc2626" if disp[i] >= threshold else "#111827"
        ax.plot([X2[i, 0], D2[i, 0]], [X2[i, 1], D2[i, 1]], color=color, alpha=0.45, linewidth=0.55)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_case(
    npz_root: Path,
    out_dir: Path,
    subject_a: str,
    subject_b: str,
    topo_a: str,
    topo_b: str,
    max_points: int,
    n_lines: int,
    rotate_180: bool,
) -> None:
    X = load_normalized_vertices_npz(npz_root / f"{subject_a}_GTready_{topo_a}.npz")
    Y = load_normalized_vertices_npz(npz_root / f"{subject_b}_GTready_{topo_b}.npz")
    Xs = sample_points(X, max_points, seed=0)
    Ys = sample_points(Y, max_points, seed=1)

    methods = ["direct", "bbox", "rigid"]
    axes_list = ["xy", "xz"]
    fig, axs = plt.subplots(len(methods), 4, figsize=(15, 11), constrained_layout=True)
    fig.suptitle(f"NICP deformation: {subject_a} {topo_a} -> {subject_b} {topo_b}", fontsize=13)

    rows = []
    for row, method in enumerate(methods):
        X_eval, X_deformed = prepare_method(Xs, Ys, method)
        disp = np.linalg.norm(X_deformed - X_eval, axis=1)
        corr = fb.chamfer_correspondence(X_deformed, Ys)
        fit = np.mean(fb.p2p_distance(X_deformed, Ys, corr))
        orig = np.mean(fb.p2p_distance(X_eval, Ys, corr))
        rows.append((method, float(np.mean(disp)), float(np.median(disp)), float(np.percentile(disp, 90)), float(fit), float(orig)))

        draw_overlay(axs[row, 0], X_eval, X_deformed, Ys, axes_list[0], f"{method} overlay (x,y)", rotate_180)
        draw_displacement(axs[row, 1], X_eval, X_deformed, Ys, axes_list[0], f"{method} displacement (x,y)", rotate_180, n_lines, row)
        draw_overlay(axs[row, 2], X_eval, X_deformed, Ys, axes_list[1], f"{method} overlay (x,z)", rotate_180)
        draw_displacement(axs[row, 3], X_eval, X_deformed, Ys, axes_list[1], f"{method} displacement (x,z)", rotate_180, n_lines, row + 100)

    axs[0, 0].legend(frameon=False, markerscale=5, loc="lower left")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"deformation_{subject_a}_to_{subject_b}_{topo_a}_to_{topo_b}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    summary = ["method,disp_mean,disp_median,disp_p90,fit_p2p,orig_p2p"]
    summary += [f"{m},{dm:.6f},{dmed:.6f},{dp90:.6f},{fit:.6f},{orig:.6f}" for m, dm, dmed, dp90, fit, orig in rows]
    (out_dir / f"deformation_{subject_a}_to_{subject_b}_{topo_a}_to_{topo_b}.csv").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(out_path)
    print("\n".join(summary))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npz_root", default="datasets/REMESH/npz_data_topo_500")
    p.add_argument("--out_dir", default="faceBench/latentVSpipeline/outputs/nicp_deformation_diagnostics")
    p.add_argument("--subject_a", default="id0000")
    p.add_argument("--subject_b", default="id0000")
    p.add_argument("--topo_a", default="crop")
    p.add_argument("--topo_b", default="down8k")
    p.add_argument("--max_points", type=int, default=0)
    p.add_argument("--n_lines", type=int, default=120)
    p.add_argument("--rotate_180", action="store_true")
    args = p.parse_args()
    plot_case(
        npz_root=Path(args.npz_root),
        out_dir=Path(args.out_dir),
        subject_a=args.subject_a,
        subject_b=args.subject_b,
        topo_a=args.topo_a,
        topo_b=args.topo_b,
        max_points=args.max_points,
        n_lines=args.n_lines,
        rotate_180=args.rotate_180,
    )


if __name__ == "__main__":
    main()
