#!/usr/bin/env python3
"""Visual diagnostics for NICP-derived correspondences.

The plots compare same-subject and different-subject pairs under three
initializations:

- direct: NICP starts from sampled source points.
- bbox: source is center/scale aligned by bounding box before NICP.
- rigid: source is bbox prealigned and then rigid ICP aligned before NICP.

For each method, correspondences are estimated on the NICP-deformed source, but
drawn on the non-deformed evaluation source. This mirrors the ablation where NICP
is used only to estimate correspondences.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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


def sample_with_indices(V: np.ndarray, max_pts: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if max_pts <= 0 or len(V) <= max_pts:
        idx = np.arange(len(V))
        return V, idx
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(V), size=max_pts, replace=False))
    return V[idx], idx


def normalize_for_view(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    all_pts = np.vstack([X, Y])
    center = all_pts.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(all_pts - center, axis=1).max()
    return (X - center) / scale, (Y - center) / scale


def project(P: np.ndarray, axes: str, rotate_180: bool = False) -> np.ndarray:
    axis_map = {"x": 0, "y": 1, "z": 2}
    P2 = P[:, [axis_map[axes[0]], axis_map[axes[1]]]]
    return -P2 if rotate_180 else P2


def method_correspondence(Xs: np.ndarray, Ys: np.ndarray, method: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return evaluation source, target, and pidx.

    pidx is estimated from the NICP-deformed source to target, but the returned
    source is the pre-NICP coordinate system where the distance is measured.
    """
    if method == "direct":
        X_eval = Xs
    elif method == "bbox":
        X_eval = prealign_by_bbox(Xs, Ys)
    elif method == "rigid":
        X_eval, _ = fb.icp_align(Xs, Ys, prealign="bbox")
    else:
        raise ValueError(f"unknown method: {method}")

    X_nicp = fb.nonrigid_icp_align(X_eval, Ys)
    pidx = fb.chamfer_correspondence(X_nicp, Ys)
    return X_eval, Ys, pidx


def choose_line_indices(X_eval: np.ndarray, Y: np.ndarray, pidx: np.ndarray, n_lines: int, seed: int) -> np.ndarray:
    """Choose a spatially broad but deterministic set of correspondence lines."""
    rng = np.random.default_rng(seed)
    if len(X_eval) <= n_lines:
        return np.arange(len(X_eval))

    # Mix high-error points with random points so both systematic drift and
    # typical correspondences are visible.
    d = np.linalg.norm(X_eval - Y[pidx], axis=1)
    high = np.argsort(d)[-max(1, n_lines // 3):]
    rest_pool = np.setdiff1d(np.arange(len(X_eval)), high, assume_unique=False)
    rand = rng.choice(rest_pool, size=n_lines - len(high), replace=False)
    return np.sort(np.concatenate([high, rand]))


def draw_panel(
    ax,
    X_eval: np.ndarray,
    Y: np.ndarray,
    pidx: np.ndarray,
    title: str,
    axes: str,
    n_lines: int,
    seed: int,
    rotate_180: bool,
) -> Dict[str, float]:
    Xv, Yv = normalize_for_view(X_eval, Y)
    X2 = project(Xv, axes, rotate_180=rotate_180)
    Y2 = project(Yv, axes, rotate_180=rotate_180)
    line_idx = choose_line_indices(X_eval, Y, pidx, n_lines=n_lines, seed=seed)

    dist = np.linalg.norm(X_eval - Y[pidx], axis=1)
    ax.scatter(Y2[:, 0], Y2[:, 1], s=1.5, c="#9ca3af", alpha=0.22, linewidths=0)
    ax.scatter(X2[:, 0], X2[:, 1], s=2.0, c="#2563eb", alpha=0.28, linewidths=0)
    for i in line_idx:
        j = pidx[i]
        color = "#dc2626" if dist[i] >= np.percentile(dist, 75) else "#111827"
        ax.plot([X2[i, 0], Y2[j, 0]], [X2[i, 1], Y2[j, 1]], color=color, alpha=0.42, linewidth=0.55)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return {
        "mean": float(np.mean(dist)),
        "median": float(np.median(dist)),
        "p90": float(np.percentile(dist, 90)),
    }


def plot_pair(
    npz_root: Path,
    out_dir: Path,
    subject_a: str,
    subject_b: str,
    topo_a: str,
    topo_b: str,
    max_points: int,
    n_lines: int,
    seed: int,
    rotate_180: bool,
) -> Dict[str, Dict[str, float]]:
    X = load_normalized_vertices_npz(npz_root / f"{subject_a}_GTready_{topo_a}.npz")
    Y = load_normalized_vertices_npz(npz_root / f"{subject_b}_GTready_{topo_b}.npz")
    Xs, _ = sample_with_indices(X, max_points, seed)
    Ys, _ = sample_with_indices(Y, max_points, seed + 1)

    methods = ["direct", "bbox", "rigid"]
    axes_list = ["xy", "xz"]
    fig, axs = plt.subplots(len(axes_list), len(methods), figsize=(12, 7), constrained_layout=True)
    stats: Dict[str, Dict[str, float]] = {}

    kind = "same" if subject_a == subject_b else "diff"
    fig.suptitle(f"{kind}: {subject_a} {topo_a} -> {subject_b} {topo_b}", fontsize=12)
    for col, method in enumerate(methods):
        X_eval, Y_eval, pidx = method_correspondence(Xs, Ys, method)
        method_stats = {}
        for row, axes in enumerate(axes_list):
            panel_stats = draw_panel(
                axs[row, col],
                X_eval,
                Y_eval,
                pidx,
                f"{method} ({axes})",
                axes=axes,
                n_lines=n_lines,
                seed=seed + row * 100 + col,
                rotate_180=rotate_180,
            )
            method_stats.update({f"{axes}_{k}": v for k, v in panel_stats.items()})
        stats[method] = method_stats

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{kind}_{subject_a}_to_{subject_b}_{topo_a}_to_{topo_b}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npz_root", default="datasets/REMESH/npz_data_topo_500")
    p.add_argument("--out_dir", default="faceBench/latentVSpipeline/outputs/nicp_correspondence_diagnostics")
    p.add_argument("--topo_a", default="crop")
    p.add_argument("--topo_b", default="down8k")
    p.add_argument("--max_points", type=int, default=1024)
    p.add_argument("--n_lines", type=int, default=90)
    p.add_argument("--rotate_180", action="store_true", help="Rotate each 2D projected panel by 180 degrees")
    args = p.parse_args()

    npz_root = Path(args.npz_root)
    out_dir = Path(args.out_dir)
    pairs = [
        ("id0000", "id0000"),
        ("id0001", "id0001"),
        ("id0000", "id0009"),
        ("id0001", "id0008"),
        ("id0003", "id0006"),
    ]

    summary_rows: List[str] = ["kind,pair,method,mean,median,p90"]
    for seed, (sa, sb) in enumerate(pairs):
        stats = plot_pair(
            npz_root=npz_root,
            out_dir=out_dir,
            subject_a=sa,
            subject_b=sb,
            topo_a=args.topo_a,
            topo_b=args.topo_b,
            max_points=args.max_points,
            n_lines=args.n_lines,
            seed=seed,
            rotate_180=args.rotate_180,
        )
        kind = "same" if sa == sb else "diff"
        for method, vals in stats.items():
            summary_rows.append(
                f"{kind},{sa}->{sb},{method},{vals['xy_mean']:.6f},{vals['xy_median']:.6f},{vals['xy_p90']:.6f}"
            )

    summary_path = out_dir / "summary.csv"
    summary_path.write_text("\n".join(summary_rows) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir}")
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
