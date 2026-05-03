#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import downsample_faceverse as dff


DEFAULT_DOWNSAMPLED_ROOT = Path(__file__).resolve().parent / "downsampled"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render all suffix variants for one subject as a front-view grid "
            "to inspect whether the suffix index behaves like pose or condition."
        )
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=DEFAULT_DOWNSAMPLED_ROOT,
        help="Root directory containing downsampled .ply files.",
    )
    parser.add_argument(
        "--subject_id",
        default="110",
        help="Subject id to visualize, for example 110.",
    )
    parser.add_argument(
        "--display_mode",
        choices=("raw", "center_bbox", "center_centroid"),
        default="center_bbox",
        help="Display-only centering applied before plotting.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=7,
        help="Number of columns in the grid.",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=0.35,
        help="Scatter marker size.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output PNG resolution.",
    )
    parser.add_argument(
        "--output_png",
        type=Path,
        default=None,
        help="Output PNG path. Defaults inside input_root.",
    )
    args = parser.parse_args()
    if args.cols <= 0:
        parser.error("--cols must be > 0")
    return args


def normalize_for_display(points: np.ndarray, display_mode: str) -> np.ndarray:
    if display_mode == "raw":
        return np.asarray(points, dtype=np.float32)
    if display_mode == "center_bbox":
        return dff.normalize_points(points, "center_bbox")
    if display_mode == "center_centroid":
        return dff.normalize_points(points, "center_centroid")
    raise ValueError(f"Unsupported display mode: {display_mode}")


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    if not input_root.exists():
        print(f"Input root does not exist: {input_root}")
        return 1

    subject_prefix = f"{args.subject_id}_"
    paths = sorted(input_root.glob(f"{subject_prefix}*.ply"))
    if not paths:
        print(f"No files found for subject {args.subject_id} under {input_root}")
        return 1

    output_png = (
        args.output_png.resolve()
        if args.output_png is not None
        else (
            input_root
            / f"subject_{args.subject_id}_suffix_grid_{args.display_mode}.png"
        )
    )

    records = []
    for path in paths:
        header = dff.parse_ply_header(path)
        points = dff.load_vertex_positions(path, header)
        points = normalize_for_display(points, args.display_mode)
        suffix = path.stem.split("_")[1]
        records.append((suffix, points))

    all_x = np.concatenate([points[:, 0] for _, points in records])
    all_y = np.concatenate([points[:, 1] for _, points in records])
    pad_x = 0.02 * float(all_x.max() - all_x.min())
    pad_y = 0.02 * float(all_y.max() - all_y.min())

    cols = args.cols
    rows = math.ceil(len(records) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.8 * rows), dpi=args.dpi)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for axis in axes.ravel():
        axis.set_visible(False)

    for index, (suffix, points) in enumerate(records):
        row = index // cols
        col = index % cols
        ax = axes[row, col]
        ax.set_visible(True)
        ax.set_facecolor("white")
        ax.scatter(points[:, 0], points[:, 1], s=args.point_size, c="black", linewidths=0)
        ax.set_xlim(float(all_x.min() - pad_x), float(all_x.max() + pad_x))
        ax.set_ylim(float(all_y.min() - pad_y), float(all_y.max() + pad_y))
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{args.subject_id}_{suffix}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Subject {args.subject_id} suffix grid ({args.display_mode})",
        fontsize=12,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"Input root: {input_root}")
    print(f"Subject: {args.subject_id}")
    print(f"Files plotted: {len(records)}")
    print(f"Display mode: {args.display_mode}")
    print(f"Saved grid PNG to {output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
