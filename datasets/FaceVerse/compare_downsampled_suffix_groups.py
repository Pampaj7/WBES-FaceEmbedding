#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
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
            "Create a side-by-side visual comparison between two suffix groups "
            "such as _01 and _02."
        )
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=DEFAULT_DOWNSAMPLED_ROOT,
        help="Root directory containing downsampled .ply files.",
    )
    parser.add_argument(
        "--suffix_a",
        default="01",
        help="First suffix group to compare.",
    )
    parser.add_argument(
        "--suffix_b",
        default="02",
        help="Second suffix group to compare.",
    )
    parser.add_argument(
        "--subject_ids",
        nargs="*",
        default=None,
        help="Optional explicit list of subject ids such as 001 002 003.",
    )
    parser.add_argument(
        "--num_subjects",
        type=int,
        default=6,
        help="Number of subjects shown when subject_ids are not provided.",
    )
    parser.add_argument(
        "--display_mode",
        choices=("raw", "center_bbox", "center_centroid"),
        default="raw",
        help="Optional display-only centering used before plotting.",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=1.0,
        help="Scatter marker size.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output PNG resolution.",
    )
    parser.add_argument(
        "--output_png",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to compare_<suffix_a>_vs_<suffix_b>.png inside input_root.",
    )
    args = parser.parse_args()
    if args.num_subjects <= 0:
        parser.error("--num_subjects must be > 0")
    return args


def normalize_for_display(points: np.ndarray, display_mode: str) -> np.ndarray:
    if display_mode == "raw":
        return np.asarray(points, dtype=np.float32)
    if display_mode == "center_bbox":
        return dff.normalize_points(points, "center_bbox")
    if display_mode == "center_centroid":
        return dff.normalize_points(points, "center_centroid")
    raise ValueError(f"Unsupported display mode: {display_mode}")


def set_axes_equal(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.5)
    if radius == 0.0:
        radius = 1.0

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    if not input_root.exists():
        print(f"Input root does not exist: {input_root}")
        return 1

    output_png = (
        args.output_png.resolve()
        if args.output_png is not None
        else (input_root / f"compare_{args.suffix_a}_vs_{args.suffix_b}.png")
    )

    by_subject: dict[str, dict[str, Path]] = defaultdict(dict)
    for path in sorted(input_root.rglob("*.ply")):
        stem_parts = path.stem.split("_")
        if len(stem_parts) != 2:
            continue
        subject_id, suffix = stem_parts
        by_subject[subject_id][suffix] = path

    if args.subject_ids:
        subject_ids = args.subject_ids
    else:
        subject_ids = sorted(
            subject_id
            for subject_id, suffix_map in by_subject.items()
            if args.suffix_a in suffix_map and args.suffix_b in suffix_map
        )[: args.num_subjects]

    pairs = []
    missing_subjects = []
    for subject_id in subject_ids:
        suffix_map = by_subject.get(subject_id, {})
        if args.suffix_a not in suffix_map or args.suffix_b not in suffix_map:
            missing_subjects.append(subject_id)
            continue
        pairs.append((subject_id, suffix_map[args.suffix_a], suffix_map[args.suffix_b]))

    if missing_subjects:
        print(f"Missing one of the requested suffixes for subjects: {', '.join(missing_subjects)}")
    if not pairs:
        print("No valid subject pairs were found.")
        return 1

    fig = plt.figure(figsize=(10, 4 * len(pairs)), dpi=args.dpi)
    for row_index, (subject_id, path_a, path_b) in enumerate(pairs, start=1):
        header_a = dff.parse_ply_header(path_a)
        header_b = dff.parse_ply_header(path_b)
        points_a = normalize_for_display(dff.load_vertex_positions(path_a, header_a), args.display_mode)
        points_b = normalize_for_display(dff.load_vertex_positions(path_b, header_b), args.display_mode)
        combined = np.vstack((points_a, points_b))

        ax_a = fig.add_subplot(len(pairs), 2, (row_index - 1) * 2 + 1, projection="3d")
        ax_b = fig.add_subplot(len(pairs), 2, (row_index - 1) * 2 + 2, projection="3d")

        for ax, points, suffix in (
            (ax_a, points_a, args.suffix_a),
            (ax_b, points_b, args.suffix_b),
        ):
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                s=args.point_size,
                c="black",
                linewidths=0.0,
                depthshade=False,
            )
            set_axes_equal(ax, combined)
            ax.view_init(elev=15, azim=35)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_title(f"{subject_id}_{suffix}", fontsize=10)

    fig.suptitle(
        f"Suffix comparison {args.suffix_a} vs {args.suffix_b} ({args.display_mode})",
        fontsize=12,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"Input root: {input_root}")
    print(f"Subjects compared: {len(pairs)}")
    print(f"Display mode: {args.display_mode}")
    print(f"Saved comparison PNG to {output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
