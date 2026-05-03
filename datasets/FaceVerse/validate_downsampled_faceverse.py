#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import downsample_faceverse as dff


DEFAULT_DOWNSAMPLED_ROOT = Path(__file__).resolve().parent / "downsampled"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a downsampled FaceVerse dataset and report loading, "
            "count, and alignment statistics."
        )
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=DEFAULT_DOWNSAMPLED_ROOT,
        help="Root directory containing downsampled .ply files.",
    )
    parser.add_argument(
        "--count_low",
        type=int,
        default=9500,
        help="Lower bound used to flag point-count outliers.",
    )
    parser.add_argument(
        "--count_high",
        type=int,
        default=10500,
        help="Upper bound used to flag point-count outliers.",
    )
    parser.add_argument(
        "--alignment_sigma",
        type=float,
        default=3.0,
        help="Sigma threshold used to flag centroid and span outliers.",
    )
    parser.add_argument(
        "--max_reported",
        type=int,
        default=10,
        help="Maximum number of outlier file names printed per category.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional path for a JSON report.",
    )
    args = parser.parse_args()
    if args.count_low < 0 or args.count_high < args.count_low:
        parser.error("count thresholds must satisfy 0 <= count_low <= count_high")
    if args.alignment_sigma <= 0.0:
        parser.error("--alignment_sigma must be > 0")
    if args.max_reported <= 0:
        parser.error("--max_reported must be > 0")
    return args


def to_serializable(value):
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    if not input_root.exists():
        print(f"Input root does not exist: {input_root}")
        return 1

    paths = sorted(input_root.rglob("*.ply"))
    if not paths:
        print(f"No .ply files found under {input_root}")
        return 1

    records = []
    load_errors = []
    for path in paths:
        relative_path = str(path.relative_to(input_root))
        try:
            header = dff.parse_ply_header(path)
            points = dff.load_vertex_positions(path, header)
        except Exception as exc:
            load_errors.append((relative_path, str(exc)))
            continue

        points = np.asarray(points, dtype=np.float32)
        finite = bool(np.isfinite(points).all())
        centroid = points.mean(axis=0)
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        span = maxs - mins
        bbox_center = (mins + maxs) * 0.5
        rms = float(np.sqrt(np.mean(np.sum((points - centroid) ** 2, axis=1))))
        suffix = path.stem.split("_")[1] if "_" in path.stem else "unknown"
        records.append(
            {
                "relative_path": relative_path,
                "name": path.name,
                "suffix": suffix,
                "count": int(points.shape[0]),
                "finite": finite,
                "format": header.fmt,
                "centroid": centroid,
                "bbox_center": bbox_center,
                "span": span,
                "rms": rms,
            }
        )

    print(f"Input root: {input_root}")
    print(f"Files discovered: {len(paths)}")
    print(f"Load errors: {len(load_errors)}")

    if not records:
        if load_errors:
            print("All files failed to load.")
            for relative_path, error_message in load_errors[: args.max_reported]:
                print(f"- {relative_path}: {error_message}")
        return 1

    counts = np.array([record["count"] for record in records], dtype=np.int32)
    finite_flags = np.array([record["finite"] for record in records], dtype=bool)
    centroids = np.stack([record["centroid"] for record in records], axis=0)
    bbox_centers = np.stack([record["bbox_center"] for record in records], axis=0)
    spans = np.stack([record["span"] for record in records], axis=0)
    rms_values = np.array([record["rms"] for record in records], dtype=np.float64)

    centroid_mean = centroids.mean(axis=0)
    bbox_center_mean = bbox_centers.mean(axis=0)
    span_mean = spans.mean(axis=0)
    centroid_dist = np.linalg.norm(centroids - centroid_mean, axis=1)
    span_dist = np.linalg.norm(spans - span_mean, axis=1)

    centroid_threshold = float(centroid_dist.mean() + args.alignment_sigma * centroid_dist.std())
    span_threshold = float(span_dist.mean() + args.alignment_sigma * span_dist.std())

    centroid_outliers = [
        records[index]["name"]
        for index, distance in enumerate(centroid_dist)
        if distance > centroid_threshold
    ]
    span_outliers = [
        records[index]["name"]
        for index, distance in enumerate(span_dist)
        if distance > span_threshold
    ]
    count_outliers = [
        records[index]["name"]
        for index, count in enumerate(counts)
        if count < args.count_low or count > args.count_high
    ]

    centroid_suffix_counts = Counter(name.split("_")[1].split(".")[0] for name in centroid_outliers)
    span_suffix_counts = Counter(name.split("_")[1].split(".")[0] for name in span_outliers)

    summary = {
        "input_root": str(input_root),
        "files_discovered": len(paths),
        "files_loaded": len(records),
        "load_errors": load_errors,
        "formats": sorted({record["format"] for record in records}),
        "all_finite": bool(finite_flags.all()),
        "non_finite_count": int((~finite_flags).sum()),
        "count_stats": {
            "min": int(counts.min()),
            "max": int(counts.max()),
            "mean": float(counts.mean()),
            "std": float(counts.std()),
            "p01": float(np.percentile(counts, 1)),
            "p99": float(np.percentile(counts, 99)),
        },
        "centroid_stats": {
            "mean": centroids.mean(axis=0),
            "std": centroids.std(axis=0),
            "max_distance": float(centroid_dist.max()),
            "max_distance_file": records[int(centroid_dist.argmax())]["name"],
            "outlier_count": len(centroid_outliers),
            "outlier_threshold": centroid_threshold,
            "outlier_suffix_counts": dict(centroid_suffix_counts.most_common()),
            "sample_outliers": centroid_outliers[: args.max_reported],
        },
        "bbox_center_stats": {
            "mean": bbox_centers.mean(axis=0),
            "std": bbox_centers.std(axis=0),
            "max_distance": float(
                np.linalg.norm(bbox_centers - bbox_center_mean, axis=1).max()
            ),
            "max_distance_file": records[
                int(np.linalg.norm(bbox_centers - bbox_center_mean, axis=1).argmax())
            ]["name"],
        },
        "span_stats": {
            "mean": spans.mean(axis=0),
            "std": spans.std(axis=0),
            "max_distance": float(span_dist.max()),
            "max_distance_file": records[int(span_dist.argmax())]["name"],
            "outlier_count": len(span_outliers),
            "outlier_threshold": span_threshold,
            "outlier_suffix_counts": dict(span_suffix_counts.most_common()),
            "sample_outliers": span_outliers[: args.max_reported],
        },
        "rms_stats": {
            "mean": float(rms_values.mean()),
            "std": float(rms_values.std()),
        },
        "count_outliers": {
            "low": args.count_low,
            "high": args.count_high,
            "count": len(count_outliers),
            "sample_outliers": count_outliers[: args.max_reported],
        },
    }

    print(f"Formats: {summary['formats']}")
    print(f"All finite: {summary['all_finite']}")
    print(
        "Point-count stats: "
        f"min={summary['count_stats']['min']} "
        f"max={summary['count_stats']['max']} "
        f"mean={summary['count_stats']['mean']:.2f} "
        f"std={summary['count_stats']['std']:.2f}"
    )
    print(
        "Centroid stats: "
        f"mean={summary['centroid_stats']['mean'].tolist()} "
        f"std={summary['centroid_stats']['std'].tolist()}"
    )
    print(
        "Span stats: "
        f"mean={summary['span_stats']['mean'].tolist()} "
        f"std={summary['span_stats']['std'].tolist()}"
    )
    print(
        "RMS stats: "
        f"mean={summary['rms_stats']['mean']:.6f} "
        f"std={summary['rms_stats']['std']:.6f}"
    )
    print(
        "Centroid outliers: "
        f"{summary['centroid_stats']['outlier_count']} "
        f"(suffix counts: {summary['centroid_stats']['outlier_suffix_counts']})"
    )
    print(
        "Span outliers: "
        f"{summary['span_stats']['outlier_count']} "
        f"(suffix counts: {summary['span_stats']['outlier_suffix_counts']})"
    )
    print(
        "Count outliers: "
        f"{summary['count_outliers']['count']} outside "
        f"[{args.count_low}, {args.count_high}]"
    )

    if load_errors:
        print("\nLoad error samples")
        for relative_path, error_message in load_errors[: args.max_reported]:
            print(f"- {relative_path}: {error_message}")

    if summary["centroid_stats"]["sample_outliers"]:
        print("\nCentroid outlier samples")
        for name in summary["centroid_stats"]["sample_outliers"]:
            print(f"- {name}")

    if summary["span_stats"]["sample_outliers"]:
        print("\nSpan outlier samples")
        for name in summary["span_stats"]["sample_outliers"]:
            print(f"- {name}")

    if summary["count_outliers"]["sample_outliers"]:
        print("\nCount outlier samples")
        for name in summary["count_outliers"]["sample_outliers"]:
            print(f"- {name}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(to_serializable(summary), indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved JSON report to {args.output_json.resolve()}")

    return 1 if load_errors or not summary["all_finite"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
