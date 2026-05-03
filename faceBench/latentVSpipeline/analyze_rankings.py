#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


DEFAULT_METRICS = [
    "latent_distance",
    "raw_chamfer",
    "rigid_chamfer",
    "fg_p2p",
    "fg_p2tri",
    "fg_corrected_p2p",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze latent-vs-FG ranking agreement.")
    p.add_argument("--pair_metrics", required=True, help="CSV produced by run_pair_table.py or compatible")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--reference", default="latent_distance", help="Ranking reference column")
    p.add_argument("--gt_col", default="gt_distance", help="Optional GT distance column")
    p.add_argument("--metrics", default=",".join(DEFAULT_METRICS), help="Metric columns to analyze")
    p.add_argument("--group_cols", default="scenario", help="Comma-separated grouping columns; empty for overall only")
    p.add_argument("--topk", default="10,50,100", help="Comma-separated top-k values")
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_float_array(rows: Sequence[Dict[str, str]], col: str) -> np.ndarray:
    vals = []
    for row in rows:
        try:
            vals.append(float(row.get(col, "")))
        except (TypeError, ValueError):
            vals.append(math.nan)
    return np.asarray(vals, dtype=np.float64)


def ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty_like(order, dtype=np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        if j - i > 1:
            r[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return r


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return math.nan
    xx = x[mask]
    yy = y[mask]
    if float(np.std(xx)) == 0.0 or float(np.std(yy)) == 0.0:
        return math.nan
    return float(np.corrcoef(xx, yy)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return math.nan
    return pearson(ranks(x[mask]), ranks(y[mask]))


def kendall_tau_sampled(x: np.ndarray, y: np.ndarray, max_pairs: int = 250_000, seed: int = 1234) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 2:
        return math.nan
    total = n * (n - 1) // 2
    rng = np.random.default_rng(seed)
    if total <= max_pairs:
        ii, jj = np.triu_indices(n, k=1)
    else:
        ii = rng.integers(0, n, size=max_pairs)
        jj = rng.integers(0, n, size=max_pairs)
        keep = ii != jj
        ii, jj = ii[keep], jj[keep]
    sx = np.sign(x[ii] - x[jj])
    sy = np.sign(y[ii] - y[jj])
    valid = (sx != 0) & (sy != 0)
    if int(valid.sum()) == 0:
        return math.nan
    return float(np.mean(sx[valid] * sy[valid]))


def topk_overlap(reference: np.ndarray, values: np.ndarray, k: int) -> float:
    mask = np.isfinite(reference) & np.isfinite(values)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return math.nan
    k = min(k, len(idx))
    ref_top = set(idx[np.argsort(reference[idx])[:k]].tolist())
    val_top = set(idx[np.argsort(values[idx])[:k]].tolist())
    return float(len(ref_top & val_top) / k)


def auc_lower_is_positive(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    wins = 0.0
    total = 0
    for value in pos:
        wins += float(np.sum(value < neg))
        wins += 0.5 * float(np.sum(value == neg))
        total += len(neg)
    return float(wins / total) if total else math.nan


def summarize_subset(
    rows: Sequence[Dict[str, str]],
    *,
    label: Dict[str, object],
    reference_col: str,
    gt_col: str,
    metrics: Sequence[str],
    topks: Sequence[int],
) -> List[Dict[str, object]]:
    ref = to_float_array(rows, reference_col)
    gt = to_float_array(rows, gt_col) if gt_col else np.full((len(rows),), math.nan)
    pair_types = [str(row.get("pair_type", "")) for row in rows]
    out: List[Dict[str, object]] = []
    for metric in metrics:
        values = to_float_array(rows, metric)
        if not np.isfinite(values).any():
            continue
        row: Dict[str, object] = {
            **label,
            "metric": metric,
            "n": int(np.sum(np.isfinite(values) & np.isfinite(ref))),
            "spearman_vs_latent": spearman(ref, values),
            "pearson_vs_latent": pearson(ref, values),
            "kendall_vs_latent_sampled": kendall_tau_sampled(ref, values),
        }
        if np.isfinite(gt).any():
            row["spearman_vs_gt"] = spearman(gt, values)
            row["pearson_vs_gt"] = pearson(gt, values)
        same_mask = np.asarray([pt == "same_subject" for pt in pair_types], dtype=bool)
        diff_mask = np.asarray([pt in {"different_subject", "diff_subject"} for pt in pair_types], dtype=bool)
        row["same_mean"] = float(np.nanmean(values[same_mask])) if same_mask.any() else math.nan
        row["diff_mean"] = float(np.nanmean(values[diff_mask])) if diff_mask.any() else math.nan
        row["gap_diff_minus_same"] = row["diff_mean"] - row["same_mean"] if math.isfinite(row["same_mean"]) and math.isfinite(row["diff_mean"]) else math.nan
        row["same_vs_diff_auc"] = auc_lower_is_positive(values[same_mask], values[diff_mask]) if same_mask.any() and diff_mask.any() else math.nan
        for k in topks:
            row[f"top{k}_overlap_vs_latent"] = topk_overlap(ref, values, k)
        out.append(row)
    return out


def group_rows(rows: Sequence[Dict[str, str]], group_cols: Sequence[str]) -> Iterable[tuple[Dict[str, object], List[Dict[str, str]]]]:
    yield {"group": "overall"}, list(rows)
    if not group_cols:
        return
    buckets: Dict[tuple[str, ...], List[Dict[str, str]]] = {}
    for row in rows:
        key = tuple(str(row.get(col, "")) for col in group_cols)
        buckets.setdefault(key, []).append(row)
    for key, subrows in sorted(buckets.items()):
        label = {"group": "/".join(key)}
        for col, value in zip(group_cols, key):
            label[col] = value
        yield label, subrows


def main() -> None:
    args = parse_args()
    rows = read_csv(Path(args.pair_metrics))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
    group_cols = [c.strip() for c in str(args.group_cols).split(",") if c.strip()]
    topks = [int(k) for k in str(args.topk).split(",") if k.strip()]

    summary_rows: List[Dict[str, object]] = []
    for label, subrows in group_rows(rows, group_cols):
        summary_rows.extend(
            summarize_subset(
                subrows,
                label=label,
                reference_col=args.reference,
                gt_col=args.gt_col,
                metrics=metrics,
                topks=topks,
            )
        )

    write_csv(out_dir / "ranking_agreement.csv", summary_rows)
    with open(out_dir / "ranking_agreement.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, allow_nan=True)
        f.write("\n")
    print(f"Wrote {out_dir / 'ranking_agreement.csv'}")


if __name__ == "__main__":
    main()
