#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate collected ranking summaries by method/scenario.")
    p.add_argument("--summary_csv", required=True)
    p.add_argument("--out_dir", required=True)
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


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def finite(values: Iterable[float]) -> List[float]:
    return [v for v in values if math.isfinite(v)]


def mean(values: Iterable[float]) -> float:
    vals = finite(values)
    return sum(vals) / len(vals) if vals else math.nan


def median(values: Iterable[float]) -> float:
    vals = sorted(finite(values))
    if not vals:
        return math.nan
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def summarize(rows: Sequence[Dict[str, str]], group_cols: Sequence[str]) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[str, ...], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(col, "")) for col in group_cols)
        buckets[key].append(row)

    out: List[Dict[str, object]] = []
    for key, subrows in sorted(buckets.items()):
        metric_vals = [as_float(row.get("metric_spearman")) for row in subrows]
        latent_vals = [as_float(row.get("latent_spearman")) for row in subrows]
        delta_vals = [as_float(row.get("delta_spearman")) for row in subrows]
        row_out: Dict[str, object] = {col: value for col, value in zip(group_cols, key)}
        row_out.update(
            {
                "n_rows": len(subrows),
                "metric_spearman_mean": mean(metric_vals),
                "metric_spearman_median": median(metric_vals),
                "latent_spearman_mean": mean(latent_vals),
                "delta_spearman_mean": mean(delta_vals),
                "model_beats_metric_rate": mean(
                    [
                        1.0 if as_float(row.get("latent_spearman")) > as_float(row.get("metric_spearman")) else 0.0
                        for row in subrows
                        if math.isfinite(as_float(row.get("latent_spearman")))
                        and math.isfinite(as_float(row.get("metric_spearman")))
                    ]
                ),
            }
        )
        out.append(row_out)
    return out


def main() -> None:
    args = parse_args()
    rows = read_csv(Path(args.summary_csv))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_rows = summarize(rows, ["method", "scenario"])
    topology_rows = summarize(rows, ["method", "scenario", "ordered_pair_label"])
    method_rows = summarize(rows, ["method"])

    write_csv(out_dir / "method_summary.csv", method_rows)
    write_csv(out_dir / "scenario_method_summary.csv", scenario_rows)
    write_csv(out_dir / "topology_pair_method_summary.csv", topology_rows)
    print(f"Wrote summaries to {out_dir}")


if __name__ == "__main__":
    main()
