#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect existing ranking_summary.csv files into one comparison table.")
    p.add_argument("--roots", nargs="+", required=True, help="Directories to scan recursively")
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def infer_method(path: Path) -> str:
    text = str(path)
    if "ranking_vs_chamfer" in text:
        return "raw_chamfer"
    if "registered_chamfer" in text and "cpd64" in text:
        return "rigid_cpd_chamfer"
    if "registered_chamfer" in text and "rigidonly" in text:
        return "rigid_chamfer"
    if "nicp_correspondence" in text:
        return "nicp_correspondence"
    if "raw_noicp" in text:
        return "raw_chamfer"
    if "rigid_only" in text:
        return "rigid_chamfer"
    return "unknown"


def metric_spearman_column(row: Dict[str, str]) -> str:
    candidates = [
        key
        for key in row
        if key.endswith("_spearman") and key not in {"latent_spearman", "delta_spearman"}
    ]
    return candidates[0] if candidates else ""


def read_summary(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    method = infer_method(path)
    out = []
    for row in rows:
        metric_col = metric_spearman_column(row)
        new_row = dict(row)
        new_row["method"] = method
        new_row["metric_spearman_column"] = metric_col
        new_row["metric_spearman"] = row.get(metric_col, "")
        new_row["source_csv"] = str(path)
        out.append(new_row)
    return out


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    keys: List[str] = []
    seen = set()
    preferred = [
        "method",
        "scenario",
        "topology_a",
        "topology_b",
        "ordered_pair_label",
        "latent_spearman",
        "metric_spearman",
        "delta_spearman",
        "metric_spearman_column",
        "n_observations",
        "source_csv",
    ]
    for key in preferred:
        keys.append(key)
        seen.add(key)
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


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, str]] = []
    for root_text in args.roots:
        root = Path(root_text)
        for path in sorted(root.rglob("ranking_summary.csv")):
            rows.extend(read_summary(path))
    out_csv = out_dir / "existing_ranking_summaries.csv"
    write_csv(out_csv, rows)
    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
