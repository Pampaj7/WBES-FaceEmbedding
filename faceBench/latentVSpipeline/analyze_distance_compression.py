#!/usr/bin/env python3
"""Analyze how registration changes cross-subject distance distributions."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


METRICS = ["latent_distance", "raw_chamfer", "rigid_p2p", "nicp_p2p", "nicp_p2tri"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="CSV files or glob patterns")
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def expand_inputs(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pat in patterns:
        matches = sorted(Path().glob(pat) if not pat.startswith("/") else Path("/").glob(pat[1:]))
        if matches:
            files.extend(m for m in matches if m.is_file())
        else:
            p = Path(pat)
            if p.is_file():
                files.append(p)
    return sorted(set(files))


def scenario_from_path(path: Path) -> tuple[str, str, str]:
    scenario = path.parent.name
    pair = path.parent.parent.name
    if scenario == "clean":
        return "clean", "none", "0.0"
    if "_sigma" in scenario:
        mode, sigma = scenario.split("_sigma", 1)
        return scenario, mode, sigma
    return scenario, scenario, ""


def quantile(xs: list[float], q: float) -> float:
    if not xs:
        return math.nan
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def summarize(xs: list[float]) -> dict[str, float]:
    xs = sorted(x for x in xs if math.isfinite(x))
    n = len(xs)
    if not n:
        return {k: math.nan for k in ["mean", "std", "cv", "min", "p05", "p25", "p50", "p75", "p95", "max", "iqr", "range"]}
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    std = math.sqrt(var)
    p25 = quantile(xs, 0.25)
    p75 = quantile(xs, 0.75)
    return {
        "mean": mean,
        "std": std,
        "cv": std / mean if mean else math.nan,
        "min": xs[0],
        "p05": quantile(xs, 0.05),
        "p25": p25,
        "p50": quantile(xs, 0.50),
        "p75": p75,
        "p95": quantile(xs, 0.95),
        "max": xs[-1],
        "iqr": p75 - p25,
        "range": xs[-1] - xs[0],
    }


def fnum(x: float) -> str:
    return "" if not math.isfinite(x) else f"{x:.10g}"


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    args = parse_args()
    files = expand_inputs(args.inputs)
    out_dir = Path(args.out_dir)
    grouped: dict[tuple[str, str, str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for path in files:
        scenario, mode, sigma = scenario_from_path(path)
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("status", "ok") != "ok":
                    continue
                ta = row.get("topology_a", "")
                tb = row.get("topology_b", "")
                pair = row.get("pair_label") or f"{ta}__to__{tb}"
                for metric in METRICS:
                    try:
                        val = float(row.get(metric, "nan"))
                    except ValueError:
                        val = math.nan
                    if math.isfinite(val):
                        grouped[(scenario, mode, sigma, "overall", "all")][metric].append(val)
                        grouped[(scenario, mode, sigma, "topology_pair", pair)][metric].append(val)

    rows = []
    for key, metric_values in sorted(grouped.items()):
        scenario, mode, sigma, group_type, group = key
        raw_summary = summarize(metric_values.get("raw_chamfer", []))
        for metric in METRICS:
            s = summarize(metric_values.get(metric, []))
            rows.append({
                "scenario": scenario,
                "noise_mode": mode,
                "sigma": sigma,
                "group_type": group_type,
                "group": group,
                "metric": metric,
                "n": len(metric_values.get(metric, [])),
                **{k: fnum(v) for k, v in s.items()},
                "std_vs_raw": fnum(s["std"] / raw_summary["std"]) if raw_summary["std"] else "",
                "iqr_vs_raw": fnum(s["iqr"] / raw_summary["iqr"]) if raw_summary["iqr"] else "",
                "range_vs_raw": fnum(s["range"] / raw_summary["range"]) if raw_summary["range"] else "",
                "median_vs_raw": fnum(s["p50"] / raw_summary["p50"]) if raw_summary["p50"] else "",
            })

    fields = [
        "scenario", "noise_mode", "sigma", "group_type", "group", "metric", "n",
        "mean", "std", "cv", "min", "p05", "p25", "p50", "p75", "p95", "max",
        "iqr", "range", "std_vs_raw", "iqr_vs_raw", "range_vs_raw", "median_vs_raw",
    ]
    write_csv(out_dir / "distance_distribution_summary.csv", rows, fields)

    selected = [r for r in rows if r["group_type"] == "overall"]
    write_csv(out_dir / "distance_distribution_overall.csv", selected, fields)

    print(f"read_files={len(files)}")
    print(f"wrote={out_dir / 'distance_distribution_summary.csv'}")
    print(f"wrote={out_dir / 'distance_distribution_overall.csv'}")


if __name__ == "__main__":
    main()
