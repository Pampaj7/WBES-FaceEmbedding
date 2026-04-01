#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two FaceVerse evaluation result directories "
            "(typically an older checkpoint vs a new checkpoint)."
        )
    )
    parser.add_argument("--old_ranking_csv", type=Path, required=True)
    parser.add_argument("--new_ranking_csv", type=Path, required=True)
    parser.add_argument("--old_sigma_csv", type=Path, required=True)
    parser.add_argument("--new_sigma_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--old_label", type=str, default="old_model")
    parser.add_argument("--new_label", type=str, default="new_model")
    return parser.parse_args()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _to_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _to_int(value: object) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _fmt(value: object, digits: int = 6) -> str:
    value_f = _to_float(value)
    if not math.isfinite(value_f):
        return ""
    return f"{value_f:.{digits}f}"


def _delta(new_value: object, old_value: object) -> float:
    new_f = _to_float(new_value)
    old_f = _to_float(old_value)
    if not (math.isfinite(new_f) and math.isfinite(old_f)):
        return float("nan")
    return new_f - old_f


def _load_map(rows: Sequence[Dict[str, str]], key_fields: Sequence[str]) -> Dict[Tuple[str, ...], Dict[str, str]]:
    result: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in key_fields)
        result[key] = row
    return result


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)
        handle.write("\n")


def _build_ranking_rows(
    *,
    old_rows: Sequence[Dict[str, str]],
    new_rows: Sequence[Dict[str, str]],
    old_label: str,
    new_label: str,
) -> List[Dict[str, object]]:
    key_fields = ("scenario",)
    metrics = (
        "latent_spearman",
        "chamfer_spearman",
        "delta_spearman",
        "latent_pearson",
        "chamfer_pearson",
        "delta_pearson",
    )
    old_map = _load_map(old_rows, key_fields)
    new_map = _load_map(new_rows, key_fields)
    all_keys = sorted(set(old_map) | set(new_map))
    out_rows: List[Dict[str, object]] = []
    for key in all_keys:
        old_row = old_map.get(key, {})
        new_row = new_map.get(key, {})
        scenario = key[0]
        out: Dict[str, object] = {"scenario": scenario}
        for metric in metrics:
            out[f"{old_label}_{metric}"] = _fmt(old_row.get(metric))
            out[f"{new_label}_{metric}"] = _fmt(new_row.get(metric))
            out[f"delta_{metric}"] = _fmt(_delta(new_row.get(metric), old_row.get(metric)))
        out[f"{old_label}_beats_chamfer"] = old_row.get("model_beats_chamfer", "")
        out[f"{new_label}_beats_chamfer"] = new_row.get("model_beats_chamfer", "")
        out["n_subjects"] = old_row.get("n_subjects") or new_row.get("n_subjects") or ""
        out["n_samples"] = old_row.get("n_samples") or new_row.get("n_samples") or ""
        out["n_pairs"] = old_row.get("n_pairs") or new_row.get("n_pairs") or ""
        out["n_mesh_pairs"] = old_row.get("n_mesh_pairs") or new_row.get("n_mesh_pairs") or ""
        out["n_subject_pairs"] = old_row.get("n_subject_pairs") or new_row.get("n_subject_pairs") or ""
        out_rows.append(out)
    return out_rows


def _build_sigma_rows(
    *,
    old_rows: Sequence[Dict[str, str]],
    new_rows: Sequence[Dict[str, str]],
    old_label: str,
    new_label: str,
) -> List[Dict[str, object]]:
    key_fields = ("scenario", "sigma")
    metrics = (
        "latent_spearman",
        "chamfer_spearman",
        "delta_spearman",
        "latent_pearson",
        "chamfer_pearson",
        "delta_pearson",
        "latent_drop_vs_clean",
        "chamfer_drop_vs_clean",
    )
    old_map = _load_map(old_rows, key_fields)
    new_map = _load_map(new_rows, key_fields)
    all_keys = sorted(set(old_map) | set(new_map), key=lambda item: (item[0], _to_float(item[1])))
    out_rows: List[Dict[str, object]] = []
    for key in all_keys:
        old_row = old_map.get(key, {})
        new_row = new_map.get(key, {})
        scenario, sigma = key
        out: Dict[str, object] = {"scenario": scenario, "sigma": sigma}
        for metric in metrics:
            out[f"{old_label}_{metric}"] = _fmt(old_row.get(metric))
            out[f"{new_label}_{metric}"] = _fmt(new_row.get(metric))
            out[f"delta_{metric}"] = _fmt(_delta(new_row.get(metric), old_row.get(metric)))
        out[f"{old_label}_beats_chamfer"] = old_row.get("model_beats_chamfer", "")
        out[f"{new_label}_beats_chamfer"] = new_row.get("model_beats_chamfer", "")
        out["n_subjects"] = old_row.get("n_subjects") or new_row.get("n_subjects") or ""
        out["n_samples"] = old_row.get("n_samples") or new_row.get("n_samples") or ""
        out["n_pairs"] = old_row.get("n_pairs") or new_row.get("n_pairs") or ""
        out["n_mesh_pairs"] = old_row.get("n_mesh_pairs") or new_row.get("n_mesh_pairs") or ""
        out["n_subject_pairs"] = old_row.get("n_subject_pairs") or new_row.get("n_subject_pairs") or ""
        out_rows.append(out)
    return out_rows


def _ranking_fieldnames(old_label: str, new_label: str) -> List[str]:
    return [
        "scenario",
        f"{old_label}_latent_spearman",
        f"{new_label}_latent_spearman",
        "delta_latent_spearman",
        f"{old_label}_chamfer_spearman",
        f"{new_label}_chamfer_spearman",
        "delta_chamfer_spearman",
        f"{old_label}_delta_spearman",
        f"{new_label}_delta_spearman",
        "delta_delta_spearman",
        f"{old_label}_latent_pearson",
        f"{new_label}_latent_pearson",
        "delta_latent_pearson",
        f"{old_label}_chamfer_pearson",
        f"{new_label}_chamfer_pearson",
        "delta_chamfer_pearson",
        f"{old_label}_delta_pearson",
        f"{new_label}_delta_pearson",
        "delta_delta_pearson",
        f"{old_label}_beats_chamfer",
        f"{new_label}_beats_chamfer",
        "n_subjects",
        "n_samples",
        "n_pairs",
        "n_mesh_pairs",
        "n_subject_pairs",
    ]


def _sigma_fieldnames(old_label: str, new_label: str) -> List[str]:
    return [
        "scenario",
        "sigma",
        f"{old_label}_latent_spearman",
        f"{new_label}_latent_spearman",
        "delta_latent_spearman",
        f"{old_label}_chamfer_spearman",
        f"{new_label}_chamfer_spearman",
        "delta_chamfer_spearman",
        f"{old_label}_delta_spearman",
        f"{new_label}_delta_spearman",
        "delta_delta_spearman",
        f"{old_label}_latent_pearson",
        f"{new_label}_latent_pearson",
        "delta_latent_pearson",
        f"{old_label}_chamfer_pearson",
        f"{new_label}_chamfer_pearson",
        "delta_chamfer_pearson",
        f"{old_label}_delta_pearson",
        f"{new_label}_delta_pearson",
        "delta_delta_pearson",
        f"{old_label}_latent_drop_vs_clean",
        f"{new_label}_latent_drop_vs_clean",
        "delta_latent_drop_vs_clean",
        f"{old_label}_chamfer_drop_vs_clean",
        f"{new_label}_chamfer_drop_vs_clean",
        "delta_chamfer_drop_vs_clean",
        f"{old_label}_beats_chamfer",
        f"{new_label}_beats_chamfer",
        "n_subjects",
        "n_samples",
        "n_pairs",
        "n_mesh_pairs",
        "n_subject_pairs",
    ]


def _collect_numeric(rows: Sequence[Dict[str, object]], field: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = _to_float(row.get(field))
        if math.isfinite(value):
            values.append(value)
    return values


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _build_summary(
    *,
    ranking_rows: Sequence[Dict[str, object]],
    sigma_rows: Sequence[Dict[str, object]],
    old_label: str,
    new_label: str,
    old_ranking_csv: Path,
    new_ranking_csv: Path,
    old_sigma_csv: Path,
    new_sigma_csv: Path,
) -> Dict[str, object]:
    ranking_improvements = _collect_numeric(ranking_rows, "delta_latent_spearman")
    ranking_margin_improvements = _collect_numeric(ranking_rows, "delta_delta_spearman")
    sigma_improvements = _collect_numeric(sigma_rows, "delta_latent_spearman")
    sigma_margin_improvements = _collect_numeric(sigma_rows, "delta_delta_spearman")
    improved_ranking_scenarios = sum(1 for value in ranking_improvements if value > 0.0)
    improved_sigma_points = sum(1 for value in sigma_improvements if value > 0.0)
    best_ranking = max(ranking_rows, key=lambda row: _to_float(row.get("delta_latent_spearman")), default=None)
    worst_ranking = min(ranking_rows, key=lambda row: _to_float(row.get("delta_latent_spearman")), default=None)
    best_sigma = max(sigma_rows, key=lambda row: _to_float(row.get("delta_latent_spearman")), default=None)
    worst_sigma = min(sigma_rows, key=lambda row: _to_float(row.get("delta_latent_spearman")), default=None)
    return {
        "labels": {"old": old_label, "new": new_label},
        "inputs": {
            "old_ranking_csv": str(old_ranking_csv),
            "new_ranking_csv": str(new_ranking_csv),
            "old_sigma_csv": str(old_sigma_csv),
            "new_sigma_csv": str(new_sigma_csv),
        },
        "ranking": {
            "num_rows": len(ranking_rows),
            "improved_latent_spearman_scenarios": improved_ranking_scenarios,
            "mean_delta_latent_spearman": _mean(ranking_improvements),
            "mean_delta_margin_spearman": _mean(ranking_margin_improvements),
            "best_row": best_ranking,
            "worst_row": worst_ranking,
        },
        "sigma_sweep": {
            "num_rows": len(sigma_rows),
            "improved_latent_spearman_points": improved_sigma_points,
            "mean_delta_latent_spearman": _mean(sigma_improvements),
            "mean_delta_margin_spearman": _mean(sigma_margin_improvements),
            "best_row": best_sigma,
            "worst_row": worst_sigma,
        },
    }


def _write_markdown(
    *,
    path: Path,
    summary: Dict[str, object],
    ranking_rows: Sequence[Dict[str, object]],
    sigma_rows: Sequence[Dict[str, object]],
    old_label: str,
    new_label: str,
) -> None:
    lines: List[str] = []
    lines.append("# FaceVerse Model Comparison")
    lines.append("")
    lines.append(f"- Old: `{old_label}`")
    lines.append(f"- New: `{new_label}`")
    lines.append("")
    lines.append("## Ranking Summary")
    lines.append("")
    ranking_summary = summary["ranking"]
    lines.append(
        f"- Improved scenarios (latent Spearman): "
        f"{ranking_summary['improved_latent_spearman_scenarios']}/{ranking_summary['num_rows']}"
    )
    lines.append(
        f"- Mean latent Spearman gain: {_fmt(ranking_summary['mean_delta_latent_spearman'])}"
    )
    lines.append(
        f"- Mean margin gain vs Chamfer: {_fmt(ranking_summary['mean_delta_margin_spearman'])}"
    )
    lines.append("")
    lines.append(
        f"| Scenario | {old_label} latent | {new_label} latent | New-Old | "
        f"{old_label} margin | {new_label} margin | Margin gain |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in ranking_rows:
        lines.append(
            f"| {row['scenario']} | "
            f"{row[f'{old_label}_latent_spearman']} | "
            f"{row[f'{new_label}_latent_spearman']} | "
            f"{row['delta_latent_spearman']} | "
            f"{row[f'{old_label}_delta_spearman']} | "
            f"{row[f'{new_label}_delta_spearman']} | "
            f"{row['delta_delta_spearman']} |"
        )
    lines.append("")
    lines.append("## Sigma Sweep Summary")
    lines.append("")
    sigma_summary = summary["sigma_sweep"]
    lines.append(
        f"- Improved sweep points (latent Spearman): "
        f"{sigma_summary['improved_latent_spearman_points']}/{sigma_summary['num_rows']}"
    )
    lines.append(
        f"- Mean latent Spearman gain: {_fmt(sigma_summary['mean_delta_latent_spearman'])}"
    )
    lines.append(
        f"- Mean margin gain vs Chamfer: {_fmt(sigma_summary['mean_delta_margin_spearman'])}"
    )
    lines.append("")
    lines.append(
        f"| Scenario | Sigma | {old_label} latent | {new_label} latent | New-Old | "
        f"{old_label} margin | {new_label} margin | Margin gain |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in sigma_rows:
        lines.append(
            f"| {row['scenario']} | {row['sigma']} | "
            f"{row[f'{old_label}_latent_spearman']} | "
            f"{row[f'{new_label}_latent_spearman']} | "
            f"{row['delta_latent_spearman']} | "
            f"{row[f'{old_label}_delta_spearman']} | "
            f"{row[f'{new_label}_delta_spearman']} | "
            f"{row['delta_delta_spearman']} |"
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    old_ranking_rows = _read_csv(args.old_ranking_csv)
    new_ranking_rows = _read_csv(args.new_ranking_csv)
    old_sigma_rows = _read_csv(args.old_sigma_csv)
    new_sigma_rows = _read_csv(args.new_sigma_csv)

    ranking_rows = _build_ranking_rows(
        old_rows=old_ranking_rows,
        new_rows=new_ranking_rows,
        old_label=args.old_label,
        new_label=args.new_label,
    )
    sigma_rows = _build_sigma_rows(
        old_rows=old_sigma_rows,
        new_rows=new_sigma_rows,
        old_label=args.old_label,
        new_label=args.new_label,
    )

    ranking_csv = args.out_dir / "ranking_comparison.csv"
    sigma_csv = args.out_dir / "sigma_sweep_comparison.csv"
    summary_json = args.out_dir / "comparison_summary.json"
    summary_md = args.out_dir / "comparison_summary.md"

    _write_csv(ranking_csv, _ranking_fieldnames(args.old_label, args.new_label), ranking_rows)
    _write_csv(sigma_csv, _sigma_fieldnames(args.old_label, args.new_label), sigma_rows)

    summary = _build_summary(
        ranking_rows=ranking_rows,
        sigma_rows=sigma_rows,
        old_label=args.old_label,
        new_label=args.new_label,
        old_ranking_csv=args.old_ranking_csv,
        new_ranking_csv=args.new_ranking_csv,
        old_sigma_csv=args.old_sigma_csv,
        new_sigma_csv=args.new_sigma_csv,
    )
    _write_json(summary_json, summary)
    _write_markdown(
        path=summary_md,
        summary=summary,
        ranking_rows=ranking_rows,
        sigma_rows=sigma_rows,
        old_label=args.old_label,
        new_label=args.new_label,
    )

    print(f"Wrote {ranking_csv}")
    print(f"Wrote {sigma_csv}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
