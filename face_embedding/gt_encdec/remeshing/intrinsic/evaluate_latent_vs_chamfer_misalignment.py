from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare latent-vs-Chamfer robustness breakdown outputs.")
    p.add_argument("--latent_dir", type=str, default="", help="Directory containing latent breakdown_summary.json/grid.csv")
    p.add_argument("--latent_summary", type=str, default="", help="Explicit latent breakdown_summary.json path")
    p.add_argument("--chamfer_dir", type=str, default="", help="Directory containing chamfer breakdown_summary.json/grid.csv")
    p.add_argument("--chamfer_summary", type=str, default="", help="Explicit chamfer breakdown_summary.json path")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for merged latent-vs-chamfer reports")
    return p.parse_args()


def _resolve_summary_and_grid(dir_arg: str, summary_arg: str) -> tuple[Path, Path]:
    if summary_arg:
        summary_path = Path(summary_arg).expanduser().resolve()
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_path}")
        grid_path = summary_path.with_name("breakdown_grid.csv")
    elif dir_arg:
        run_dir = Path(dir_arg).expanduser().resolve()
        summary_path = run_dir / "breakdown_summary.json"
        grid_path = run_dir / "breakdown_grid.csv"
    else:
        raise ValueError("Provide either --*_dir or --*_summary for both latent and chamfer inputs")

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid CSV not found: {grid_path}")
    return summary_path, grid_path


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _maybe_float(text: object) -> float:
    if text is None:
        return float("nan")
    raw = str(text).strip()
    if raw == "":
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.6f}"


def _sigma_key(value: object) -> str:
    sigma = _maybe_float(value)
    if not math.isfinite(sigma):
        return "nan"
    return f"{sigma:.12e}"


def _scenario_lookup(summary_pack: dict) -> Dict[str, dict]:
    scenarios = {}
    for row in summary_pack.get("summaries", []):
        scenario = str(row.get("scenario", "")).strip()
        if scenario:
            scenarios[scenario] = row
    return scenarios


def _threshold_keys(*scenario_maps: Dict[str, dict]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for scenario_map in scenario_maps:
        for row in scenario_map.values():
            for key in row.get("thresholds", {}).keys():
                if key not in seen:
                    seen.add(key)
                    ordered.append(str(key))
    return ordered


def _grid_lookup(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, Dict[str, str]]]:
    out: Dict[str, Dict[str, Dict[str, str]]] = {}
    for row in rows:
        scenario = str(row.get("scenario", "")).strip()
        if not scenario:
            continue
        out.setdefault(scenario, {})[_sigma_key(row.get("sigma"))] = row
    return out


def _scenario_order(latent_scenarios: Dict[str, dict], chamfer_scenarios: Dict[str, dict]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for source in (latent_scenarios, chamfer_scenarios):
        for scenario in source.keys():
            if scenario not in seen:
                ordered.append(scenario)
                seen.add(scenario)
    return ordered


def _metadata_mismatches(latent_pack: dict, chamfer_pack: dict) -> List[str]:
    keys = (
        "run_dir",
        "pair_mode",
        "aggregation_level",
        "subject_split",
        "n_subjects_kept",
        "n_eval_samples",
        "n_pairs",
        "n_mesh_pairs",
        "n_subject_pairs",
    )
    mismatches = []
    for key in keys:
        latent_val = latent_pack.get(key)
        chamfer_val = chamfer_pack.get(key)
        if latent_val != chamfer_val:
            mismatches.append(f"{key}: latent={latent_val!r} chamfer={chamfer_val!r}")
    return mismatches


def _build_summary_rows(
    scenario_order: Sequence[str],
    latent_scenarios: Dict[str, dict],
    chamfer_scenarios: Dict[str, dict],
    threshold_keys: Sequence[str],
) -> List[dict]:
    rows: List[dict] = []
    for scenario in scenario_order:
        latent = latent_scenarios.get(scenario, {})
        chamfer = chamfer_scenarios.get(scenario, {})
        row = {
            "scenario": scenario,
            "latent_clean_spearman_vs_gt_rmse": _maybe_float(latent.get("spearman_clean")),
            "latent_clean_pearson_vs_gt_rmse": _maybe_float(latent.get("pearson_clean")),
            "latent_noisy_max_spearman_vs_gt_rmse": _maybe_float(latent.get("spearman_noisy_max")),
            "latent_ratio_noisy_max": _maybe_float(latent.get("ratio_noisy_max")),
            "latent_worst_spearman_vs_gt_rmse": _maybe_float(latent.get("worst_spearman")),
            "latent_worst_ratio": _maybe_float(latent.get("worst_ratio")),
            "latent_auc_r": _maybe_float(latent.get("auc_r")),
            "chamfer_clean_spearman_vs_gt_rmse": _maybe_float(chamfer.get("spearman_clean")),
            "chamfer_clean_pearson_vs_gt_rmse": _maybe_float(chamfer.get("pearson_clean")),
            "chamfer_noisy_max_spearman_vs_gt_rmse": _maybe_float(chamfer.get("spearman_noisy_max")),
            "chamfer_ratio_noisy_max": _maybe_float(chamfer.get("ratio_noisy_max")),
            "chamfer_worst_spearman_vs_gt_rmse": _maybe_float(chamfer.get("worst_spearman")),
            "chamfer_worst_ratio": _maybe_float(chamfer.get("worst_ratio")),
            "chamfer_auc_r": _maybe_float(chamfer.get("auc_r")),
        }
        row["delta_clean"] = row["latent_clean_spearman_vs_gt_rmse"] - row["chamfer_clean_spearman_vs_gt_rmse"]
        row["delta_worst"] = row["latent_worst_spearman_vs_gt_rmse"] - row["chamfer_worst_spearman_vs_gt_rmse"]

        latent_thresholds = latent.get("thresholds", {})
        chamfer_thresholds = chamfer.get("thresholds", {})
        for key in threshold_keys:
            row[f"latent_{key}"] = _maybe_float(latent_thresholds.get(key))
            row[f"chamfer_{key}"] = _maybe_float(chamfer_thresholds.get(key))
        rows.append(row)
    return rows


def _build_grid_rows(
    scenario_order: Sequence[str],
    latent_grid: Dict[str, Dict[str, Dict[str, str]]],
    chamfer_grid: Dict[str, Dict[str, Dict[str, str]]],
) -> List[dict]:
    rows: List[dict] = []
    for scenario in scenario_order:
        latent_rows = latent_grid.get(scenario, {})
        chamfer_rows = chamfer_grid.get(scenario, {})
        sigma_keys = sorted(
            set(latent_rows.keys()) | set(chamfer_rows.keys()),
            key=lambda key: (_maybe_float(key), key),
        )
        for sigma_key in sigma_keys:
            latent = latent_rows.get(sigma_key, {})
            chamfer = chamfer_rows.get(sigma_key, {})
            sigma = _maybe_float(latent.get("sigma", chamfer.get("sigma")))
            is_clean = str(latent.get("is_clean", chamfer.get("is_clean", ""))).strip()
            latent_sp = _maybe_float(latent.get("spearman"))
            chamfer_sp = _maybe_float(chamfer.get("spearman"))
            latent_ratio = _maybe_float(latent.get("ratio"))
            chamfer_ratio = _maybe_float(chamfer.get("ratio"))
            rows.append(
                {
                    "scenario": scenario,
                    "sigma": sigma,
                    "is_clean": 1 if is_clean == "1" or (math.isfinite(sigma) and abs(sigma) <= 1e-15) else 0,
                    "latent_spearman_vs_gt_rmse": latent_sp,
                    "latent_pearson_vs_gt_rmse": _maybe_float(latent.get("pearson")),
                    "latent_ratio": latent_ratio,
                    "chamfer_spearman_vs_gt_rmse": chamfer_sp,
                    "chamfer_pearson_vs_gt_rmse": _maybe_float(chamfer.get("pearson")),
                    "chamfer_ratio": chamfer_ratio,
                    "latent_minus_chamfer": latent_sp - chamfer_sp,
                    "ratio_delta": latent_ratio - chamfer_ratio,
                }
            )
    return rows


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=True)
        f.write("\n")


def _write_grid_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary_md(path: Path, rows: Sequence[dict], threshold_keys: Sequence[str]) -> None:
    headers = [
        "Scenario",
        "Lat Clean Sp",
        "Lat Worst Sp",
        "Lat AUC_R",
        "Cham Clean Sp",
        "Cham Worst Sp",
        "Cham AUC_R",
        "Delta Clean",
        "Delta Worst",
    ]
    for key in threshold_keys:
        headers.extend([f"Lat {key}", f"Cham {key}"])

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Latent vs Chamfer Misalignment Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            cells = [
                str(row["scenario"]),
                _format_float(row["latent_clean_spearman_vs_gt_rmse"]),
                _format_float(row["latent_worst_spearman_vs_gt_rmse"]),
                _format_float(row["latent_auc_r"]),
                _format_float(row["chamfer_clean_spearman_vs_gt_rmse"]),
                _format_float(row["chamfer_worst_spearman_vs_gt_rmse"]),
                _format_float(row["chamfer_auc_r"]),
                _format_float(row["delta_clean"]),
                _format_float(row["delta_worst"]),
            ]
            for key in threshold_keys:
                cells.append(_format_float(row[f"latent_{key}"]))
                cells.append(_format_float(row[f"chamfer_{key}"]))
            f.write("| " + " | ".join(cells) + " |\n")


def main() -> None:
    args = parse_args()
    latent_summary_path, latent_grid_path = _resolve_summary_and_grid(args.latent_dir, args.latent_summary)
    chamfer_summary_path, chamfer_grid_path = _resolve_summary_and_grid(args.chamfer_dir, args.chamfer_summary)

    latent_pack = _load_json(latent_summary_path)
    chamfer_pack = _load_json(chamfer_summary_path)
    latent_scenarios = _scenario_lookup(latent_pack)
    chamfer_scenarios = _scenario_lookup(chamfer_pack)
    if not latent_scenarios:
        raise RuntimeError(f"No scenario summaries found in latent pack: {latent_summary_path}")
    if not chamfer_scenarios:
        raise RuntimeError(f"No scenario summaries found in chamfer pack: {chamfer_summary_path}")

    threshold_keys = _threshold_keys(latent_scenarios, chamfer_scenarios)
    ordered_scenarios = _scenario_order(latent_scenarios, chamfer_scenarios)

    summary_rows = _build_summary_rows(
        scenario_order=ordered_scenarios,
        latent_scenarios=latent_scenarios,
        chamfer_scenarios=chamfer_scenarios,
        threshold_keys=threshold_keys,
    )
    grid_rows = _build_grid_rows(
        scenario_order=ordered_scenarios,
        latent_grid=_grid_lookup(_load_csv(latent_grid_path)),
        chamfer_grid=_grid_lookup(_load_csv(chamfer_grid_path)),
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "latent_summary_path": str(latent_summary_path),
        "latent_grid_path": str(latent_grid_path),
        "chamfer_summary_path": str(chamfer_summary_path),
        "chamfer_grid_path": str(chamfer_grid_path),
        "run_dir_latent": latent_pack.get("run_dir", ""),
        "run_dir_chamfer": chamfer_pack.get("run_dir", ""),
        "pair_mode_latent": latent_pack.get("pair_mode", ""),
        "pair_mode_chamfer": chamfer_pack.get("pair_mode", ""),
        "aggregation_level_latent": latent_pack.get("aggregation_level", ""),
        "aggregation_level_chamfer": chamfer_pack.get("aggregation_level", ""),
        "subject_split_latent": latent_pack.get("subject_split", ""),
        "subject_split_chamfer": chamfer_pack.get("subject_split", ""),
        "metadata_mismatches": _metadata_mismatches(latent_pack, chamfer_pack),
        "threshold_keys": list(threshold_keys),
        "summary_rows": summary_rows,
        "n_grid_rows": len(grid_rows),
    }

    _write_json(out_dir / "latent_vs_chamfer_summary.json", payload)
    _write_grid_csv(out_dir / "latent_vs_chamfer_grid.csv", grid_rows)
    _write_summary_md(out_dir / "latent_vs_chamfer_summary.md", summary_rows, threshold_keys)


if __name__ == "__main__":
    main()
