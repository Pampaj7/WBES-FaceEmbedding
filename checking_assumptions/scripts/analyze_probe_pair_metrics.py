#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROBE_DIR = REPO_ROOT / "checking_assumptions" / "outputs" / "same_vs_diff_gap_probe_all12_nicp"

METRIC_LABELS = {
    "latent_distance": "Latent",
    "raw_chamfer": "Raw Chamfer",
    "rigid_registered_chamfer": "Rigid ICP + Chamfer",
    "nicp_correspondence": "Rigid ICP + nICP corr",
    "cpd_registered_chamfer": "Rigid ICP + CPD",
}

METRIC_COLORS = {
    "latent_distance": "#9467bd",
    "raw_chamfer": "#1f77b4",
    "rigid_registered_chamfer": "#ff7f0e",
    "nicp_correspondence": "#2ca02c",
    "cpd_registered_chamfer": "#d62728",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze pair-level same-vs-different probe outputs.")
    p.add_argument("--probe_dir", type=str, default=str(DEFAULT_PROBE_DIR))
    p.add_argument("--label", type=str, default="")
    p.add_argument("--out_dir", type=str, default="")
    return p.parse_args()


def _metric_columns(df: pd.DataFrame) -> List[str]:
    cols = [
        col
        for col in [
            "latent_distance",
            "raw_chamfer",
            "rigid_registered_chamfer",
            "nicp_correspondence",
            "cpd_registered_chamfer",
        ]
        if col in df.columns and np.isfinite(pd.to_numeric(df[col], errors="coerce")).any()
    ]
    return cols


def _spearman_corr(x: pd.Series, y: pd.Series) -> float:
    xr = x.rank(method="average")
    yr = y.rank(method="average")
    return float(xr.corr(yr))


def _distribution_summary(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    rows: List[dict] = []
    for metric in metrics:
        metric_vals = pd.to_numeric(df[metric], errors="coerce")
        for pair_type, group in df.assign(_metric=metric_vals).groupby("pair_type"):
            vals = group["_metric"].dropna()
            rows.append(
                {
                    "metric": metric,
                    "pair_type": pair_type,
                    "count": int(vals.shape[0]),
                    "mean": float(vals.mean()) if not vals.empty else np.nan,
                    "median": float(vals.median()) if not vals.empty else np.nan,
                    "std": float(vals.std(ddof=0)) if not vals.empty else np.nan,
                    "q10": float(vals.quantile(0.10)) if not vals.empty else np.nan,
                    "q90": float(vals.quantile(0.90)) if not vals.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _gt_quantile_summary(diff_df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    work = diff_df.copy()
    work["gt_quantile"] = pd.qcut(work["gt_distance"], q=min(5, work.shape[0]), duplicates="drop")
    rows: List[dict] = []
    for metric in metrics:
        metric_vals = pd.to_numeric(work[metric], errors="coerce")
        for quantile, group in work.assign(_metric=metric_vals).groupby("gt_quantile", observed=False):
            vals = group["_metric"].dropna()
            rows.append(
                {
                    "metric": metric,
                    "gt_quantile": str(quantile),
                    "count": int(vals.shape[0]),
                    "gt_mean": float(group["gt_distance"].mean()),
                    "metric_mean": float(vals.mean()) if not vals.empty else np.nan,
                    "metric_median": float(vals.median()) if not vals.empty else np.nan,
                    "metric_q10": float(vals.quantile(0.10)) if not vals.empty else np.nan,
                    "metric_q90": float(vals.quantile(0.90)) if not vals.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _rank_error_summary(diff_df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    gt_rank = diff_df["gt_distance"].rank(method="average")
    rows: List[dict] = []
    for metric in metrics:
        vals = pd.to_numeric(diff_df[metric], errors="coerce")
        mask = vals.notna()
        if int(mask.sum()) < 2:
            continue
        metric_rank = vals[mask].rank(method="average")
        gt_rank_masked = gt_rank[mask]
        abs_rank_error = (metric_rank - gt_rank_masked).abs()
        rows.append(
            {
                "metric": metric,
                "count": int(mask.sum()),
                "mean_abs_rank_error": float(abs_rank_error.mean()),
                "median_abs_rank_error": float(abs_rank_error.median()),
                "spearman_vs_gt": _spearman_corr(diff_df.loc[mask, "gt_distance"], vals[mask]),
                "pearson_vs_gt": float(diff_df.loc[mask, "gt_distance"].corr(vals[mask])),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_rank_error")


def _shrinkage_summary(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    if "raw_chamfer" not in metrics:
        return pd.DataFrame()
    rows: List[dict] = []
    raw = pd.to_numeric(df["raw_chamfer"], errors="coerce")
    for metric in metrics:
        if metric in {"latent_distance", "raw_chamfer"}:
            continue
        vals = pd.to_numeric(df[metric], errors="coerce")
        delta = vals - raw
        ratio = vals / raw.replace(0.0, np.nan)
        work = df.assign(metric_value=vals, delta_from_raw=delta, ratio_to_raw=ratio)
        for pair_type, group in work.groupby("pair_type"):
            rows.append(
                {
                    "metric": metric,
                    "pair_type": pair_type,
                    "count": int(group["delta_from_raw"].notna().sum()),
                    "delta_mean": float(group["delta_from_raw"].mean()),
                    "delta_median": float(group["delta_from_raw"].median()),
                    "ratio_mean": float(group["ratio_to_raw"].mean()),
                    "ratio_median": float(group["ratio_to_raw"].median()),
                }
            )
    return pd.DataFrame(rows)


def _rank_inversion_examples(diff_df: pd.DataFrame, metrics: List[str], top_k: int = 25) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    gt_rank = diff_df["gt_distance"].rank(method="average")
    raw_rank = pd.to_numeric(diff_df["raw_chamfer"], errors="coerce").rank(method="average")
    raw_abs_error = (raw_rank - gt_rank).abs()
    base_cols = [
        "scenario",
        "subject_a",
        "topology_a",
        "sample_name_a",
        "subject_b",
        "topology_b",
        "sample_name_b",
        "gt_distance",
        "raw_chamfer",
        "rigid_registered_chamfer",
        "nicp_correspondence",
    ]
    keep_cols = [col for col in base_cols if col in diff_df.columns]
    for metric in metrics:
        if metric in {"latent_distance", "raw_chamfer"}:
            continue
        vals = pd.to_numeric(diff_df[metric], errors="coerce")
        mask = vals.notna()
        if int(mask.sum()) < 2:
            continue
        metric_rank = vals[mask].rank(method="average")
        gt_rank_masked = gt_rank[mask]
        raw_abs_error_masked = raw_abs_error[mask]
        metric_abs_error = (metric_rank - gt_rank_masked).abs()
        work = diff_df.loc[mask, keep_cols].copy()
        work["metric"] = metric
        work["metric_value"] = vals[mask].values
        work["gt_rank"] = gt_rank_masked.values
        work["raw_abs_rank_error"] = raw_abs_error_masked.values
        work["metric_abs_rank_error"] = metric_abs_error.values
        work["extra_abs_rank_error_vs_raw"] = work["metric_abs_rank_error"] - work["raw_abs_rank_error"]
        work = work.sort_values(
            ["extra_abs_rank_error_vs_raw", "gt_distance"],
            ascending=[False, False],
        ).head(top_k)
        rows.append(work)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _plot_gt_vs_metric(diff_df: pd.DataFrame, metrics: List[str], out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.0), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        vals = pd.to_numeric(diff_df[metric], errors="coerce")
        mask = vals.notna()
        ax.scatter(
            diff_df.loc[mask, "gt_distance"],
            vals[mask],
            s=18,
            alpha=0.55,
            color=METRIC_COLORS.get(metric, "#333333"),
            edgecolors="none",
        )
        sp = _spearman_corr(diff_df.loc[mask, "gt_distance"], vals[mask]) if int(mask.sum()) >= 2 else np.nan
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)}\nSpearman={sp:.3f}")
        ax.set_xlabel("GT distance")
        ax.set_ylabel("Observed distance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_same_vs_diff(df: pd.DataFrame, metrics: List[str], out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.0), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        vals_same = pd.to_numeric(df.loc[df["pair_type"] == "same_subject", metric], errors="coerce").dropna()
        vals_diff = pd.to_numeric(df.loc[df["pair_type"] == "different_subject", metric], errors="coerce").dropna()
        if vals_same.empty or vals_diff.empty:
            continue
        bins = np.linspace(float(min(vals_same.min(), vals_diff.min())), float(max(vals_same.max(), vals_diff.max())), 35)
        ax.hist(vals_same, bins=bins, alpha=0.45, density=True, label="same", color="#4daf4a")
        ax.hist(vals_diff, bins=bins, alpha=0.45, density=True, label="different", color="#e41a1c")
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_xlabel("Observed distance")
        ax.set_ylabel("Density")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_shrinkage(diff_df: pd.DataFrame, metrics: List[str], out_path: Path) -> None:
    compare_metrics = [metric for metric in metrics if metric not in {"latent_distance", "raw_chamfer"}]
    if not compare_metrics:
        return
    raw = pd.to_numeric(diff_df["raw_chamfer"], errors="coerce")
    fig, axes = plt.subplots(1, len(compare_metrics), figsize=(4.2 * len(compare_metrics), 4.0), squeeze=False)
    for ax, metric in zip(axes[0], compare_metrics):
        vals = pd.to_numeric(diff_df[metric], errors="coerce")
        mask = vals.notna() & raw.notna()
        delta = vals[mask] - raw[mask]
        ax.scatter(
            diff_df.loc[mask, "gt_distance"],
            delta,
            s=18,
            alpha=0.55,
            color=METRIC_COLORS.get(metric, "#333333"),
            edgecolors="none",
        )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)} - Raw")
        ax.set_xlabel("GT distance")
        ax.set_ylabel("Delta from raw")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_rank_error(rank_summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    labels = [METRIC_LABELS.get(metric, metric) for metric in rank_summary["metric"]]
    ax.bar(
        labels,
        rank_summary["mean_abs_rank_error"],
        color=[METRIC_COLORS.get(metric, "#333333") for metric in rank_summary["metric"]],
    )
    ax.set_ylabel("Mean absolute rank error")
    ax.set_title("Different-subject rank error vs GT")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def analyze_probe(probe_dir: Path, out_dir: Path, label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    overall = pd.read_csv(probe_dir / "overall_metric_summary.csv")
    overall.to_csv(out_dir / "overall_metric_summary.csv", index=False)

    summary_payload: Dict[str, Dict] = {"label": label, "scenarios": {}}
    readme_lines = [
        f"# Probe Analysis: {label}",
        "",
        f"Source probe dir: `{probe_dir}`",
        "",
    ]

    for scenario_dir in sorted(p for p in probe_dir.iterdir() if p.is_dir()):
        pair_path = scenario_dir / "pair_metrics.csv"
        if not pair_path.exists():
            continue
        scenario = scenario_dir.name
        scenario_out = out_dir / scenario
        scenario_out.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(pair_path)
        metrics = _metric_columns(df)
        diff_df = df[df["pair_type"] == "different_subject"].copy()

        distribution = _distribution_summary(df, metrics)
        quantiles = _gt_quantile_summary(diff_df, metrics)
        rank_summary = _rank_error_summary(diff_df, metrics)
        shrinkage = _shrinkage_summary(df, metrics)
        inversions = _rank_inversion_examples(diff_df, metrics)

        distribution.to_csv(scenario_out / "distribution_summary.csv", index=False)
        quantiles.to_csv(scenario_out / "gt_quantile_summary.csv", index=False)
        rank_summary.to_csv(scenario_out / "rank_error_summary.csv", index=False)
        shrinkage.to_csv(scenario_out / "shrinkage_summary.csv", index=False)
        inversions.to_csv(scenario_out / "rank_inversion_examples.csv", index=False)

        _plot_gt_vs_metric(diff_df, metrics, scenario_out / "gt_vs_metric_scatter.png")
        _plot_same_vs_diff(df, metrics, scenario_out / "same_vs_diff_hist.png")
        _plot_shrinkage(diff_df, metrics, scenario_out / "shrinkage_vs_gt.png")
        _plot_rank_error(rank_summary, scenario_out / "rank_error_bar.png")

        overall_scenario = overall[overall["scenario"] == scenario].copy()
        best_metric_row = overall_scenario.sort_values("different_only_spearman_vs_gt", ascending=False).iloc[0]
        summary_payload["scenarios"][scenario] = {
            "metrics": metrics,
            "best_metric_by_diff_spearman": str(best_metric_row["metric"]),
            "best_metric_diff_spearman": float(best_metric_row["different_only_spearman_vs_gt"]),
            "n_same_pairs": int((df["pair_type"] == "same_subject").sum()),
            "n_diff_pairs": int((df["pair_type"] == "different_subject").sum()),
        }

        readme_lines.extend(
            [
                f"## {scenario}",
                "",
                f"- metrics analyzed: `{', '.join(metrics)}`",
                f"- best metric on different-subject Spearman vs GT: `{best_metric_row['metric']}` = `{float(best_metric_row['different_only_spearman_vs_gt']):.4f}`",
                f"- same pairs: `{int((df['pair_type'] == 'same_subject').sum())}`",
                f"- different pairs: `{int((df['pair_type'] == 'different_subject').sum())}`",
                "",
                "Artifacts:",
                f"- `{scenario}/distribution_summary.csv`",
                f"- `{scenario}/gt_quantile_summary.csv`",
                f"- `{scenario}/rank_error_summary.csv`",
                f"- `{scenario}/shrinkage_summary.csv`",
                f"- `{scenario}/rank_inversion_examples.csv`",
                f"- `{scenario}/gt_vs_metric_scatter.png`",
                f"- `{scenario}/same_vs_diff_hist.png`",
                f"- `{scenario}/shrinkage_vs_gt.png`",
                f"- `{scenario}/rank_error_bar.png`",
                "",
            ]
        )

    _write_json(out_dir / "summary.json", summary_payload)
    (out_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    probe_dir = Path(args.probe_dir).expanduser().resolve()
    label = args.label or probe_dir.name
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = probe_dir / "analysis"
    analyze_probe(probe_dir=probe_dir, out_dir=out_dir, label=label)


if __name__ == "__main__":
    main()
