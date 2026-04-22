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
DEFAULT_PROBE_DIR = REPO_ROOT / "checking_assumptions" / "outputs" / "same_vs_diff_gap_probe_rigid_scenarios"
DEFAULT_OUT_DIR = REPO_ROOT / "checking_assumptions" / "outputs" / "perturbation_stability_analysis"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze Ec / Ep / D=|Ec-Ep| for perturbation-only scenarios.")
    p.add_argument("--probe_dir", type=str, default=str(DEFAULT_PROBE_DIR))
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--scenarios", type=str, default="clean,translation,rotation,mixed")
    return p.parse_args()


def _load_pair_metrics(probe_dir: Path, scenario: str) -> pd.DataFrame:
    path = probe_dir / scenario / "pair_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing pair metrics file: {path}")
    return pd.read_csv(path)


def _pair_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["sample_name_a"].astype(str)
        + "||"
        + df["sample_name_b"].astype(str)
        + "||"
        + df["topology_a"].astype(str)
        + "||"
        + df["topology_b"].astype(str)
    )


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    xr = pd.to_numeric(x, errors="coerce").rank(method="average")
    yr = pd.to_numeric(y, errors="coerce").rank(method="average")
    mask = xr.notna() & yr.notna()
    if int(mask.sum()) < 2:
        return float("nan")
    return float(xr[mask].corr(yr[mask]))


def _save_plot(df: pd.DataFrame, metric_name: str, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, pair_type in zip(axes, ["same_subject", "different_subject"]):
        sub = df[df["pair_type"] == pair_type].copy()
        ax.hist(sub["D_abs_delta"], bins=40, color="#1f77b4" if pair_type == "different_subject" else "#2ca02c", alpha=0.75)
        ax.set_title(f"{metric_name}: {pair_type}")
        ax.set_xlabel("|Ec - Ep|")
        ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_dir / f"{metric_name}_D_hist.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.scatter(df["Ec"], df["Ep"], s=15, alpha=0.55, color="#d62728", edgecolors="none")
    lo = float(np.nanmin([df["Ec"].min(), df["Ep"].min()]))
    hi = float(np.nanmax([df["Ec"].max(), df["Ep"].max()]))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8)
    ax.set_xlabel("Ec (clean Chamfer)")
    ax.set_ylabel("Ep (perturbed Chamfer)")
    ax.set_title(f"{metric_name}: clean vs perturbed")
    fig.tight_layout()
    fig.savefig(out_dir / f"{metric_name}_Ec_vs_Ep.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    probe_dir = Path(args.probe_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_names = [s.strip() for s in str(args.scenarios).split(",") if s.strip()]
    frames = []
    for scenario in scenario_names:
        df = _load_pair_metrics(probe_dir, scenario)
        df = df.copy()
        df["scenario"] = scenario
        df["pair_key"] = _pair_key(df)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    if "raw_chamfer" not in combined.columns:
        raise RuntimeError("Expected raw_chamfer column in pair_metrics.csv")

    clean = combined[combined["scenario"] == "clean"][["pair_key", "raw_chamfer", "gt_distance", "pair_type"]].rename(
        columns={"raw_chamfer": "Ec"}
    )

    rows: List[dict] = []
    for scenario in scenario_names:
        if scenario == "clean":
            continue
        sub = combined[combined["scenario"] == scenario][["pair_key", "raw_chamfer"]].rename(columns={"raw_chamfer": "Ep"})
        merged = clean.merge(sub, on="pair_key", how="inner")
        if merged.empty:
            continue
        merged["scenario"] = scenario
        merged["D"] = merged["Ep"] - merged["Ec"]
        merged["D_abs_delta"] = merged["D"].abs()
        merged["Ec_rank"] = merged["Ec"].rank(method="average")
        merged["Ep_rank"] = merged["Ep"].rank(method="average")
        merged["rank_abs_change"] = (merged["Ep_rank"] - merged["Ec_rank"]).abs()
        merged["pair_type"] = clean.set_index("pair_key").loc[merged["pair_key"], "pair_type"].values
        rows.append(merged)

    if not rows:
        raise RuntimeError("No perturbation comparisons could be built")

    pairwise = pd.concat(rows, ignore_index=True)
    pairwise.to_csv(out_dir / "pairwise_perturbation_stability.csv", index=False)

    summary_rows: List[dict] = []
    for scenario, group in pairwise.groupby("scenario", sort=False):
        summary_rows.append(
            {
                "scenario": scenario,
                "count": int(group.shape[0]),
                "D_abs_mean": float(group["D_abs_delta"].mean()),
                "D_abs_median": float(group["D_abs_delta"].median()),
                "D_abs_std": float(group["D_abs_delta"].std(ddof=0)),
                "D_abs_q10": float(group["D_abs_delta"].quantile(0.10)),
                "D_abs_q90": float(group["D_abs_delta"].quantile(0.90)),
                "Ec_mean": float(group["Ec"].mean()),
                "Ep_mean": float(group["Ep"].mean()),
                "Ec_vs_Ep_pearson": float(group["Ec"].corr(group["Ep"])),
                "Ec_vs_Ep_spearman": _safe_spearman(group["Ec"], group["Ep"]),
                "rank_abs_change_mean": float(group["rank_abs_change"].mean()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "scenario_stability_summary.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
        f.write("\n")

    for scenario, group in pairwise.groupby("scenario", sort=False):
        _save_plot(group, metric_name=scenario, out_dir=out_dir)

    md_lines = [
        "# Perturbation Stability Analysis",
        "",
        "This analysis compares the same mesh pairs between the clean scenario and the perturbed scenarios.",
        "",
        "| scenario | count | mean | median | std | q10 | q90 | pearson(Ec,Ep) | spearman(Ec,Ep) | mean rank change |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        md_lines.append(
            "| {scenario} | {count} | {mean:.6f} | {median:.6f} | {std:.6f} | {q10:.6f} | {q90:.6f} | {pearson:.6f} | {spearman:.6f} | {rank:.3f} |".format(
                scenario=row["scenario"],
                count=row["count"],
                mean=row["D_abs_mean"],
                median=row["D_abs_median"],
                std=row["D_abs_std"],
                q10=row["D_abs_q10"],
                q90=row["D_abs_q90"],
                pearson=row["Ec_vs_Ep_pearson"],
                spearman=row["Ec_vs_Ep_spearman"],
                rank=row["rank_abs_change_mean"],
            )
        )
    (out_dir / "README.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[done] wrote perturbation stability analysis to {out_dir}")


if __name__ == "__main__":
    main()
