#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "checking_assumptions" / "outputs" / "full_method_comparison"


ORIGINAL_SOURCES = {
    "raw_noicp": REPO_ROOT
    / "face_embedding"
    / "gt_encdec"
    / "remeshing"
    / "intrinsic"
    / "newdata"
    / "dn_mixed_topology_v1"
    / "perturbation_ranking_vs_chamfer_topology_breakdown"
    / "best_by_xtopo_mesh_clean_split-eval_pairs-cross_topology_topopairs-cross_only_ordered-1_subjects-100_meshes-10_scenarios-clean-jitter-translation-rotation-mixed_noicp",
    "rigid_only": REPO_ROOT
    / "face_embedding"
    / "gt_encdec"
    / "remeshing"
    / "intrinsic"
    / "newdata"
    / "dn_mixed_topology_v1"
    / "perturbation_ranking_vs_registered_chamfer_topology_breakdown"
    / "best_by_xtopo_mesh_clean_split-eval_pairs-cross_topology_topopairs-cross_only_ordered-1_subjects-100_meshes-10_scenarios-clean-jitter-translation-rotation-mixed_icp128_rigidonly",
    "nicp_corr": REPO_ROOT
    / "face_embedding"
    / "gt_encdec"
    / "remeshing"
    / "intrinsic"
    / "newdata"
    / "dn_mixed_topology_v1"
    / "perturbation_ranking_vs_nicp_correspondence_topology_breakdown"
    / "best_by_xtopo_mesh_clean_split-eval_pairs-cross_topology_topopairs-cross_only_ordered-1_subjects-100_meshes-10_scenarios-clean-jitter-translation-rotation-mixed_icp128_nicp128_iter8",
    "rigid_cpd": REPO_ROOT
    / "face_embedding"
    / "gt_encdec"
    / "remeshing"
    / "intrinsic"
    / "newdata"
    / "dn_mixed_topology_v1"
    / "perturbation_ranking_vs_registered_chamfer_topology_breakdown"
    / "best_by_xtopo_mesh_clean_split-eval_pairs-cross_topology_topopairs-cross_only_ordered-1_subjects-100_meshes-10_scenarios-clean-jitter-translation-rotation-mixed_icp128_cpd64",
}

SURROGATE_SOURCES = {
    "raw_noicp": REPO_ROOT
    / "checking_assumptions"
    / "outputs"
    / "full_meshpair_surrogate_gt_raw_vs_rigid"
    / "raw_noicp_clean_translation",
    "rigid_only": REPO_ROOT
    / "checking_assumptions"
    / "outputs"
    / "full_meshpair_surrogate_gt_raw_vs_rigid"
    / "rigid_only_clean_translation",
}

METHOD_COLORS = {
    "raw_noicp": "#1f77b4",
    "rigid_only": "#ff7f0e",
    "nicp_corr": "#2ca02c",
    "rigid_cpd": "#d62728",
}

METHOD_LABELS = {
    "raw_noicp": "Raw Chamfer",
    "rigid_only": "Rigid ICP + Chamfer",
    "nicp_corr": "Rigid ICP + nICP corr",
    "rigid_cpd": "Rigid ICP + CPD",
}


def _metric_column(df: pd.DataFrame) -> str:
    candidates = [
        col
        for col in df.columns
        if col.endswith("_spearman") and col not in {"latent_spearman", "delta_spearman"}
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Could not infer metric spearman column from {list(df.columns)}")
    return candidates[0]


def _load_summary_table(root: Path, method: str, gt_reference: str, kind: str) -> pd.DataFrame:
    path = root / f"overall_{kind}_summary.csv"
    df = pd.read_csv(path)
    metric_col = _metric_column(df)
    df = df.copy()
    df["method"] = method
    df["gt_reference"] = gt_reference
    df["metric_spearman"] = df[metric_col]
    df["metric_column"] = metric_col
    return df


def _scenario_table() -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for method, root in ORIGINAL_SOURCES.items():
        rows.append(_load_summary_table(root, method=method, gt_reference="original_gt", kind="scenario"))
    for method, root in SURROGATE_SOURCES.items():
        rows.append(_load_summary_table(root, method=method, gt_reference="surrogate_gt_normalized", kind="scenario"))
    return pd.concat(rows, ignore_index=True)


def _topology_table(gt_reference: str) -> pd.DataFrame:
    sources = ORIGINAL_SOURCES if gt_reference == "original_gt" else SURROGATE_SOURCES
    rows: List[pd.DataFrame] = []
    for method, root in sources.items():
        rows.append(_load_summary_table(root, method=method, gt_reference=gt_reference, kind="topology_breakdown"))
    return pd.concat(rows, ignore_index=True)


def _write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _plot_scenario_bars(df: pd.DataFrame, gt_reference: str, out_path: Path) -> None:
    sub = df[df["gt_reference"] == gt_reference].copy()
    scenarios = list(dict.fromkeys(sub["scenario"].tolist()))
    methods = [m for m in METHOD_LABELS if m in set(sub["method"])]
    width = 0.18
    xs = list(range(len(scenarios)))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    offsets = [((idx - (len(methods) - 1) / 2.0) * width) for idx in range(len(methods))]
    for offset, method in zip(offsets, methods):
        vals = [
            float(
                sub.loc[
                    (sub["scenario"] == scenario) & (sub["method"] == method),
                    "metric_spearman",
                ].iloc[0]
            )
            for scenario in scenarios
        ]
        ax.bar(
            [x + offset for x in xs],
            vals,
            width=width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            alpha=0.9,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Metric Spearman vs GT")
    ax.set_title(f"Method Comparison: {gt_reference.replace('_', ' ')}")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _build_flip_table(original_topology: pd.DataFrame, surrogate_topology: pd.DataFrame) -> pd.DataFrame:
    merge_cols = ["scenario", "topology_a", "topology_b", "ordered_pair_label"]
    original_pivot = (
        original_topology.pivot_table(index=merge_cols, columns="method", values="metric_spearman")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    surrogate_pivot = (
        surrogate_topology.pivot_table(index=merge_cols, columns="method", values="metric_spearman")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    merged = original_pivot.merge(
        surrogate_pivot,
        on=merge_cols,
        suffixes=("_original_gt", "_surrogate_gt"),
    )
    merged["raw_minus_rigid_original_gt"] = merged["raw_noicp_original_gt"] - merged["rigid_only_original_gt"]
    merged["raw_minus_rigid_surrogate_gt"] = merged["raw_noicp_surrogate_gt"] - merged["rigid_only_surrogate_gt"]
    merged["winner_original_gt"] = merged.apply(
        lambda row: "raw_noicp" if row["raw_noicp_original_gt"] >= row["rigid_only_original_gt"] else "rigid_only",
        axis=1,
    )
    merged["winner_surrogate_gt"] = merged.apply(
        lambda row: "raw_noicp" if row["raw_noicp_surrogate_gt"] >= row["rigid_only_surrogate_gt"] else "rigid_only",
        axis=1,
    )
    merged["winner_flips"] = merged["winner_original_gt"] != merged["winner_surrogate_gt"]
    return merged.sort_values(["scenario", "winner_flips", "raw_minus_rigid_original_gt"], ascending=[True, False, False])


def _plot_flip_scatter(flip_df: pd.DataFrame, scenario: str, out_path: Path) -> None:
    sub = flip_df[flip_df["scenario"] == scenario].copy()
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    colors = sub["winner_flips"].map({True: "#d62728", False: "#1f77b4"}).tolist()
    ax.scatter(
        sub["raw_minus_rigid_original_gt"],
        sub["raw_minus_rigid_surrogate_gt"],
        c=colors,
        alpha=0.8,
        edgecolors="none",
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Raw - Rigid on original GT")
    ax.set_ylabel("Raw - Rigid on surrogate GT")
    ax.set_title(f"Topology-Pair Winner Flip: {scenario}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scenario_df = _scenario_table().sort_values(["gt_reference", "scenario", "method"]).reset_index(drop=True)
    scenario_df.to_csv(OUT_DIR / "scenario_method_comparison.csv", index=False)

    original_topology = _topology_table("original_gt").sort_values(["scenario", "method", "ordered_pair_label"]).reset_index(drop=True)
    surrogate_topology = _topology_table("surrogate_gt_normalized").sort_values(["scenario", "method", "ordered_pair_label"]).reset_index(drop=True)
    original_topology.to_csv(OUT_DIR / "topology_method_comparison_original_gt.csv", index=False)
    surrogate_topology.to_csv(OUT_DIR / "topology_method_comparison_surrogate_gt.csv", index=False)

    _plot_scenario_bars(scenario_df, gt_reference="original_gt", out_path=OUT_DIR / "scenario_bars_original_gt.png")
    _plot_scenario_bars(scenario_df, gt_reference="surrogate_gt_normalized", out_path=OUT_DIR / "scenario_bars_surrogate_gt.png")

    flip_df = _build_flip_table(original_topology, surrogate_topology)
    flip_df.to_csv(OUT_DIR / "raw_vs_rigid_flip_table.csv", index=False)
    for scenario in sorted(flip_df["scenario"].unique()):
        _plot_flip_scatter(flip_df, scenario=scenario, out_path=OUT_DIR / f"raw_vs_rigid_flip_scatter_{scenario}.png")

    summary_payload = {
        "original_gt": {
            "scenario_rows": int((scenario_df["gt_reference"] == "original_gt").sum()),
            "methods": sorted(scenario_df.loc[scenario_df["gt_reference"] == "original_gt", "method"].unique().tolist()),
        },
        "surrogate_gt_normalized": {
            "scenario_rows": int((scenario_df["gt_reference"] == "surrogate_gt_normalized").sum()),
            "methods": sorted(scenario_df.loc[scenario_df["gt_reference"] == "surrogate_gt_normalized", "method"].unique().tolist()),
        },
        "flip_counts": {
            scenario: int(group["winner_flips"].sum())
            for scenario, group in flip_df.groupby("scenario")
        },
    }
    _write_json(OUT_DIR / "summary.json", summary_payload)

    lines = [
        "# Full Method Comparison",
        "",
        "This bundle compares the completed REMESH mesh-pair benchmarks under two GT definitions:",
        "",
        "- `original_gt`: the historical benchmark GT matrix",
        "- `surrogate_gt_normalized`: a GT surrogate built directly in the normalized geometry space used by the loader",
        "",
        "Main files:",
        "- `scenario_method_comparison.csv`: scenario-level Spearman comparison across methods",
        "- `raw_vs_rigid_flip_table.csv`: topology-pair level raw-vs-rigid comparison under both GTs",
        "- `scenario_bars_original_gt.png`: macro comparison under original GT",
        "- `scenario_bars_surrogate_gt.png`: macro comparison under surrogate GT",
        "",
        "Winner flips raw vs rigid by scenario:",
    ]
    for scenario, count in summary_payload["flip_counts"].items():
        lines.append(f"- `{scenario}`: `{count}` topology-pair winner flips")
    lines.append("")
    lines.append("Interpretation target:")
    lines.append("- If many topology-pair winners flip when the GT is moved into normalized space, the rigid-only conclusion is GT-sensitive rather than universally stable.")
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
