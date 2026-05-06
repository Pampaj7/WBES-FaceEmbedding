#!/usr/bin/env python3
"""Subject-level bootstrap confidence intervals for ranking correlations.

This script reuses stored evaluation CSVs. It does not recompute mesh distances
or retrain any model.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "paper_artifacts" / "bootstrap_ci"

REMESH_ORIGINAL_DIR = (
    REPO_ROOT / "faceBench" / "latentVSpipeline" / "outputs" / "facebench_original_original_100subj_norm"
)
REMESH_CROSS_TOPO_DIR = (
    REPO_ROOT / "faceBench" / "latentVSpipeline" / "outputs" / "facebench_remesh_100subj_norm"
)
REMESH_SAME_TOPO_DIR = (
    REPO_ROOT / "faceBench" / "latentVSpipeline" / "outputs" / "facebench_same_topology_raw_100subj_norm"
)
REMESH_TABLE1_CROSS_DIR = REPO_ROOT / "paper_artifacts" / "bootstrap_ci" / "table1_pairlevel_exact"

TOPOLOGIES = ("crop", "down8k", "noisy", "original", "remesh", "up60k")

METHODS = [
    ("Latent distance", "latent_distance"),
    ("Chamfer", "raw_chamfer"),
    ("ICP + Chamfer", "rigid_p2p"),
    ("ICP + NICP + P2P", "nicp_p2p"),
    ("ICP + NICP + P2Tri", "nicp_p2tri"),
]

FACEVERSE_METHODS = [
    ("Latent", "latent_spearman"),
    ("Chamfer", "chamfer_spearman"),
]


@dataclass(frozen=True)
class BootstrapTask:
    dataset: str
    setting: str
    csv_path: Path
    filter_name: str
    filter_fn: Callable[[pd.DataFrame], pd.Series]
    methods: tuple[tuple[str, str], ...]
    table_setting: str | None = None


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute subject-level bootstrap CIs for stored Spearman ranking results."
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--remesh-original-dir", type=Path, default=REMESH_ORIGINAL_DIR)
    parser.add_argument("--remesh-cross-topology-dir", type=Path, default=REMESH_CROSS_TOPO_DIR)
    parser.add_argument("--remesh-same-topology-dir", type=Path, default=REMESH_SAME_TOPO_DIR)
    parser.add_argument(
        "--remesh-table1-cross-dir",
        type=Path,
        default=REMESH_TABLE1_CROSS_DIR,
        help="Pair-level artifact directory for the exact REMESH Table 1 off-diagonal cells.",
    )
    parser.add_argument(
        "--include-faceverse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Try to include optional FaceVerse CIs when pair-level files are available.",
    )
    return parser.parse_args()


def read_pairs(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        warn(f"Missing required CSV: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        warn(f"Could not read {path}: {exc}")
        return None


def find_pair_csv(result_dir: Path, pair_label: str | None = None) -> Path:
    if pair_label:
        pair_metrics = result_dir / pair_label / "pair_metrics.csv"
        if pair_metrics.exists():
            return pair_metrics
    all_pairs = result_dir / "all_pairs.csv"
    if all_pairs.exists():
        return all_pairs
    if pair_label:
        return result_dir / pair_label / "pair_metrics.csv"
    return all_pairs


def finite_spearman(gt: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(gt) & np.isfinite(values)
    if int(mask.sum()) < 3:
        return math.nan
    x = gt[mask]
    y = values[mask]
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return math.nan
    return float(spearmanr(x, y).statistic)


def weighted_bootstrap_spearman(
    df: pd.DataFrame,
    value_col: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, int, int]:
    required = {"subject_a", "subject_b", "gt_distance", value_col}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"missing columns for {value_col}: {', '.join(missing)}")

    working = df.loc[:, ["subject_a", "subject_b", "gt_distance", value_col]].copy()
    working["gt_distance"] = pd.to_numeric(working["gt_distance"], errors="coerce")
    working[value_col] = pd.to_numeric(working[value_col], errors="coerce")
    working = working[np.isfinite(working["gt_distance"]) & np.isfinite(working[value_col])]
    working = working[working["subject_a"].astype(str) != working["subject_b"].astype(str)]
    if working.empty:
        return math.nan, math.nan, math.nan, 0, 0

    subjects = np.array(sorted(set(working["subject_a"].astype(str)) | set(working["subject_b"].astype(str))))
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
    subj_a = working["subject_a"].astype(str).map(subject_to_idx).to_numpy(dtype=np.int32)
    subj_b = working["subject_b"].astype(str).map(subject_to_idx).to_numpy(dtype=np.int32)
    gt = working["gt_distance"].to_numpy(dtype=np.float64)
    values = working[value_col].to_numpy(dtype=np.float64)

    point = finite_spearman(gt, values)
    boot_values: list[float] = []

    for _ in range(n_bootstrap):
        sampled = rng.integers(0, len(subjects), size=len(subjects))
        counts = np.bincount(sampled, minlength=len(subjects)).astype(np.int16, copy=False)
        weights = counts[subj_a].astype(np.int32) * counts[subj_b].astype(np.int32)
        keep = weights > 0
        if int(keep.sum()) < 3:
            continue
        x = np.repeat(gt[keep], weights[keep])
        y = np.repeat(values[keep], weights[keep])
        corr = finite_spearman(x, y)
        if math.isfinite(corr):
            boot_values.append(corr)

    if not boot_values:
        return point, math.nan, math.nan, len(subjects), len(working)

    ci_low, ci_high = np.percentile(np.asarray(boot_values, dtype=np.float64), [2.5, 97.5])
    return point, float(ci_low), float(ci_high), len(subjects), len(working)


def remesh_tasks(args: argparse.Namespace) -> list[BootstrapTask]:
    methods = tuple(METHODS)
    return [
        BootstrapTask(
            dataset="REMESH",
            setting="Original-to-original",
            csv_path=find_pair_csv(args.remesh_original_dir, "original__to__original"),
            filter_name="topology_a == original and topology_b == original",
            filter_fn=lambda df: (df["topology_a"].astype(str) == "original")
            & (df["topology_b"].astype(str) == "original"),
            methods=methods,
            table_setting="Original-to-original",
        ),
        BootstrapTask(
            dataset="REMESH",
            setting="No-crop cross-topology",
            csv_path=find_pair_csv(args.remesh_cross_topology_dir),
            filter_name="topology_a/topology_b are not crop and differ",
            filter_fn=lambda df: (df["topology_a"].astype(str) != "crop")
            & (df["topology_b"].astype(str) != "crop")
            & (df["topology_a"].astype(str) != df["topology_b"].astype(str)),
            methods=methods,
            table_setting="No-crop cross-topology",
        ),
    ]


def discover_faceverse_tasks() -> list[BootstrapTask]:
    """Return FaceVerse tasks only when pair-level rows are available.

    The currently common FaceVerse artifacts in this repository are summary-only
    CSVs. Those are useful for point estimates, but not enough for subject-level
    bootstrap because they do not retain per-subject-pair distances.
    """
    roots = [
        REPO_ROOT / "datasets" / "FaceVerse" / "faceverse_ranking_vs_gt_neutral_full_mixed_xtopo_9a81466d_best_by_xtopo_mesh_clean_noicp",
        REPO_ROOT / "datasets" / "FaceVerse" / "faceverse_xtopo10k_ranking_vs_gt_mixed_xtopo_9a81466d_best_by_xtopo_mesh_clean_postperturb_icp",
        REPO_ROOT / "datasets" / "FaceVerse" / "FINE_tuning",
    ]
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(root.rglob("all_pairs.csv"))
            candidates.extend(root.rglob("pair_metrics.csv"))
    if not candidates:
        warn(
            "No FaceVerse pair-level all_pairs.csv/pair_metrics.csv files found; "
            "skipping optional FaceVerse bootstrap CIs."
        )
    else:
        warn(
            "Found FaceVerse pair-level-looking files, but their protocol labels are not "
            "mapped in this script yet; skipping optional FaceVerse CIs."
        )
    return []


def compute_task(task: BootstrapTask, n_bootstrap: int, seed: int) -> list[dict[str, object]]:
    df = read_pairs(task.csv_path)
    if df is None:
        return []

    required_base = {"subject_a", "subject_b", "gt_distance"}
    missing_base = sorted(required_base.difference(df.columns))
    if missing_base:
        warn(f"{task.csv_path} is missing required columns: {', '.join(missing_base)}")
        return []

    try:
        mask = task.filter_fn(df)
    except Exception as exc:  # noqa: BLE001
        warn(f"Could not apply filter '{task.filter_name}' to {task.csv_path}: {exc}")
        return []

    filtered = df.loc[mask].copy()
    if "status" in filtered.columns:
        filtered = filtered[filtered["status"].astype(str).str.lower().eq("ok")]
    if filtered.empty:
        warn(f"No rows left for {task.dataset} / {task.setting} after filter: {task.filter_name}")
        return []

    rows: list[dict[str, object]] = []
    for method_idx, (method_name, value_col) in enumerate(task.methods):
        if value_col not in filtered.columns:
            warn(f"Skipping {task.dataset} / {task.setting} / {method_name}: missing column {value_col}")
            continue
        method_rng = np.random.default_rng(seed + 1009 * method_idx + 9176 * len(rows))
        try:
            spearman, ci_low, ci_high, n_subjects, n_pairs = weighted_bootstrap_spearman(
                filtered,
                value_col=value_col,
                n_bootstrap=n_bootstrap,
                rng=method_rng,
            )
        except ValueError as exc:
            warn(f"Skipping {task.dataset} / {task.setting} / {method_name}: {exc}")
            continue
        rows.append(
            {
                "dataset": task.dataset,
                "setting": task.setting,
                "method": method_name,
                "spearman": spearman,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_subjects": n_subjects,
                "n_pairs": n_pairs,
                "n_bootstrap": n_bootstrap,
            }
        )
    return rows


def compute_topology_table_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    same_df = read_pairs(find_pair_csv(args.remesh_same_topology_dir))
    if same_df is None:
        return []

    rows: list[dict[str, object]] = []
    methods = (("Chamfer", "raw_chamfer"), ("Latent distance", "latent_distance"))
    task_index = 0
    for source in TOPOLOGIES:
        for target in TOPOLOGIES:
            if source == target:
                source_df = same_df
            else:
                source_df = read_pairs(find_pair_csv(args.remesh_table1_cross_dir, f"{source}__to__{target}"))
                if source_df is None:
                    warn(
                        f"Missing exact REMESH Table 1 pair-level rows for {source} -> {target}; "
                        "skipping this cell."
                    )
                    continue
            needed = {"topology_a", "topology_b"}
            missing = sorted(needed.difference(source_df.columns))
            if missing:
                warn(f"Cannot compute {source} -> {target}: missing columns {', '.join(missing)}")
                continue
            subset = source_df[
                source_df["topology_a"].astype(str).eq(source)
                & source_df["topology_b"].astype(str).eq(target)
            ].copy()
            if "status" in subset.columns:
                subset = subset[subset["status"].astype(str).str.lower().eq("ok")]
            if subset.empty:
                warn(f"No pair rows found for REMESH Table 1 cell {source} -> {target}")
                continue

            for method_idx, (method_name, value_col) in enumerate(methods):
                if value_col not in subset.columns:
                    warn(f"Skipping REMESH Table 1 {source} -> {target} / {method_name}: missing {value_col}")
                    continue
                rng = np.random.default_rng(args.seed + 104729 + 1543 * task_index + 37 * method_idx)
                try:
                    spearman, ci_low, ci_high, n_subjects, n_pairs = weighted_bootstrap_spearman(
                        subset,
                        value_col=value_col,
                        n_bootstrap=args.n_bootstrap,
                        rng=rng,
                    )
                except ValueError as exc:
                    warn(f"Skipping REMESH Table 1 {source} -> {target} / {method_name}: {exc}")
                    continue
                rows.append(
                    {
                        "dataset": "REMESH",
                        "setting": f"{source} -> {target}",
                        "method": method_name,
                        "spearman": spearman,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "n_subjects": n_subjects,
                        "n_pairs": n_pairs,
                        "n_bootstrap": args.n_bootstrap,
                    }
                )
            task_index += 1
    return rows


def fmt_interval(row: pd.Series | None) -> str:
    if row is None:
        return "--"
    vals = [row["spearman"], row["ci_low"], row["ci_high"]]
    if any(not math.isfinite(float(v)) for v in vals):
        return "--"
    return f"{float(vals[0]):.3f} [{float(vals[1]):.3f}, {float(vals[2]):.3f}]"


def make_latex_table(results: pd.DataFrame) -> str:
    remesh = results[results["dataset"].eq("REMESH")]
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & Original-to-original & No-crop cross-topology \\",
        r"\midrule",
    ]
    for method, _ in METHODS:
        row_orig = remesh[(remesh["setting"].eq("Original-to-original")) & (remesh["method"].eq(method))]
        row_x = remesh[(remesh["setting"].eq("No-crop cross-topology")) & (remesh["method"].eq(method))]
        orig = fmt_interval(row_orig.iloc[0] if len(row_orig) else None)
        cross = fmt_interval(row_x.iloc[0] if len(row_x) else None)
        lines.append(f"{method} & {orig} & {cross} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def make_topology_latex_table(results: pd.DataFrame) -> str:
    def lookup(method: str, source: str, target: str) -> str:
        setting = f"{source} -> {target}"
        row = results[
            results["dataset"].eq("REMESH")
            & results["setting"].eq(setting)
            & results["method"].eq(method)
        ]
        return fmt_interval(row.iloc[0] if len(row) else None)

    lines: list[str] = []
    for title, method in [("Raw Chamfer", "Chamfer"), ("Latent distance", "Latent distance")]:
        lines.extend(
            [
                r"\begin{tabular}{l" + "c" * len(TOPOLOGIES) + "}",
                r"\toprule",
                "Source & " + " & ".join(TOPOLOGIES) + r" \\",
                r"\midrule",
            ]
        )
        lines.append(r"\multicolumn{" + str(len(TOPOLOGIES) + 1) + r"}{c}{" + title + r"} \\")
        for source in TOPOLOGIES:
            cells = [lookup(method, source, target) for target in TOPOLOGIES]
            lines.append(source + " & " + " & ".join(cells) + r" \\")
        lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines).rstrip() + "\n"


def write_readme(out_dir: Path, csv_path: Path, tex_path: Path, topology_tex_path: Path) -> None:
    text = f"""# Bootstrap Confidence Intervals

This folder contains subject-level bootstrap confidence intervals for stored
ranking correlations. The script samples held-out subjects with replacement,
rebuilds the induced cross-subject pair set with multiplicity, and recomputes
Spearman correlation between each stored method distance and `D_GT`.

Subject-level bootstrap is used instead of pair-level bootstrap because pair
distances are not independent: each subject appears in many pairwise
comparisons. Resampling individual pairs would underestimate uncertainty by
treating those dependent observations as independent samples.

The confidence intervals capture uncertainty from the finite held-out subject
set under the fixed trained checkpoint and fixed evaluation protocol. They do
not capture training-run variability, split variability, hyperparameter
selection uncertainty, architecture choices, or uncertainty from recomputing
mesh distances.

Generated files:

- `{csv_path.name}`: numeric results with point estimates and percentile 95%
  confidence intervals.
- `{tex_path.name}`: LaTeX table snippet for the main REMESH alignment table.
- `{topology_tex_path.name}`: LaTeX table snippet for the REMESH ordered
  topology-pair table.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.n_bootstrap <= 0:
        raise SystemExit("--n-bootstrap must be positive")

    tasks = remesh_tasks(args)
    if args.include_faceverse:
        tasks.extend(discover_faceverse_tasks())

    all_rows: list[dict[str, object]] = []
    for task_idx, task in enumerate(tasks):
        print(f"Loading {task.dataset} / {task.setting}: {task.csv_path}")
        all_rows.extend(compute_task(task, n_bootstrap=args.n_bootstrap, seed=args.seed + 7919 * task_idx))
    print("Computing REMESH ordered topology-pair bootstrap table")
    all_rows.extend(compute_topology_table_rows(args))

    if not all_rows:
        raise SystemExit("No bootstrap results were produced; see warnings above.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(all_rows)
    csv_path = args.out_dir / "bootstrap_ci.csv"
    tex_path = args.out_dir / "alignment_effect_bootstrap_table.tex"
    topology_tex_path = args.out_dir / "cross_topology_bootstrap_table.tex"
    readme_path = args.out_dir / "README.md"

    ordered_cols = [
        "dataset",
        "setting",
        "method",
        "spearman",
        "ci_low",
        "ci_high",
        "n_subjects",
        "n_pairs",
        "n_bootstrap",
    ]
    results.to_csv(csv_path, index=False, columns=ordered_cols, quoting=csv.QUOTE_MINIMAL)
    latex = make_latex_table(results)
    topology_latex = make_topology_latex_table(results)
    tex_path.write_text(latex + "\n", encoding="utf-8")
    topology_tex_path.write_text(topology_latex, encoding="utf-8")
    write_readme(args.out_dir, csv_path=csv_path, tex_path=tex_path, topology_tex_path=topology_tex_path)

    print("\nGenerated LaTeX table:\n")
    print(latex)
    print(f"\nCSV: {csv_path}")
    print(f"LaTeX: {tex_path}")
    print(f"Topology LaTeX: {topology_tex_path}")
    print(f"README: {readme_path}")


if __name__ == "__main__":
    main()
