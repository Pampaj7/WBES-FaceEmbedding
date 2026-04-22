#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PERTURBATED_DIR = REPO_ROOT / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic" / "perturbated"
if str(PERTURBATED_DIR) not in sys.path:
    sys.path.insert(0, str(PERTURBATED_DIR))

import compare_model_vs_chamfer_rankings as base  # noqa: E402
import registration_utils as reg_utils  # noqa: E402


DEFAULT_PROBE_DIR = REPO_ROOT / "checking_assumptions" / "outputs" / "same_vs_diff_gap_probe_all12_nicp"
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "checking_assumptions" / "outputs" / "probe_analysis_original_gt_all12_nicp"
DEFAULT_OUT_DIR = REPO_ROOT / "checking_assumptions" / "outputs" / "rigid_icp_example_visuals"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render rigid-ICP before/after overlays for top rank-inversion examples.")
    p.add_argument("--probe_dir", type=str, default=str(DEFAULT_PROBE_DIR))
    p.add_argument("--analysis_dir", type=str, default=str(DEFAULT_ANALYSIS_DIR))
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--data_dir", type=str, default=str(REPO_ROOT / "datasets" / "REMESH" / "npz_data_topo_500_withops"))
    p.add_argument("--scenario", type=str, default="clean")
    p.add_argument("--metric", type=str, default="rigid_registered_chamfer")
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--base_seed", type=int, default=1234)
    p.add_argument("--translation_sigma", type=float, default=0.05)
    p.add_argument("--rigid_rot_deg", type=float, default=20.0)
    p.add_argument("--rigid_trans_scale", type=float, default=0.05)
    p.add_argument("--rigid_rot_deg_min", type=float, default=1.0)
    p.add_argument("--rigid_trans_scale_min", type=float, default=0.002)
    p.add_argument("--icp_points", type=int, default=128)
    p.add_argument("--icp_max_correspondence_distance", type=float, default=0.05)
    p.add_argument("--icp_max_iteration", type=int, default=20)
    p.add_argument("--plot_points", type=int, default=5000)
    return p.parse_args()


def _make_params(args: argparse.Namespace) -> base.PerturbationParams:
    ns = SimpleNamespace(
        outlier_frac=0.0,
        outlier_scale=0.0,
        jitter_sigma=0.05,
        translation_sigma=float(args.translation_sigma),
        rotation_sigma=0.05,
        mixed_jitter_sigma=0.05,
        mixed_translation_sigma=0.05,
        mixed_rotation_sigma=0.05,
        rigid_rot_deg=float(args.rigid_rot_deg),
        rigid_trans_scale=float(args.rigid_trans_scale),
        rigid_rot_deg_min=float(args.rigid_rot_deg_min),
        rigid_trans_scale_min=float(args.rigid_trans_scale_min),
    )
    return base.PerturbationParams.from_namespace(ns)


def _scenario_spec(args: argparse.Namespace) -> tuple[base.ScenarioSpec, int]:
    scenario_name = str(args.scenario)
    scenarios = base._parse_scenarios(
        scenario_name,
        base_sigma=0.05,
        cli_args=SimpleNamespace(
            jitter_sigma=-1.0,
            translation_sigma=float(args.translation_sigma),
            rotation_sigma=-1.0,
            mixed_jitter_sigma=-1.0,
            mixed_translation_sigma=-1.0,
            mixed_rotation_sigma=-1.0,
        ),
    )
    if len(scenarios) != 1:
        raise RuntimeError(f"Expected exactly one scenario, got {scenarios}")
    scenario = scenarios[0]
    scenario_index = {
        "clean": 0,
        "jitter": 1,
        "translation": 2,
        "rotation": 3,
        "mixed": 4,
    }[scenario.name]
    return scenario, scenario_index


def _sample_for_plot(points: np.ndarray, n_points: int, seed: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] <= int(n_points):
        return pts
    idx = reg_utils.build_sample_vertex_indices(vertex_count=pts.shape[0], n_points=int(n_points), seed=int(seed))
    return pts[idx]


def _plot_overlay(raw_src: np.ndarray, aligned_src: np.ndarray, tgt: np.ndarray, title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.5))
    panels = [
        (axes[0, 0], raw_src[:, 0], raw_src[:, 1], tgt[:, 0], tgt[:, 1], "Raw front (x,y)"),
        (axes[0, 1], aligned_src[:, 0], aligned_src[:, 1], tgt[:, 0], tgt[:, 1], "Rigid ICP front (x,y)"),
        (axes[1, 0], raw_src[:, 2], raw_src[:, 1], tgt[:, 2], tgt[:, 1], "Raw depth (z,y)"),
        (axes[1, 1], aligned_src[:, 2], aligned_src[:, 1], tgt[:, 2], tgt[:, 1], "Rigid ICP depth (z,y)"),
    ]
    for ax, src_x, src_y, tgt_x, tgt_y, panel_title in panels:
        ax.scatter(tgt_x, tgt_y, s=1.2, alpha=0.28, color="#ff7f0e", label="target")
        ax.scatter(src_x, src_y, s=1.2, alpha=0.28, color="#1f77b4", label="source")
        ax.set_title(panel_title)
        ax.set_aspect("equal", adjustable="box")
    axes[0, 0].legend(frameon=False, markerscale=4)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_dir = Path(args.probe_dir).expanduser().resolve()
    analysis_dir = Path(args.analysis_dir).expanduser().resolve()
    inversion_path = analysis_dir / args.scenario / "rank_inversion_examples.csv"
    inv = np.genfromtxt(inversion_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if inv.size == 0:
        raise RuntimeError(f"No inversion examples found in {inversion_path}")
    if inv.ndim == 0:
        rows = [inv]
    else:
        rows = [row for row in inv if str(row["metric"]) == str(args.metric)]
    rows = rows[: int(args.top_k)]
    if not rows:
        raise RuntimeError(f"No rows for metric={args.metric} in {inversion_path}")

    dataset = base.GTReadyDataset(args.data_dir)
    name_to_idx: Dict[str, int] = {Path(path).stem: idx for idx, path in enumerate(dataset.files)}
    params = _make_params(args)
    scenario, scenario_index = _scenario_spec(args)

    manifest: List[dict] = []
    for example_idx, row in enumerate(rows, start=1):
        sample_name_a = str(row["sample_name_a"])
        sample_name_b = str(row["sample_name_b"])
        idx_a = int(name_to_idx[sample_name_a])
        idx_b = int(name_to_idx[sample_name_b])

        sample_a = dataset[idx_a]
        sample_b = dataset[idx_b]
        V_a = sample_a["verts"].clone()
        V_b = sample_b["verts"].clone()

        seed_a = base._scenario_seed(base_seed=int(args.base_seed), scenario_index=int(scenario_index), dataset_idx=idx_a)
        seed_b = base._scenario_seed(base_seed=int(args.base_seed), scenario_index=int(scenario_index), dataset_idx=idx_b)
        V_a_pert = base._apply_scenario(V=V_a, params=params, scenario=scenario, seed=seed_a).contiguous()
        V_b_pert = base._apply_scenario(V=V_b, params=params, scenario=scenario, seed=seed_b).contiguous()

        icp_idx_a = reg_utils.build_sample_vertex_indices(vertex_count=int(V_a_pert.shape[0]), n_points=int(args.icp_points), seed=idx_a * 97 + 11)
        icp_idx_b = reg_utils.build_sample_vertex_indices(vertex_count=int(V_b_pert.shape[0]), n_points=int(args.icp_points), seed=idx_b * 97 + 11)
        src_icp = reg_utils.extract_point_subset(V_a_pert, icp_idx_a)
        tgt_icp = reg_utils.extract_point_subset(V_b_pert, icp_idx_b)
        transform = reg_utils._estimate_rigid_icp_transform(
            source_points=src_icp,
            target_points=tgt_icp,
            max_correspondence_distance=float(args.icp_max_correspondence_distance),
            max_iteration=int(args.icp_max_iteration),
        )
        V_a_aligned = reg_utils.apply_rigid_transform(V_a_pert, transform)

        raw_src = _sample_for_plot(V_a_pert.detach().cpu().numpy(), n_points=int(args.plot_points), seed=idx_a * 17 + example_idx)
        aligned_src = _sample_for_plot(V_a_aligned.detach().cpu().numpy(), n_points=int(args.plot_points), seed=idx_a * 23 + example_idx)
        tgt = _sample_for_plot(V_b_pert.detach().cpu().numpy(), n_points=int(args.plot_points), seed=idx_b * 29 + example_idx)

        title = (
            f"{args.scenario} | {sample_name_a} -> {sample_name_b}\n"
            f"GT={float(row['gt_distance']):.4f} | raw={float(row['raw_chamfer']):.4f} | rigid={float(row['rigid_registered_chamfer']):.4f}"
        )
        out_path = out_dir / f"{args.scenario}_{args.metric}_example{example_idx}.png"
        _plot_overlay(raw_src=raw_src, aligned_src=aligned_src, tgt=tgt, title=title, out_path=out_path)

        manifest.append(
            {
                "scenario": str(args.scenario),
                "metric": str(args.metric),
                "sample_name_a": sample_name_a,
                "sample_name_b": sample_name_b,
                "subject_a": str(row["subject_a"]),
                "subject_b": str(row["subject_b"]),
                "topology_a": str(row["topology_a"]),
                "topology_b": str(row["topology_b"]),
                "gt_distance": float(row["gt_distance"]),
                "raw_chamfer": float(row["raw_chamfer"]),
                "rigid_registered_chamfer": float(row["rigid_registered_chamfer"]),
                "extra_abs_rank_error_vs_raw": float(row["extra_abs_rank_error_vs_raw"]),
                "output_png": str(out_path),
            }
        )

    (out_dir / f"{args.scenario}_{args.metric}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
