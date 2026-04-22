#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PERTURBATED_DIR = REPO_ROOT / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic" / "perturbated"
if str(PERTURBATED_DIR) not in sys.path:
    sys.path.insert(0, str(PERTURBATED_DIR))

import compare_model_vs_chamfer_rankings as base  # noqa: E402
import registration_utils as reg_utils  # noqa: E402


DEFAULT_OUT_DIR = REPO_ROOT / "checking_assumptions" / "outputs" / "rotation_icp_triplet_visuals"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render clean vs high-rotation vs rigid-ICP triplet overlays.")
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--data_dir", type=str, default=str(REPO_ROOT / "datasets" / "REMESH" / "npz_data_topo_500_withops"))
    p.add_argument("--sample_name_a", type=str, default="id0084_GTready_down8k")
    p.add_argument("--sample_name_b", type=str, default="id0478_GTready_up60k")
    p.add_argument("--rotation_sigma", type=float, default=0.20)
    p.add_argument("--base_seed", type=int, default=1234)
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
        jitter_sigma=0.0,
        translation_sigma=0.0,
        rotation_sigma=float(args.rotation_sigma),
        mixed_jitter_sigma=0.0,
        mixed_translation_sigma=0.0,
        mixed_rotation_sigma=float(args.rotation_sigma),
        rigid_rot_deg=float(args.rigid_rot_deg),
        rigid_trans_scale=float(args.rigid_trans_scale),
        rigid_rot_deg_min=float(args.rigid_rot_deg_min),
        rigid_trans_scale_min=float(args.rigid_trans_scale_min),
    )
    return base.PerturbationParams.from_namespace(ns)


def _scenario_spec(args: argparse.Namespace) -> tuple[base.ScenarioSpec, int]:
    scenarios = base._parse_scenarios(
        "rotation",
        base_sigma=float(args.rotation_sigma),
        cli_args=SimpleNamespace(
            jitter_sigma=-1.0,
            translation_sigma=-1.0,
            rotation_sigma=float(args.rotation_sigma),
            mixed_jitter_sigma=-1.0,
            mixed_translation_sigma=-1.0,
            mixed_rotation_sigma=float(args.rotation_sigma),
        ),
    )
    if len(scenarios) != 1:
        raise RuntimeError(f"Expected exactly one scenario, got {scenarios}")
    scenario = scenarios[0]
    scenario_index = 3
    return scenario, scenario_index


def _sample_for_plot(points: np.ndarray, n_points: int, seed: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] <= int(n_points):
        return pts
    idx = reg_utils.build_sample_vertex_indices(vertex_count=pts.shape[0], n_points=int(n_points), seed=int(seed))
    return pts[idx]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = base.GTReadyDataset(args.data_dir)
    name_to_idx = {Path(path).stem: idx for idx, path in enumerate(dataset.files)}
    idx_a = int(name_to_idx[str(args.sample_name_a)])
    idx_b = int(name_to_idx[str(args.sample_name_b)])

    sample_a = dataset[idx_a]
    sample_b = dataset[idx_b]
    V_a = sample_a["verts"].clone()
    V_b = sample_b["verts"].clone()

    params = _make_params(args)
    scenario, scenario_index = _scenario_spec(args)

    seed_a = base._scenario_seed(base_seed=int(args.base_seed), scenario_index=int(scenario_index), dataset_idx=idx_a)
    seed_b = base._scenario_seed(base_seed=int(args.base_seed), scenario_index=int(scenario_index), dataset_idx=idx_b)

    clean_spec = base.ScenarioSpec(name="clean", jitter_sigma=0.0, rotation_sigma=0.0, translation_sigma=0.0)
    V_a_clean = base._apply_scenario(V=V_a, params=params, scenario=clean_spec, seed=seed_a).contiguous()
    V_b_clean = base._apply_scenario(V=V_b, params=params, scenario=clean_spec, seed=seed_b).contiguous()
    V_a_rot = base._apply_scenario(V=V_a, params=params, scenario=scenario, seed=seed_a).contiguous()
    V_b_rot = base._apply_scenario(V=V_b, params=params, scenario=scenario, seed=seed_b).contiguous()

    icp_idx_a = reg_utils.build_sample_vertex_indices(vertex_count=int(V_a_rot.shape[0]), n_points=int(args.icp_points), seed=idx_a * 97 + 11)
    icp_idx_b = reg_utils.build_sample_vertex_indices(vertex_count=int(V_b_rot.shape[0]), n_points=int(args.icp_points), seed=idx_b * 97 + 11)
    src_icp = reg_utils.extract_point_subset(V_a_rot, icp_idx_a)
    tgt_icp = reg_utils.extract_point_subset(V_b_rot, icp_idx_b)
    transform = reg_utils._estimate_rigid_icp_transform(
        source_points=src_icp,
        target_points=tgt_icp,
        max_correspondence_distance=float(args.icp_max_correspondence_distance),
        max_iteration=int(args.icp_max_iteration),
    )
    V_a_aligned = reg_utils.apply_rigid_transform(V_a_rot, transform)

    clean_src = _sample_for_plot(V_a_clean.detach().cpu().numpy(), n_points=int(args.plot_points), seed=idx_a * 17 + 1)
    clean_tgt = _sample_for_plot(V_b_clean.detach().cpu().numpy(), n_points=int(args.plot_points), seed=idx_b * 17 + 2)
    rot_src = _sample_for_plot(V_a_rot.detach().cpu().numpy(), n_points=int(args.plot_points), seed=idx_a * 17 + 3)
    rot_tgt = _sample_for_plot(V_b_rot.detach().cpu().numpy(), n_points=int(args.plot_points), seed=idx_b * 17 + 4)
    aligned_src = _sample_for_plot(V_a_aligned.detach().cpu().numpy(), n_points=int(args.plot_points), seed=idx_a * 17 + 5)

    title = (
        f"rotation sigma={float(args.rotation_sigma):.2f} | {args.sample_name_a} -> {args.sample_name_b}\n"
        f"clean vs target, rotated vs target, post-ICP vs target"
    )
    out_path = out_dir / f"rotation_sigma{float(args.rotation_sigma):.2f}_{args.sample_name_a}__to__{args.sample_name_b}.png"
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 11.5))
    panels = [
        (axes[0, 0], clean_src[:, 0], clean_src[:, 1], clean_tgt[:, 0], clean_tgt[:, 1], "Clean front (x,y)"),
        (axes[0, 1], clean_src[:, 2], clean_src[:, 1], clean_tgt[:, 2], clean_tgt[:, 1], "Clean depth (z,y)"),
        (axes[1, 0], rot_src[:, 0], rot_src[:, 1], rot_tgt[:, 0], rot_tgt[:, 1], "Rotation front (x,y)"),
        (axes[1, 1], rot_src[:, 2], rot_src[:, 1], rot_tgt[:, 2], rot_tgt[:, 1], "Rotation depth (z,y)"),
        (axes[2, 0], aligned_src[:, 0], aligned_src[:, 1], rot_tgt[:, 0], rot_tgt[:, 1], "Post-ICP front (x,y)"),
        (axes[2, 1], aligned_src[:, 2], aligned_src[:, 1], rot_tgt[:, 2], rot_tgt[:, 1], "Post-ICP depth (z,y)"),
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

    manifest = {
        "sample_name_a": str(args.sample_name_a),
        "sample_name_b": str(args.sample_name_b),
        "scenario": "rotation",
        "rotation_sigma": float(args.rotation_sigma),
        "clean_vs_target_note": "source and target both clean (no perturbation)",
        "rotated_vs_target_note": "source and target both rotated independently with the same scenario seed policy used by the benchmark",
        "output_png": str(out_path),
        "icp_points": int(args.icp_points),
        "icp_max_correspondence_distance": float(args.icp_max_correspondence_distance),
        "icp_max_iteration": int(args.icp_max_iteration),
    }
    (out_dir / f"rotation_sigma{float(args.rotation_sigma):.2f}_{args.sample_name_a}__to__{args.sample_name_b}_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
