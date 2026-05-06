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


DEFAULT_OUT_DIR = REPO_ROOT / "checking_assumptions" / "outputs" / "rigid_mixed_triplet_visuals"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render clean vs rigid mixed (rotation+translation) vs rigid-ICP triplet overlays.")
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--data_dir", type=str, default=str(REPO_ROOT / "datasets" / "REMESH" / "npz_data_topo_500_withops"))
    p.add_argument("--sample_name_a", type=str, default="id0084_GTready_down8k")
    p.add_argument("--sample_name_b", type=str, default="id0478_GTready_up60k")
    p.add_argument("--rigid_sigma", type=float, default=0.20)
    p.add_argument("--base_seed", type=int, default=1234)
    p.add_argument("--rigid_rot_deg", type=float, default=20.0)
    p.add_argument("--rigid_trans_scale", type=float, default=0.05)
    p.add_argument("--rigid_rot_deg_min", type=float, default=1.0)
    p.add_argument("--rigid_trans_scale_min", type=float, default=0.002)
    p.add_argument("--icp_points", type=int, default=128)
    p.add_argument("--icp_max_correspondence_distance", type=float, default=0.05)
    p.add_argument("--icp_max_iteration", type=int, default=20)
    p.add_argument("--plot_points", type=int, default=5000)
    p.add_argument("--rotate_180", action="store_true", help="Rotate each 2D panel by 180 degrees")
    p.add_argument(
        "--paper_identity_layout",
        action="store_true",
        help="Render a 2x3 paper layout without the rigid-mixed row and with separated identity/topology panels.",
    )
    return p.parse_args()


def _make_params(args: argparse.Namespace) -> base.PerturbationParams:
    ns = SimpleNamespace(
        outlier_frac=0.0,
        outlier_scale=0.0,
        jitter_sigma=0.0,
        translation_sigma=float(args.rigid_sigma),
        rotation_sigma=float(args.rigid_sigma),
        mixed_jitter_sigma=0.0,
        mixed_translation_sigma=float(args.rigid_sigma),
        mixed_rotation_sigma=float(args.rigid_sigma),
        rigid_rot_deg=float(args.rigid_rot_deg),
        rigid_trans_scale=float(args.rigid_trans_scale),
        rigid_rot_deg_min=float(args.rigid_rot_deg_min),
        rigid_trans_scale_min=float(args.rigid_trans_scale_min),
    )
    return base.PerturbationParams.from_namespace(ns)


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

    seed_a = base._scenario_seed(base_seed=int(args.base_seed), scenario_index=4, dataset_idx=idx_a)
    seed_b = base._scenario_seed(base_seed=int(args.base_seed), scenario_index=4, dataset_idx=idx_b)

    clean_spec = base.ScenarioSpec(name="clean", jitter_sigma=0.0, rotation_sigma=0.0, translation_sigma=0.0)
    rigid_spec = base.ScenarioSpec(name="mixed", jitter_sigma=0.0, rotation_sigma=float(args.rigid_sigma), translation_sigma=float(args.rigid_sigma))

    V_a_clean = base._apply_scenario(V=V_a, params=params, scenario=clean_spec, seed=seed_a).contiguous()
    V_b_clean = base._apply_scenario(V=V_b, params=params, scenario=clean_spec, seed=seed_b).contiguous()
    V_a_rigid = base._apply_scenario(V=V_a, params=params, scenario=rigid_spec, seed=seed_a).contiguous()
    V_b_rigid = base._apply_scenario(V=V_b, params=params, scenario=rigid_spec, seed=seed_b).contiguous()

    icp_idx_a = reg_utils.build_sample_vertex_indices(vertex_count=int(V_a_rigid.shape[0]), n_points=int(args.icp_points), seed=idx_a * 97 + 11)
    icp_idx_b = reg_utils.build_sample_vertex_indices(vertex_count=int(V_b_rigid.shape[0]), n_points=int(args.icp_points), seed=idx_b * 97 + 11)
    src_icp = reg_utils.extract_point_subset(V_a_rigid, icp_idx_a)
    tgt_icp = reg_utils.extract_point_subset(V_b_rigid, icp_idx_b)
    transform = reg_utils._estimate_rigid_icp_transform(
        source_points=src_icp,
        target_points=tgt_icp,
        max_correspondence_distance=float(args.icp_max_correspondence_distance),
        max_iteration=int(args.icp_max_iteration),
    )
    V_a_aligned = reg_utils.apply_rigid_transform(V_a_rigid, transform)

    clean_src = _sample_for_plot(V_a_clean.detach().cpu().numpy(), n_points=int(args.plot_points), seed=idx_a * 17 + 1)
    clean_tgt = _sample_for_plot(V_b_clean.detach().cpu().numpy(), n_points=int(args.plot_points), seed=idx_b * 17 + 2)
    rigid_src = _sample_for_plot(V_a_rigid.detach().cpu().numpy(), n_points=int(args.plot_points), seed=idx_a * 17 + 3)
    rigid_tgt = _sample_for_plot(V_b_rigid.detach().cpu().numpy(), n_points=int(args.plot_points), seed=idx_b * 17 + 4)
    aligned_src = _sample_for_plot(V_a_aligned.detach().cpu().numpy(), n_points=int(args.plot_points), seed=idx_a * 17 + 5)

    title = (
        f"rigid mixed sigma={float(args.rigid_sigma):.2f} | {args.sample_name_a} -> {args.sample_name_b}\n"
        f"clean vs target, rigid-mixed vs target, post-ICP vs target"
    )
    out_path = out_dir / f"rigid_mixed_sigma{float(args.rigid_sigma):.2f}_{args.sample_name_a}__to__{args.sample_name_b}.png"

    def maybe_rotate(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if args.rotate_180:
            return -x, -y
        return x, y

    if args.paper_identity_layout:
        out_path = out_dir / (
            f"rigid_mixed_sigma{float(args.rigid_sigma):.2f}_{args.sample_name_a}"
            f"__to__{args.sample_name_b}_paper_identity.png"
        )
        fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.0))
        panels = [
            (axes[0, 0], clean_src[:, 0], clean_src[:, 1], clean_tgt[:, 0], clean_tgt[:, 1], "Clean front (x,y)"),
            (axes[0, 1], clean_src[:, 2], clean_src[:, 1], clean_tgt[:, 2], clean_tgt[:, 1], "Clean depth (z,y)"),
            (axes[1, 0], aligned_src[:, 0], aligned_src[:, 1], rigid_tgt[:, 0], rigid_tgt[:, 1], "Post-ICP front (x,y)"),
            (axes[1, 1], aligned_src[:, 2], aligned_src[:, 1], rigid_tgt[:, 2], rigid_tgt[:, 1], "Post-ICP depth (z,y)"),
        ]
        for ax, src_x, src_y, tgt_x, tgt_y, panel_title in panels:
            src_x, src_y = maybe_rotate(src_x, src_y)
            tgt_x, tgt_y = maybe_rotate(tgt_x, tgt_y)
            ax.scatter(tgt_x, tgt_y, s=1.2, alpha=0.28, color="#ff7f0e", label="target")
            ax.scatter(src_x, src_y, s=1.2, alpha=0.28, color="#1f77b4", label="source")
            ax.set_title(panel_title)
            ax.set_aspect("equal", adjustable="box")

        sep_specs = [
            (axes[0, 2], clean_src[:, 0], clean_src[:, 1], clean_tgt[:, 0], clean_tgt[:, 1], "Separated front"),
            (axes[1, 2], clean_src[:, 2], clean_src[:, 1], clean_tgt[:, 2], clean_tgt[:, 1], "Separated depth"),
        ]
        for ax, src_x, src_y, tgt_x, tgt_y, panel_title in sep_specs:
            src_x, src_y = maybe_rotate(src_x, src_y)
            tgt_x, tgt_y = maybe_rotate(tgt_x, tgt_y)
            all_x = np.concatenate([src_x, tgt_x])
            gap = 0.25 * max(float(np.ptp(all_x)), 1e-6)
            src_shift = -0.5 * (float(np.ptp(src_x)) + gap)
            tgt_shift = 0.5 * (float(np.ptp(tgt_x)) + gap)
            ax.scatter(src_x + src_shift, src_y, s=1.2, alpha=0.34, color="#1f77b4", label=args.sample_name_a)
            ax.scatter(tgt_x + tgt_shift, tgt_y, s=1.2, alpha=0.34, color="#ff7f0e", label=args.sample_name_b)
            ax.set_title(panel_title)
            ax.set_aspect("equal", adjustable="box")
            ax.legend(frameon=False, markerscale=4, fontsize=8, loc="lower center")

        axes[0, 0].legend(frameon=False, markerscale=4)
        fig.suptitle(
            f"rigid mixed sigma={float(args.rigid_sigma):.2f} | {args.sample_name_a} -> {args.sample_name_b}"
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)

        manifest = {
            "sample_name_a": str(args.sample_name_a),
            "sample_name_b": str(args.sample_name_b),
            "scenario": "rigid_mixed",
            "layout": "paper_identity",
            "rigid_sigma": float(args.rigid_sigma),
            "output_png": str(out_path),
            "icp_points": int(args.icp_points),
            "icp_max_correspondence_distance": float(args.icp_max_correspondence_distance),
            "icp_max_iteration": int(args.icp_max_iteration),
        }
        (out_dir / f"rigid_mixed_sigma{float(args.rigid_sigma):.2f}_{args.sample_name_a}__to__{args.sample_name_b}_paper_identity_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        return

    fig, axes = plt.subplots(3, 2, figsize=(11.5, 11.5))
    panels = [
        (axes[0, 0], clean_src[:, 0], clean_src[:, 1], clean_tgt[:, 0], clean_tgt[:, 1], "Clean front (x,y)"),
        (axes[0, 1], clean_src[:, 2], clean_src[:, 1], clean_tgt[:, 2], clean_tgt[:, 1], "Clean depth (z,y)"),
        (axes[1, 0], rigid_src[:, 0], rigid_src[:, 1], rigid_tgt[:, 0], rigid_tgt[:, 1], "Rigid mixed front (x,y)"),
        (axes[1, 1], rigid_src[:, 2], rigid_src[:, 1], rigid_tgt[:, 2], rigid_tgt[:, 1], "Rigid mixed depth (z,y)"),
        (axes[2, 0], aligned_src[:, 0], aligned_src[:, 1], rigid_tgt[:, 0], rigid_tgt[:, 1], "Post-ICP front (x,y)"),
        (axes[2, 1], aligned_src[:, 2], aligned_src[:, 1], rigid_tgt[:, 2], rigid_tgt[:, 1], "Post-ICP depth (z,y)"),
    ]
    for ax, src_x, src_y, tgt_x, tgt_y, panel_title in panels:
        src_x, src_y = maybe_rotate(src_x, src_y)
        tgt_x, tgt_y = maybe_rotate(tgt_x, tgt_y)
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
        "scenario": "rigid_mixed",
        "rigid_sigma": float(args.rigid_sigma),
        "output_png": str(out_path),
        "icp_points": int(args.icp_points),
        "icp_max_correspondence_distance": float(args.icp_max_correspondence_distance),
        "icp_max_iteration": int(args.icp_max_iteration),
    }
    (out_dir / f"rigid_mixed_sigma{float(args.rigid_sigma):.2f}_{args.sample_name_a}__to__{args.sample_name_b}_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
