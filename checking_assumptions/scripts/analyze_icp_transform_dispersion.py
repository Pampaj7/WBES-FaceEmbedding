#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PERTURBATED_DIR = REPO_ROOT / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic" / "perturbated"
if str(PERTURBATED_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PERTURBATED_DIR))

import compare_model_vs_chamfer_rankings as base  # noqa: E402
import registration_utils as reg_utils  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "checking_assumptions" / "outputs" / "icp_transform_dispersion"
DEFAULT_MODEL = (
    REPO_ROOT
    / "face_embedding"
    / "gt_encdec"
    / "remeshing"
    / "intrinsic"
    / "newdata"
    / "dn_mixed_topology_v1"
    / "mixed_xtopo_rank0p5_id0p25_bs5_best"
    / "checkpoints"
    / "best_by_xtopo_mesh_clean.pth"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract rigid ICP transform dispersion and a mean-ICP sanity check.")
    p.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL))
    p.add_argument("--checkpoint_selector", type=str, default="best_by_xtopo_mesh_clean")
    p.add_argument("--config_json", type=str, default="")
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--data_dir", type=str, default="")
    p.add_argument("--dist_npz", type=str, default="")
    p.add_argument("--subject_split", type=str, default="eval")
    p.add_argument("--eval_fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--max_subjects", type=int, default=12)
    p.add_argument("--max_meshes_per_subject_eval", type=int, default=6)
    p.add_argument("--topology_labels", type=str, default="crop,down8k,noisy,original,remesh,up60k")
    p.add_argument("--preload_workers", type=int, default=4)
    p.add_argument("--same_pair_limit", type=int, default=300)
    p.add_argument("--different_pair_multiplier", type=float, default=1.0)
    p.add_argument("--scenarios", type=str, default="clean,translation,rotation,mixed")
    p.add_argument("--base_sigma", type=float, default=-1.0)
    p.add_argument("--translation_sigma", type=float, default=0.05)
    p.add_argument("--rotation_sigma", type=float, default=0.05)
    p.add_argument("--mixed_translation_sigma", type=float, default=0.05)
    p.add_argument("--mixed_rotation_sigma", type=float, default=0.05)
    p.add_argument("--rigid_rot_deg", type=float, default=20.0)
    p.add_argument("--rigid_trans_scale", type=float, default=0.05)
    p.add_argument("--rigid_rot_deg_min", type=float, default=1.0)
    p.add_argument("--rigid_trans_scale_min", type=float, default=0.002)
    p.add_argument("--icp_points", type=int, default=128)
    p.add_argument("--registration_workers", type=int, default=8)
    p.add_argument("--icp_max_correspondence_distance", type=float, default=0.05)
    p.add_argument("--icp_max_iteration", type=int, default=20)
    p.add_argument("--chamfer_batch_pairs", type=int, default=32)
    return p.parse_args()


def _safe_angle_deg(R: np.ndarray) -> float:
    R = np.asarray(R, dtype=np.float64)
    tr = float(np.trace(R))
    c = (tr - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _rotation_to_axis_angle(R: np.ndarray) -> tuple[np.ndarray, float]:
    R = np.asarray(R, dtype=np.float64)
    angle = np.radians(_safe_angle_deg(R))
    if abs(angle) < 1.0e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.0
    denom = 2.0 * np.sin(angle)
    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    ) / denom
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-12:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        axis = axis / norm
    return axis, float(np.degrees(angle))


def _average_rotation(rotations: np.ndarray) -> np.ndarray:
    if rotations.shape[0] == 0:
        return np.eye(3, dtype=np.float64)
    M = np.mean(rotations, axis=0)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return np.asarray(R, dtype=np.float64)


def _plot_hist(values: pd.Series, title: str, xlabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.hist(values[np.isfinite(values)], bins=40, color="#1f77b4", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _select_pairs(sample_records, same_pair_limit: int, different_pair_limit: int, seed: int):
    tri_i, tri_j = np.triu_indices(len(sample_records), k=1)
    subject_ids = np.asarray([rec.subject_id for rec in sample_records], dtype=object)
    topology_labels = np.asarray([rec.topology_label for rec in sample_records], dtype=object)
    cross_topology_mask = topology_labels[tri_i] != topology_labels[tri_j]
    same_subject_mask = subject_ids[tri_i] == subject_ids[tri_j]
    same_idx = np.where(cross_topology_mask & same_subject_mask)[0]
    diff_idx = np.where(cross_topology_mask & (~same_subject_mask))[0]
    if same_idx.size == 0 or diff_idx.size == 0:
        raise RuntimeError("Could not build both same-subject and different-subject cross-topology pairs")
    rng = np.random.default_rng(int(seed))
    same_take = min(int(same_pair_limit), int(same_idx.size))
    diff_take = min(int(different_pair_limit), int(diff_idx.size))
    same_pick = np.sort(rng.choice(same_idx, size=same_take, replace=False))
    diff_pick = np.sort(rng.choice(diff_idx, size=diff_take, replace=False))
    picked = np.concatenate([same_pick, diff_pick], axis=0)
    pair_i = tri_i[picked].astype(np.int64, copy=False)
    pair_j = tri_j[picked].astype(np.int64, copy=False)
    is_same = (subject_ids[pair_i] == subject_ids[pair_j]).astype(bool, copy=False)
    return pair_i, pair_j, is_same


def _make_params(args: argparse.Namespace) -> base.PerturbationParams:
    ns = SimpleNamespace(
        outlier_frac=0.0,
        outlier_scale=0.0,
        jitter_sigma=0.0,
        translation_sigma=0.0,
        rotation_sigma=0.0,
        mixed_jitter_sigma=0.0,
        mixed_translation_sigma=0.0,
        mixed_rotation_sigma=0.0,
        rigid_rot_deg=float(args.rigid_rot_deg),
        rigid_trans_scale=float(args.rigid_trans_scale),
        rigid_rot_deg_min=float(args.rigid_rot_deg_min),
        rigid_trans_scale_min=float(args.rigid_trans_scale_min),
    )
    return base.PerturbationParams.from_namespace(ns)


def _scenario_spec(args: argparse.Namespace):
    return base._parse_scenarios(
        "clean,translation,rotation,mixed",
        base_sigma=float(args.translation_sigma if args.base_sigma < 0 else args.base_sigma),
        cli_args=SimpleNamespace(
            jitter_sigma=-1.0,
            translation_sigma=float(args.translation_sigma),
            rotation_sigma=float(args.rotation_sigma),
            mixed_jitter_sigma=-1.0,
            mixed_translation_sigma=float(args.mixed_translation_sigma),
            mixed_rotation_sigma=float(args.mixed_rotation_sigma),
        ),
    )


def _apply_mean_transform(vertices: List[torch.Tensor], pair_i: np.ndarray, mean_transform: np.ndarray) -> List[torch.Tensor]:
    R = torch.as_tensor(mean_transform[:3, :3], dtype=vertices[0].dtype, device=vertices[0].device)
    t = torch.as_tensor(mean_transform[:3, 3], dtype=vertices[0].dtype, device=vertices[0].device)
    return [V @ R.transpose(0, 1) + t for V in vertices]


def _compute_mean_icp_pair_values(
    *,
    vertex_sets: Sequence[torch.Tensor],
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    mean_transform: np.ndarray,
    batch_pairs: int,
) -> np.ndarray:
    pair_i_np = np.asarray(pair_i, dtype=np.int64)
    pair_j_np = np.asarray(pair_j, dtype=np.int64)
    n_pairs = int(pair_i_np.size)
    values = np.empty((n_pairs,), dtype=np.float64)
    shape_groups: Dict[tuple[int, int], List[int]] = {}
    for pair_idx in range(n_pairs):
        src = vertex_sets[int(pair_i_np[pair_idx])]
        tgt = vertex_sets[int(pair_j_np[pair_idx])]
        shape_groups.setdefault((int(src.shape[0]), int(tgt.shape[0])), []).append(int(pair_idx))

    R = torch.as_tensor(mean_transform[:3, :3], dtype=vertex_sets[0].dtype, device=vertex_sets[0].device)
    t = torch.as_tensor(mean_transform[:3, 3], dtype=vertex_sets[0].dtype, device=vertex_sets[0].device)
    pair_batch = max(1, int(batch_pairs))
    for shape_key, group_pair_indices in shape_groups.items():
        for start in range(0, len(group_pair_indices), pair_batch):
            stop = min(start + pair_batch, len(group_pair_indices))
            batch_pair_indices = group_pair_indices[start:stop]
            X_batch: List[torch.Tensor] = []
            Y_batch: List[torch.Tensor] = []
            for pair_idx in batch_pair_indices:
                src_idx = int(pair_i_np[pair_idx])
                tgt_idx = int(pair_j_np[pair_idx])
                X_batch.append(vertex_sets[src_idx] @ R.transpose(0, 1) + t)
                Y_batch.append(vertex_sets[tgt_idx])
            batch_vals = reg_utils.symmetric_chamfer_same_shape_batch(
                X=torch.stack(X_batch, dim=0),
                Y=torch.stack(Y_batch, dim=0),
            ).detach().cpu().numpy()
            values[np.asarray(batch_pair_indices, dtype=np.int64)] = batch_vals
    return values


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir, checkpoint_path = base._resolve_run_dir_and_checkpoint(args.model_path, selector=str(args.checkpoint_selector))
    base_args = base.merge_run_args(checkpoint_path, explicit_config_json=args.config_json)
    model_args = base._resolve_runtime_args(args, base_args)
    if not Path(str(model_args.data_dir)).expanduser().exists():
        model_args.data_dir = str(base.DEFAULT_DATA_DIR)
    if not Path(str(model_args.dist_npz)).expanduser().exists():
        model_args.dist_npz = str(base.DEFAULT_DIST_NPZ)
    params = base.PerturbationParams.from_namespace(model_args)
    base.seed_everything(int(model_args.seed))
    device = base._resolve_device(args.device)

    dataset = base.GTReadyDataset(model_args.data_dir)
    gt_matrix, gt_name_to_idx = base.load_gt_distance_matrix(model_args.dist_npz, subject_re=base.SUBJECT_RE_ANY, dtype=np.float64)
    subject_map = base.build_subject_map(dataset.files, subject_re=base.SUBJECT_RE_ANY)
    subjects = sorted([sid for sid in subject_map.keys() if sid in gt_name_to_idx])
    _, _, target_subjects = base._select_subject_subset(
        subjects=subjects,
        subject_split=str(args.subject_split),
        eval_fraction=float(args.eval_fraction),
        seed=int(model_args.seed),
        max_subjects=int(args.max_subjects),
    )
    eval_plan = base.build_eval_plan(
        subj_map=subject_map,
        eval_subjects=target_subjects,
        max_meshes_per_subject_eval=int(args.max_meshes_per_subject_eval),
        seed=int(model_args.seed),
    )
    sample_cache = base.preload_eval_samples(dataset=dataset, eval_plan=eval_plan, workers=int(args.preload_workers))
    sample_records = base.build_sample_eval_records(dataset=dataset, eval_plan=eval_plan, eval_subjects=target_subjects, sample_cache=sample_cache)
    requested_topology_labels = base._parse_topology_labels(args.topology_labels)
    if requested_topology_labels:
        sample_records = base._filter_sample_records_by_topology_labels(sample_records=sample_records, topology_labels=requested_topology_labels)

    pair_i, pair_j, is_same = _select_pairs(
        sample_records=sample_records,
        same_pair_limit=int(args.same_pair_limit),
        different_pair_limit=int(round(float(args.same_pair_limit) * float(args.different_pair_multiplier))),
        seed=int(args.seed if args.seed >= 0 else model_args.seed),
    )
    gt_distances = np.asarray(
        [
            0.0 if sample_records[int(i)].subject_id == sample_records[int(j)].subject_id else gt_matrix[gt_name_to_idx[sample_records[int(i)].subject_id], gt_name_to_idx[sample_records[int(j)].subject_id]]
            for i, j in zip(pair_i, pair_j)
        ],
        dtype=np.float64,
    )

    model = base.build_model(args=model_args, device=device)
    checkpoint_bundle = base.load_checkpoint_bundle(checkpoint_path)
    model.load_state_dict(checkpoint_bundle["state_dict"], strict=True)
    model.eval()

    scenario_specs = _scenario_spec(args)
    scenario_rows: List[dict] = []
    pair_rows_all: List[dict] = []
    mean_rows: List[dict] = []

    for scenario_index, scenario in enumerate(scenario_specs):
        vertex_sets: List[torch.Tensor] = []
        icp_points: List[np.ndarray] = []
        latents: List[torch.Tensor] = []
        for rec in sample_records:
            sample = sample_cache[int(rec.dataset_idx)]
            sample_d = base.sample_to_device(sample, device=device)
            pert_seed = base._scenario_seed(base_seed=int(model_args.seed), scenario_index=scenario_index, dataset_idx=int(rec.dataset_idx))
            V_in = base._apply_scenario(V=sample_d["verts"], params=params, scenario=scenario, seed=pert_seed).contiguous()
            z, _ = base.forward_model(model=model, sample_dict=sample_d, V_in=V_in, return_gate_info=False, add_noise=False)
            vertex_sets.append(V_in)
            latents.append(z.squeeze(0))
            idx = reg_utils.build_sample_vertex_indices(
                vertex_count=int(V_in.shape[0]),
                n_points=int(args.icp_points),
                seed=int(model_args.seed) * 1_000_003 + int(rec.dataset_idx) * 97 + 11,
            )
            icp_points.append(reg_utils.extract_point_subset(V_in, idx))

        raw_values = base.compute_pairwise_chamfer_values(
            vertex_sets=vertex_sets,
            pair_i=pair_i,
            pair_j=pair_j,
            batch_pairs=int(args.chamfer_batch_pairs),
            progress_desc=f"{scenario.name} raw mean-ICP probe",
            show_progress=True,
        )
        rigid_values, rigid_timings = reg_utils.compute_pairwise_registered_chamfer_values(
            vertex_sets=vertex_sets,
            icp_point_sets=icp_points,
            cpd_point_sets=None,
            pair_i=pair_i,
            pair_j=pair_j,
            batch_pairs=int(args.chamfer_batch_pairs),
            registration_workers=int(args.registration_workers),
            icp_max_correspondence_distance=float(args.icp_max_correspondence_distance),
            icp_max_iteration=int(args.icp_max_iteration),
            use_nonrigid_cpd=False,
            cpd_alpha=2.0,
            cpd_beta=0.2,
            cpd_max_iteration=15,
            cpd_tolerance=1.0e-4,
            cpd_outlier_weight=0.05,
            warp_chunk_size=4096,
            progress_desc=f"{scenario.name} rigid mean-ICP probe",
            show_progress=True,
            return_transforms=True,
        )
        transforms = rigid_timings["rigid_transforms"]
        rotation_mats = transforms[:, :3, :3]
        translations = transforms[:, :3, 3]
        angles = np.asarray([_safe_angle_deg(R) for R in rotation_mats], dtype=np.float64)
        axes_angles = np.asarray([_rotation_to_axis_angle(R)[0] for R in rotation_mats], dtype=np.float64)
        mean_transform = np.eye(4, dtype=np.float64)
        mean_transform[:3, :3] = _average_rotation(rotation_mats)
        mean_transform[:3, 3] = np.mean(translations, axis=0)

        mean_vertices = _apply_mean_transform(vertex_sets, pair_i, mean_transform)
        mean_values = base.compute_pairwise_chamfer_values(
            vertex_sets=mean_vertices,
            pair_i=pair_i,
            pair_j=pair_j,
            batch_pairs=int(args.chamfer_batch_pairs),
            progress_desc=f"{scenario.name} mean-ICP probe",
            show_progress=True,
        )

        pair_df = pd.DataFrame(
            {
                "scenario": scenario.name,
                "pair_index": np.arange(pair_i.shape[0], dtype=np.int64),
                "pair_type": np.where(is_same, "same_subject", "different_subject"),
                "subject_a": [sample_records[int(i)].subject_id for i in pair_i],
                "subject_b": [sample_records[int(j)].subject_id for j in pair_j],
                "topology_a": [sample_records[int(i)].topology_label for i in pair_i],
                "topology_b": [sample_records[int(j)].topology_label for j in pair_j],
                "sample_name_a": [sample_records[int(i)].sample_name for i in pair_i],
                "sample_name_b": [sample_records[int(j)].sample_name for j in pair_j],
                "gt_distance": gt_distances,
                "raw_chamfer": raw_values,
                "rigid_chamfer": rigid_values,
                "mean_icp_chamfer": mean_values,
                "rotation_angle_deg": angles,
                "translation_x": translations[:, 0],
                "translation_y": translations[:, 1],
                "translation_z": translations[:, 2],
                "translation_norm": np.linalg.norm(translations, axis=1),
            }
        )
        pair_df.to_csv(out_dir / f"{scenario.name}_pair_transform_metrics.csv", index=False)
        pair_rows_all.append(pair_df)

        summary = {
            "scenario": scenario.name,
            "n_pairs": int(pair_i.shape[0]),
            "raw_spearman": float(pd.Series(gt_distances).rank(method="average").corr(pd.Series(raw_values).rank(method="average"))),
            "rigid_spearman": float(pd.Series(gt_distances).rank(method="average").corr(pd.Series(rigid_values).rank(method="average"))),
            "mean_icp_spearman": float(pd.Series(gt_distances).rank(method="average").corr(pd.Series(mean_values).rank(method="average"))),
            "rotation_angle_mean": float(np.mean(angles)),
            "rotation_angle_std": float(np.std(angles)),
            "translation_norm_mean": float(np.mean(np.linalg.norm(translations, axis=1))),
            "translation_norm_std": float(np.std(np.linalg.norm(translations, axis=1))),
            "raw_mean": float(np.mean(raw_values)),
            "rigid_mean": float(np.mean(rigid_values)),
            "mean_icp_mean": float(np.mean(mean_values)),
            "mean_transform_translation_x": float(mean_transform[0, 3]),
            "mean_transform_translation_y": float(mean_transform[1, 3]),
            "mean_transform_translation_z": float(mean_transform[2, 3]),
            "mean_transform_rotation_angle_deg": float(_safe_angle_deg(mean_transform[:3, :3])),
        }
        scenario_rows.append(summary)
        mean_rows.append(
            {
                "scenario": scenario.name,
                "mean_transform": mean_transform.tolist(),
                "rotation_axes_mean": np.mean(axes_angles, axis=0).tolist(),
            }
        )

        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
        axes[0].hist(angles[np.isfinite(angles)], bins=35, color="#1f77b4", alpha=0.85)
        axes[0].set_title(f"{scenario.name}: rigid rotation angles")
        axes[0].set_xlabel("Angle (deg)")
        axes[0].set_ylabel("Count")
        axes[1].hist(np.linalg.norm(translations, axis=1), bins=35, color="#ff7f0e", alpha=0.85)
        axes[1].set_title(f"{scenario.name}: rigid translation norms")
        axes[1].set_xlabel("Translation norm")
        axes[1].set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(out_dir / f"{scenario.name}_transform_histograms.png", dpi=180)
        plt.close(fig)

    summary_df = pd.DataFrame(scenario_rows)
    summary_df.to_csv(out_dir / "scenario_transform_summary.csv", index=False)
    pd.DataFrame(mean_rows).to_json(out_dir / "mean_transforms.json", orient="records", indent=2)
    combined = pd.concat(pair_rows_all, ignore_index=True)
    combined.to_csv(out_dir / "pair_transform_metrics.csv", index=False)

    md_lines = [
        "# ICP Transform Dispersion",
        "",
        "| scenario | n_pairs | raw spearman | rigid spearman | mean-ICP spearman | rot mean | rot std | trans mean | trans std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in scenario_rows:
        md_lines.append(
            "| {scenario} | {n_pairs} | {raw_spearman:.4f} | {rigid_spearman:.4f} | {mean_icp_spearman:.4f} | {rotation_angle_mean:.4f} | {rotation_angle_std:.4f} | {translation_norm_mean:.4f} | {translation_norm_std:.4f} |".format(
                **row
            )
        )
    (out_dir / "README.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(scenario_rows, f, indent=2)
        f.write("\n")

    print(f"[done] wrote ICP transform dispersion analysis to {out_dir}")


if __name__ == "__main__":
    main()
