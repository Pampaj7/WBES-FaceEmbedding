from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PERTURBATED_DIR = REPO_ROOT / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic" / "perturbated"
if str(PERTURBATED_DIR) not in sys.path:
    sys.path.insert(0, str(PERTURBATED_DIR))

import compare_model_vs_chamfer_rankings as base  # noqa: E402
import registration_utils as reg_utils  # noqa: E402


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
    p = argparse.ArgumentParser(
        description=(
            "Probe whether raw / rigid / CPD metrics preserve or destroy the same-subject vs different-subject gap "
            "on a controlled REMESH cross-topology subset."
        )
    )
    p.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL))
    p.add_argument("--checkpoint_selector", type=str, default="best_by_xtopo_mesh_clean")
    p.add_argument("--config_json", type=str, default="")
    p.add_argument("--out_dir", type=str, default=str(REPO_ROOT / "checking_assumptions" / "outputs" / "same_vs_diff_gap_probe"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--data_dir", type=str, default="")
    p.add_argument("--dist_npz", type=str, default="")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--subject_split", type=str, default="eval", choices=("eval", "train", "all"))
    p.add_argument("--eval_fraction", type=float, default=0.2)
    p.add_argument("--max_subjects", type=int, default=12)
    p.add_argument("--max_meshes_per_subject_eval", type=int, default=6)
    p.add_argument("--topology_labels", type=str, default="crop,down8k,noisy,original,remesh,up60k")
    p.add_argument("--preload_workers", type=int, default=4)
    p.add_argument("--same_pair_limit", type=int, default=300)
    p.add_argument("--different_pair_multiplier", type=float, default=1.0)
    p.add_argument("--scenarios", type=str, default="clean,translation")
    p.add_argument("--base_sigma", type=float, default=-1.0)
    p.add_argument("--jitter_sigma", type=float, default=-1.0)
    p.add_argument("--translation_sigma", type=float, default=0.05)
    p.add_argument("--rotation_sigma", type=float, default=-1.0)
    p.add_argument("--mixed_jitter_sigma", type=float, default=-1.0)
    p.add_argument("--mixed_translation_sigma", type=float, default=-1.0)
    p.add_argument("--mixed_rotation_sigma", type=float, default=-1.0)
    p.add_argument("--rigid_rot_deg", type=float, default=-1.0)
    p.add_argument("--rigid_trans_scale", type=float, default=-1.0)
    p.add_argument("--rigid_rot_deg_min", type=float, default=-1.0)
    p.add_argument("--rigid_trans_scale_min", type=float, default=-1.0)
    p.add_argument("--icp_points", type=int, default=128)
    p.add_argument("--cpd_points", type=int, default=64)
    p.add_argument("--nicp_points", type=int, default=128)
    p.add_argument("--registration_workers", type=int, default=8)
    p.add_argument("--icp_max_correspondence_distance", type=float, default=0.05)
    p.add_argument("--icp_max_iteration", type=int, default=20)
    p.add_argument("--nicp_iterations", type=int, default=8)
    p.add_argument("--nicp_neighbors", type=int, default=8)
    p.add_argument("--nicp_smoothness_lambda", type=float, default=2.0)
    p.add_argument("--nicp_damping", type=float, default=1.0)
    p.add_argument("--cpd_alpha", type=float, default=2.0)
    p.add_argument("--cpd_beta", type=float, default=0.2)
    p.add_argument("--cpd_max_iteration", type=int, default=15)
    p.add_argument("--cpd_tolerance", type=float, default=1.0e-4)
    p.add_argument("--cpd_outlier_weight", type=float, default=0.05)
    p.add_argument("--warp_chunk_size", type=int, default=4096)
    p.add_argument("--chamfer_batch_pairs", type=int, default=8)
    p.add_argument("--skip_cpd", action="store_true")
    return p.parse_args()


def _format_float(x: float) -> str:
    return "" if not math.isfinite(float(x)) else f"{float(x):.6f}"


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _auc_lower_is_positive(same_vals: np.ndarray, diff_vals: np.ndarray) -> float:
    same = np.asarray(same_vals, dtype=np.float64)
    diff = np.asarray(diff_vals, dtype=np.float64)
    same = same[np.isfinite(same)]
    diff = diff[np.isfinite(diff)]
    if same.size == 0 or diff.size == 0:
        return float("nan")

    scores = np.concatenate([-same, -diff], axis=0)
    labels = np.concatenate([np.ones(same.size, dtype=np.int64), np.zeros(diff.size, dtype=np.int64)], axis=0)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < sorted_scores.shape[0]:
        j = i + 1
        while j < sorted_scores.shape[0] and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    pos_ranks = ranks[labels == 1]
    n_pos = float(pos_ranks.size)
    n_neg = float((labels == 0).sum())
    return float((pos_ranks.sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def _metric_summary(values: np.ndarray, is_same_mask: np.ndarray, gt_distances: np.ndarray | None = None) -> dict:
    vals = np.asarray(values, dtype=np.float64)
    mask_same = np.asarray(is_same_mask, dtype=bool)
    same = vals[mask_same]
    diff = vals[~mask_same]
    same_finite = same[np.isfinite(same)]
    diff_finite = diff[np.isfinite(diff)]
    row = {
        "same_mean": float(np.mean(same_finite)) if same_finite.size else float("nan"),
        "same_median": float(np.median(same_finite)) if same_finite.size else float("nan"),
        "different_mean": float(np.mean(diff_finite)) if diff_finite.size else float("nan"),
        "different_median": float(np.median(diff_finite)) if diff_finite.size else float("nan"),
        "gap_diff_minus_same": (
            float(np.mean(diff_finite) - np.mean(same_finite)) if same_finite.size and diff_finite.size else float("nan")
        ),
        "same_vs_diff_auc": _auc_lower_is_positive(same, diff),
    }
    if gt_distances is not None:
        gt = np.asarray(gt_distances, dtype=np.float64)
        diff_mask = (~mask_same) & np.isfinite(vals) & np.isfinite(gt)
        if int(diff_mask.sum()) >= 2:
            row["different_only_spearman_vs_gt"] = float(base.spearman_corr(gt[diff_mask], vals[diff_mask]))
            row["different_only_pearson_vs_gt"] = float(base.pearson_corr(gt[diff_mask], vals[diff_mask]))
        else:
            row["different_only_spearman_vs_gt"] = float("nan")
            row["different_only_pearson_vs_gt"] = float("nan")
    return row


def _select_pairs(
    sample_records: Sequence[object],
    *,
    same_pair_limit: int,
    different_pair_limit: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    base_sigma = float(args.base_sigma)
    if base_sigma < 0.0:
        base_sigma = float(getattr(model_args, "sigma_max", 0.05))
    scenarios = base._parse_scenarios(args.scenarios, base_sigma=base_sigma, cli_args=args)

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
        max_meshes_per_subject_eval=int(model_args.max_meshes_per_subject_eval),
        seed=int(model_args.seed),
    )
    sample_cache = base.preload_eval_samples(dataset=dataset, eval_plan=eval_plan, workers=int(args.preload_workers))
    sample_records = base.build_sample_eval_records(
        dataset=dataset,
        eval_plan=eval_plan,
        eval_subjects=target_subjects,
        sample_cache=sample_cache,
    )
    requested_topology_labels = base._parse_topology_labels(args.topology_labels)
    if requested_topology_labels:
        sample_records = base._filter_sample_records_by_topology_labels(sample_records=sample_records, topology_labels=requested_topology_labels)

    pair_i, pair_j, is_same = _select_pairs(
        sample_records=sample_records,
        same_pair_limit=int(args.same_pair_limit),
        different_pair_limit=int(round(float(args.same_pair_limit) * float(args.different_pair_multiplier))),
        seed=int(args.seed),
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

    scenario_summary_rows: List[dict] = []
    for scenario_index, scenario in enumerate(scenarios):
        vertex_sets: List[torch.Tensor] = []
        latent_vectors: List[torch.Tensor] = []
        icp_point_sets: List[np.ndarray] = []
        cpd_point_sets: List[np.ndarray] = []
        nicp_point_sets: List[np.ndarray] = []

        for rec in sample_records:
            sample = sample_cache[int(rec.dataset_idx)]
            sample_d = base.sample_to_device(sample, device=device)
            pert_seed = base._scenario_seed(base_seed=int(model_args.seed), scenario_index=scenario_index, dataset_idx=int(rec.dataset_idx))
            V_in = base._apply_scenario(V=sample_d["verts"], params=params, scenario=scenario, seed=pert_seed).contiguous()
            z, _ = base.forward_model(
                model=model,
                sample_dict=sample_d,
                V_in=V_in,
                return_gate_info=False,
                add_noise=False,
            )
            latent_vectors.append(z.squeeze(0).detach())
            vertex_sets.append(V_in)

            verts_np = V_in.detach().cpu().numpy()
            icp_idx = reg_utils.build_sample_vertex_indices(
                vertex_count=verts_np.shape[0],
                n_points=int(args.icp_points),
                seed=int(rec.dataset_idx) * 97 + 11,
            )
            nicp_idx = reg_utils.build_sample_vertex_indices(
                vertex_count=verts_np.shape[0],
                n_points=int(args.nicp_points),
                seed=int(model_args.seed) * 1_000_003 + int(rec.dataset_idx) * 97 + 23,
            )
            cpd_idx = reg_utils.build_sample_vertex_indices(
                vertex_count=verts_np.shape[0],
                n_points=int(args.cpd_points),
                seed=int(rec.dataset_idx) * 193 + 17,
            )
            icp_point_sets.append(reg_utils.extract_point_subset(V_in, icp_idx))
            nicp_point_sets.append(reg_utils.extract_point_subset(V_in, nicp_idx))
            cpd_point_sets.append(reg_utils.extract_point_subset(V_in, cpd_idx))

        Z = torch.stack(latent_vectors, dim=0)
        latent_values = torch.linalg.vector_norm(
            Z.index_select(0, torch.as_tensor(pair_i, device=device, dtype=torch.long))
            - Z.index_select(0, torch.as_tensor(pair_j, device=device, dtype=torch.long)),
            dim=1,
        ).detach().cpu().numpy().astype(np.float64, copy=False)

        raw_values = base.compute_pairwise_chamfer_values(
            vertex_sets=vertex_sets,
            pair_i=pair_i,
            pair_j=pair_j,
            batch_pairs=int(args.chamfer_batch_pairs),
            progress_desc=f"{scenario.name} raw subset",
            show_progress=True,
        )

        rigid_values, rigid_timings = reg_utils.compute_pairwise_registered_chamfer_values(
            vertex_sets=vertex_sets,
            icp_point_sets=icp_point_sets,
            cpd_point_sets=cpd_point_sets,
            pair_i=pair_i,
            pair_j=pair_j,
            batch_pairs=int(args.chamfer_batch_pairs),
            registration_workers=int(args.registration_workers),
            icp_max_correspondence_distance=float(args.icp_max_correspondence_distance),
            icp_max_iteration=int(args.icp_max_iteration),
            use_nonrigid_cpd=False,
            cpd_alpha=float(args.cpd_alpha),
            cpd_beta=float(args.cpd_beta),
            cpd_max_iteration=int(args.cpd_max_iteration),
            cpd_tolerance=float(args.cpd_tolerance),
            cpd_outlier_weight=float(args.cpd_outlier_weight),
            warp_chunk_size=int(args.warp_chunk_size),
            progress_desc=f"{scenario.name} rigid subset",
            show_progress=True,
        )

        nicp_values, nicp_timings = reg_utils.compute_pairwise_nicp_correspondence_values(
            icp_point_sets=icp_point_sets,
            nicp_point_sets=nicp_point_sets,
            pair_i=pair_i,
            pair_j=pair_j,
            registration_workers=int(args.registration_workers),
            icp_max_correspondence_distance=float(args.icp_max_correspondence_distance),
            icp_max_iteration=int(args.icp_max_iteration),
            nicp_iterations=int(args.nicp_iterations),
            nicp_neighbors=int(args.nicp_neighbors),
            nicp_smoothness_lambda=float(args.nicp_smoothness_lambda),
            nicp_damping=float(args.nicp_damping),
            progress_desc=f"{scenario.name} nicp subset",
            show_progress=True,
        )

        if bool(args.skip_cpd):
            cpd_values = np.full_like(rigid_values, np.nan, dtype=np.float64)
            cpd_timings = {
                "total_pair_seconds": np.full_like(rigid_values, np.nan, dtype=np.float64),
                "registration_seconds": np.full_like(rigid_values, np.nan, dtype=np.float64),
            }
        else:
            cpd_values, cpd_timings = reg_utils.compute_pairwise_registered_chamfer_values(
                vertex_sets=vertex_sets,
                icp_point_sets=icp_point_sets,
                cpd_point_sets=cpd_point_sets,
                pair_i=pair_i,
                pair_j=pair_j,
                batch_pairs=int(args.chamfer_batch_pairs),
                registration_workers=int(args.registration_workers),
                icp_max_correspondence_distance=float(args.icp_max_correspondence_distance),
                icp_max_iteration=int(args.icp_max_iteration),
                use_nonrigid_cpd=True,
                cpd_alpha=float(args.cpd_alpha),
                cpd_beta=float(args.cpd_beta),
                cpd_max_iteration=int(args.cpd_max_iteration),
                cpd_tolerance=float(args.cpd_tolerance),
                cpd_outlier_weight=float(args.cpd_outlier_weight),
                warp_chunk_size=int(args.warp_chunk_size),
                progress_desc=f"{scenario.name} cpd subset",
                show_progress=True,
            )

        pair_rows: List[dict] = []
        for idx in range(pair_i.shape[0]):
            i = int(pair_i[idx])
            j = int(pair_j[idx])
            rec_i = sample_records[i]
            rec_j = sample_records[j]
            pair_rows.append(
                {
                    "scenario": str(scenario.name),
                    "pair_index": int(idx),
                    "pair_type": "same_subject" if bool(is_same[idx]) else "different_subject",
                    "subject_a": str(rec_i.subject_id),
                    "topology_a": str(rec_i.topology_label),
                    "sample_name_a": str(rec_i.sample_name),
                    "subject_b": str(rec_j.subject_id),
                    "topology_b": str(rec_j.topology_label),
                    "sample_name_b": str(rec_j.sample_name),
                    "gt_distance": float(gt_distances[idx]),
                    "latent_distance": float(latent_values[idx]),
                    "raw_chamfer": float(raw_values[idx]),
                    "rigid_registered_chamfer": float(rigid_values[idx]),
                    "nicp_correspondence": float(nicp_values[idx]),
                    "cpd_registered_chamfer": float(cpd_values[idx]),
                }
            )

        scenario_dir = out_dir / scenario.name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(scenario_dir / "pair_metrics.csv", pair_rows)

        metric_map = {
            "latent_distance": latent_values,
            "raw_chamfer": raw_values,
            "rigid_registered_chamfer": rigid_values,
            "nicp_correspondence": nicp_values,
            "cpd_registered_chamfer": cpd_values,
        }
        metric_summary_rows: List[dict] = []
        for metric_name, metric_values in metric_map.items():
            summary = _metric_summary(metric_values, is_same_mask=is_same, gt_distances=gt_distances)
            summary_row = {
                "scenario": str(scenario.name),
                "metric": metric_name,
                **summary,
            }
            metric_summary_rows.append(summary_row)
            scenario_summary_rows.append(summary_row)

        _write_csv(scenario_dir / "metric_summary.csv", metric_summary_rows)
        with open(scenario_dir / "metric_summary.json", "w", encoding="utf-8") as f:
            json.dump(metric_summary_rows, f, indent=2)
            f.write("\n")

        md_lines = [
            f"# Same-vs-Different Gap Probe: {scenario.name}",
            "",
            f"- pairs: `{len(pair_rows)}`",
            f"- same-subject pairs: `{int(is_same.sum())}`",
            f"- different-subject pairs: `{int((~is_same).sum())}`",
            "",
            "| metric | same mean | diff mean | gap diff-same | auc(same if smaller) | diff-only spearman vs gt |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in metric_summary_rows:
            md_lines.append(
                "| {metric} | {same_mean} | {different_mean} | {gap_diff_minus_same} | {same_vs_diff_auc} | {different_only_spearman_vs_gt} |".format(
                    metric=row["metric"],
                    same_mean=_format_float(row["same_mean"]),
                    different_mean=_format_float(row["different_mean"]),
                    gap_diff_minus_same=_format_float(row["gap_diff_minus_same"]),
                    same_vs_diff_auc=_format_float(row["same_vs_diff_auc"]),
                    different_only_spearman_vs_gt=_format_float(row["different_only_spearman_vs_gt"]),
                )
            )
        with open(scenario_dir / "metric_summary.md", "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

        with open(scenario_dir / "timings.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "rigid_mean_total_seconds": float(np.mean(rigid_timings["total_pair_seconds"])),
                    "nicp_mean_total_seconds": float(np.mean(nicp_timings["total_pair_seconds"])),
                    "cpd_mean_total_seconds": float(np.mean(cpd_timings["total_pair_seconds"])),
                    "rigid_mean_registration_seconds": float(np.mean(rigid_timings["registration_seconds"])),
                    "nicp_mean_registration_seconds": float(
                        np.mean(nicp_timings["rigid_icp_seconds"] + nicp_timings["nonrigid_refine_seconds"])
                    ),
                    "cpd_mean_registration_seconds": float(np.mean(cpd_timings["registration_seconds"])),
                },
                f,
                indent=2,
            )
            f.write("\n")

    _write_csv(out_dir / "overall_metric_summary.csv", scenario_summary_rows)
    with open(out_dir / "overall_metric_summary.json", "w", encoding="utf-8") as f:
        json.dump(scenario_summary_rows, f, indent=2)
        f.write("\n")

    root_md = [
        "# Same-vs-Different Gap Probe",
        "",
        "This probe asks a more direct question than the GT ranking benchmark:",
        "",
        "- do same-subject cross-topology pairs stay closer than different-subject pairs?",
        "- does rigid ICP improve or shrink that separation?",
        "- does CPD collapse it even more?",
        "",
        "| scenario | metric | same mean | diff mean | gap diff-same | auc(same if smaller) | diff-only spearman vs gt |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in scenario_summary_rows:
        root_md.append(
            "| {scenario} | {metric} | {same_mean} | {different_mean} | {gap_diff_minus_same} | {same_vs_diff_auc} | {different_only_spearman_vs_gt} |".format(
                scenario=row["scenario"],
                metric=row["metric"],
                same_mean=_format_float(row["same_mean"]),
                different_mean=_format_float(row["different_mean"]),
                gap_diff_minus_same=_format_float(row["gap_diff_minus_same"]),
                same_vs_diff_auc=_format_float(row["same_vs_diff_auc"]),
                different_only_spearman_vs_gt=_format_float(row["different_only_spearman_vs_gt"]),
            )
        )
    with open(out_dir / "README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(root_md) + "\n")

    print(f"[done] wrote same-vs-different gap probe outputs to {out_dir}")


if __name__ == "__main__":
    main()
