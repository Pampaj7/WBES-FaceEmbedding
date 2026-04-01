#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

import compare_model_vs_chamfer_rankings_faceverse as faceverse_base


benchmark_base = faceverse_base.base
THIS_DIR = Path(__file__).resolve().parent

ALLOWED_SWEEP_SCENARIOS = ("jitter", "translation", "rotation", "mixed")
CLEAN_SPEC = benchmark_base.ScenarioSpec(
    name="clean",
    jitter_sigma=0.0,
    rotation_sigma=0.0,
    translation_sigma=0.0,
)


def parse_args() -> argparse.Namespace:
    parser = faceverse_base.build_arg_parser()
    parser.description = (
        "Run the FaceVerse GT ranking benchmark over a progressive sigma sweep while "
        "preserving the same model/Chamfer evaluation protocol."
    )
    parser.set_defaults(chamfer_use_icp=False)
    parser.add_argument(
        "--sweep_scenarios",
        type=str,
        default="jitter,translation,rotation,mixed",
        help="Comma-separated scenarios from: jitter,translation,rotation,mixed",
    )
    parser.add_argument(
        "--sigma_values",
        type=str,
        default="0.00,0.02,0.05,0.10,0.15,0.20",
        help="Comma-separated sigma values for the sweep",
    )
    parser.add_argument(
        "--include_clean_once",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute the clean baseline once and reuse it across the sweep",
    )
    parser.add_argument(
        "--progressive_output_layout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store a dedicated subdirectory for each scenario/sigma run",
    )
    parser.add_argument(
        "--summary_filename",
        type=str,
        default="sigma_sweep_summary.csv",
        help="Filename for the aggregated sweep CSV written at the root output directory",
    )
    return parser.parse_args()


def _parse_sweep_scenarios(text: str) -> List[str]:
    names: List[str] = []
    seen = set()
    for raw in str(text).split(","):
        name = raw.strip().lower()
        if not name:
            continue
        if name == "clean":
            raise ValueError("Do not include 'clean' in --sweep_scenarios; use --include_clean_once instead")
        if name not in ALLOWED_SWEEP_SCENARIOS:
            raise ValueError(
                f"Unsupported sweep scenario '{name}'. Allowed={list(ALLOWED_SWEEP_SCENARIOS)}"
            )
        if name not in seen:
            names.append(name)
            seen.add(name)
    if not names:
        raise ValueError("sweep_scenarios must contain at least one valid scenario")
    return names


def _parse_sigma_values(text: str) -> List[float]:
    values: List[float] = []
    seen = set()
    for raw in str(text).split(","):
        token = raw.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(f"Invalid sigma value '{token}'") from exc
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"Sigma values must be finite and non-negative, got {value!r}")
        key = f"{value:.12g}"
        if key not in seen:
            values.append(float(value))
            seen.add(key)
    if not values:
        raise ValueError("sigma_values must contain at least one valid sigma")
    return values


def _is_zero_sigma(value: float) -> bool:
    return abs(float(value)) <= 1e-15


def _sigma_token(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = f"{text}.00"
    elif len(text.split(".", 1)[1]) == 1:
        text = f"{text}0"
    return text.replace("-", "m").replace(".", "p")


def _build_scenario_spec(name: str, sigma: float):
    sigma = float(sigma)
    if name == "jitter":
        return benchmark_base.ScenarioSpec(name=name, jitter_sigma=sigma, rotation_sigma=0.0, translation_sigma=0.0)
    if name == "translation":
        return benchmark_base.ScenarioSpec(name=name, jitter_sigma=0.0, rotation_sigma=0.0, translation_sigma=sigma)
    if name == "rotation":
        return benchmark_base.ScenarioSpec(name=name, jitter_sigma=0.0, rotation_sigma=sigma, translation_sigma=0.0)
    if name == "mixed":
        return benchmark_base.ScenarioSpec(name=name, jitter_sigma=sigma, rotation_sigma=sigma, translation_sigma=sigma)
    raise ValueError(f"Unsupported sweep scenario: {name}")


def _finalize_result(row: Dict[str, object]) -> Dict[str, object]:
    out = dict(row)
    out["delta_spearman"] = float(out["latent_spearman"]) - float(out["chamfer_spearman"])
    out["delta_pearson"] = float(out["latent_pearson"]) - float(out["chamfer_pearson"])
    out["model_beats_chamfer"] = bool(float(out["latent_spearman"]) > float(out["chamfer_spearman"]))
    return out


def _clone_clean_as_scenario(clean_row: Dict[str, object], scenario_name: str) -> Dict[str, object]:
    out = dict(clean_row)
    out["scenario"] = str(scenario_name)
    return out


def _format_float(value: object) -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(value_f):
        return ""
    return f"{value_f:.6f}"


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)
        handle.write("\n")


def _build_default_out_dir(
    checkpoint_path: Path,
    cli_args,
    target_subjects: Sequence[str],
    max_meshes_per_subject_eval: int,
    sweep_scenarios: Sequence[str],
    sigma_values: Sequence[float],
) -> Path:
    scenario_token = "-".join(sweep_scenarios)
    sigma_token = "-".join(_sigma_token(value) for value in sigma_values)
    align_token = "icp" if bool(cli_args.chamfer_use_icp) else "noicp"
    slug = benchmark_base.slugify_token(
        f"{checkpoint_path.stem}_split-{cli_args.subject_split}_pairs-{cli_args.pair_mode}_"
        f"agglvl-{cli_args.aggregation_level}_subjects-{len(target_subjects)}_"
        f"meshes-{int(max_meshes_per_subject_eval)}_align-{align_token}_"
        f"sweep-{scenario_token}_sigmas-{sigma_token}"
    )
    return THIS_DIR / "faceverse_ranking_vs_gt_sigma_sweep" / slug


def _resolve_item_paths(
    root_out_dir: Path,
    label: str,
    progressive_output_layout: bool,
) -> tuple[Path, Path, Path, Path]:
    root_out_dir.mkdir(parents=True, exist_ok=True)
    if progressive_output_layout:
        item_out_dir = root_out_dir / label
        item_out_dir.mkdir(parents=True, exist_ok=True)
        return (
            item_out_dir,
            item_out_dir / "ranking_summary.json",
            item_out_dir / "ranking_summary.csv",
            item_out_dir / "ranking_summary.md",
        )
    return (
        root_out_dir,
        root_out_dir / f"{label}__ranking_summary.json",
        root_out_dir / f"{label}__ranking_summary.csv",
        root_out_dir / f"{label}__ranking_summary.md",
    )


def _build_payload(
    *,
    checkpoint_path: Path,
    run_dir: Path,
    cli_args,
    model_args,
    device,
    target_subjects: Sequence[str],
    inference_filter_summary: Dict[str, object],
    gt_filter_summary: Dict[str, object],
    gt_dist_summary: Dict[str, object],
    eval_plan,
    sample_records,
    pair_ctx,
    chamfer_cache_stats: Dict[str, object],
    params,
    scenario_spec,
    sigma: float,
    include_clean_once: bool,
    clean_row: Dict[str, object] | None,
    rows_for_outputs: Sequence[dict],
) -> dict:
    return {
        "checkpoint": str(checkpoint_path),
        "run_dir": str(run_dir),
        "checkpoint_selector": str(cli_args.checkpoint_selector),
        "device": str(device),
        "data_dir": str(model_args.data_dir),
        "gt_mesh_dir": str(Path(cli_args.gt_mesh_dir).expanduser().resolve()),
        "dist_npz": str(Path(cli_args.dist_npz).expanduser().resolve()),
        "reference_metric": "gt_vertex_mean_l2",
        "subject_split": str(cli_args.subject_split),
        "eval_fraction": float(cli_args.eval_fraction),
        "seed": int(model_args.seed),
        "max_subjects_requested": int(cli_args.max_subjects),
        "max_meshes_per_subject_eval": int(model_args.max_meshes_per_subject_eval),
        "pair_mode": str(cli_args.pair_mode),
        "aggregation_level": str(cli_args.aggregation_level),
        "selected_subjects": list(target_subjects),
        "inference_filter_summary": dict(inference_filter_summary),
        "gt_filter_summary": dict(gt_filter_summary),
        "gt_distance_matrix_summary": dict(gt_dist_summary),
        "eval_plan_summary": benchmark_base.summarize_eval_plan(eval_plan),
        "sample_eval_summary": benchmark_base.summarize_sample_records(sample_records),
        "pair_context": {
            "n_subjects": int(pair_ctx.n_subjects),
            "n_samples": int(pair_ctx.n_samples),
            "n_pairs": int(pair_ctx.pair_count),
            "n_mesh_pairs": int(pair_ctx.mesh_pair_count),
            "n_subject_pairs": int(pair_ctx.subject_pair_count),
            "n_topology_labels": int(pair_ctx.n_topology_labels),
        },
        "chamfer_protocol": {
            "use_icp": bool(cli_args.chamfer_use_icp),
            "alignment_stage": "precomputed_clean_pairs" if bool(cli_args.chamfer_use_icp) else "none",
            "icp_points": int(cli_args.icp_points),
            "icp_max_correspondence_distance": float(cli_args.icp_max_correspondence_distance),
            "icp_max_iteration": int(cli_args.icp_max_iteration),
            "icp_workers": int(cli_args.icp_workers),
        },
        "chamfer_cache_verts": dict(chamfer_cache_stats),
        "perturbation_params": {
            "rigid_rot_deg": float(params.rigid_rot_deg),
            "rigid_trans_scale": float(params.rigid_trans_scale),
            "rigid_rot_deg_min": float(params.rigid_rot_deg_min),
            "rigid_trans_scale_min": float(params.rigid_trans_scale_min),
        },
        "sweep": {
            "scenario": str(scenario_spec.name),
            "sigma": float(sigma),
            "include_clean_once": bool(include_clean_once),
            "scenario_parameters": benchmark_base._describe_scenario(scenario_spec, params=params),
        },
        "clean_baseline": dict(clean_row) if clean_row is not None else None,
        "rows": [dict(row) for row in rows_for_outputs],
    }


def _build_summary_row(
    *,
    scenario_name: str,
    sigma: float,
    scenario_spec,
    result_row: Dict[str, object],
    clean_row: Dict[str, object] | None,
    params,
    item_out_dir: Path,
    json_path: Path,
    csv_path: Path,
    md_path: Path,
) -> dict:
    desc = benchmark_base._describe_scenario(scenario_spec, params=params)
    clean_latent = float(clean_row["latent_spearman"]) if clean_row is not None else float("nan")
    clean_chamfer = float(clean_row["chamfer_spearman"]) if clean_row is not None else float("nan")
    return {
        "scenario": str(scenario_name),
        "sigma": float(sigma),
        "jitter_sigma": float(desc["jitter_sigma"]),
        "rotation_sigma": float(desc["rotation_sigma"]),
        "translation_sigma": float(desc["translation_sigma"]),
        "rotation_angle_max_deg": float(desc["rotation_angle_max_deg"]),
        "translation_axis_std": float(desc["translation_axis_std"]),
        "latent_spearman": float(result_row["latent_spearman"]),
        "chamfer_spearman": float(result_row["chamfer_spearman"]),
        "delta_spearman": float(result_row["delta_spearman"]),
        "latent_pearson": float(result_row["latent_pearson"]),
        "chamfer_pearson": float(result_row["chamfer_pearson"]),
        "delta_pearson": float(result_row["delta_pearson"]),
        "model_beats_chamfer": bool(result_row["model_beats_chamfer"]),
        "clean_latent_spearman": clean_latent,
        "clean_chamfer_spearman": clean_chamfer,
        "latent_drop_vs_clean": (
            clean_latent - float(result_row["latent_spearman"]) if clean_row is not None else float("nan")
        ),
        "chamfer_drop_vs_clean": (
            clean_chamfer - float(result_row["chamfer_spearman"]) if clean_row is not None else float("nan")
        ),
        "n_subjects": int(result_row["n_subjects"]),
        "n_samples": int(result_row["n_samples"]),
        "n_pairs": int(result_row["n_pairs"]),
        "n_mesh_pairs": int(result_row["n_mesh_pairs"]),
        "n_subject_pairs": int(result_row["n_subject_pairs"]),
        "output_dir": str(item_out_dir),
        "output_json": str(json_path),
        "output_csv": str(csv_path),
        "output_md": str(md_path),
    }


def _write_sweep_summary_csv(path: Path, rows: Sequence[dict]) -> None:
    header = [
        "scenario",
        "sigma",
        "jitter_sigma",
        "rotation_sigma",
        "translation_sigma",
        "rotation_angle_max_deg",
        "translation_axis_std",
        "latent_spearman",
        "chamfer_spearman",
        "delta_spearman",
        "latent_pearson",
        "chamfer_pearson",
        "delta_pearson",
        "model_beats_chamfer",
        "clean_latent_spearman",
        "clean_chamfer_spearman",
        "latent_drop_vs_clean",
        "chamfer_drop_vs_clean",
        "n_subjects",
        "n_samples",
        "n_pairs",
        "n_mesh_pairs",
        "n_subject_pairs",
        "output_dir",
        "output_json",
        "output_csv",
        "output_md",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "scenario": row["scenario"],
                    "sigma": _format_float(row["sigma"]),
                    "jitter_sigma": _format_float(row["jitter_sigma"]),
                    "rotation_sigma": _format_float(row["rotation_sigma"]),
                    "translation_sigma": _format_float(row["translation_sigma"]),
                    "rotation_angle_max_deg": _format_float(row["rotation_angle_max_deg"]),
                    "translation_axis_std": _format_float(row["translation_axis_std"]),
                    "latent_spearman": _format_float(row["latent_spearman"]),
                    "chamfer_spearman": _format_float(row["chamfer_spearman"]),
                    "delta_spearman": _format_float(row["delta_spearman"]),
                    "latent_pearson": _format_float(row["latent_pearson"]),
                    "chamfer_pearson": _format_float(row["chamfer_pearson"]),
                    "delta_pearson": _format_float(row["delta_pearson"]),
                    "model_beats_chamfer": int(bool(row["model_beats_chamfer"])),
                    "clean_latent_spearman": _format_float(row["clean_latent_spearman"]),
                    "clean_chamfer_spearman": _format_float(row["clean_chamfer_spearman"]),
                    "latent_drop_vs_clean": _format_float(row["latent_drop_vs_clean"]),
                    "chamfer_drop_vs_clean": _format_float(row["chamfer_drop_vs_clean"]),
                    "n_subjects": int(row["n_subjects"]),
                    "n_samples": int(row["n_samples"]),
                    "n_pairs": int(row["n_pairs"]),
                    "n_mesh_pairs": int(row["n_mesh_pairs"]),
                    "n_subject_pairs": int(row["n_subject_pairs"]),
                    "output_dir": row["output_dir"],
                    "output_json": row["output_json"],
                    "output_csv": row["output_csv"],
                    "output_md": row["output_md"],
                }
            )


def main() -> None:
    cli_args = parse_args()
    sweep_scenarios = _parse_sweep_scenarios(cli_args.sweep_scenarios)
    sigma_values = _parse_sigma_values(cli_args.sigma_values)

    run_dir, checkpoint_path = benchmark_base._resolve_run_dir_and_checkpoint(
        cli_args.model_path,
        selector=str(cli_args.checkpoint_selector),
    )
    model_args = benchmark_base._resolve_runtime_args(
        cli_args,
        benchmark_base.merge_run_args(checkpoint_path, explicit_config_json=cli_args.config_json),
    )
    params = benchmark_base.PerturbationParams.from_namespace(model_args)

    benchmark_base.seed_everything(int(model_args.seed))
    device = benchmark_base._resolve_device(cli_args.device)

    dataset = benchmark_base.GTReadyDataset(model_args.data_dir)
    inference_subject_map, inference_filter_summary = faceverse_base._build_faceverse_inference_subject_map(
        files=dataset.files,
        pattern=str(cli_args.pattern),
        subject_ids=cli_args.subject_ids,
        pose_ids=cli_args.pose_ids,
    )

    gt_subject_paths, gt_filter_summary = faceverse_base._resolve_gt_subject_paths(
        gt_mesh_dir=str(cli_args.gt_mesh_dir),
        gt_pattern=str(cli_args.gt_pattern),
        subject_ids=cli_args.subject_ids,
        pose_ids=cli_args.pose_ids,
        dist_npz=str(cli_args.dist_npz),
        recompute_gt_dist=bool(cli_args.recompute_gt_dist),
    )

    overlapping_subjects = sorted(set(inference_subject_map.keys()) & set(gt_subject_paths.keys()))
    if not overlapping_subjects:
        raise RuntimeError("No overlapping FaceVerse subjects between inference data and original GT meshes")

    _, _, target_subjects = benchmark_base._select_subject_subset(
        subjects=overlapping_subjects,
        subject_split=str(cli_args.subject_split),
        eval_fraction=float(cli_args.eval_fraction),
        seed=int(model_args.seed),
        max_subjects=int(cli_args.max_subjects),
    )
    if not target_subjects:
        raise RuntimeError("No FaceVerse subjects remained after subject selection")

    gt_matrix, gt_name_to_idx, gt_dist_summary = faceverse_base._resolve_faceverse_gt_distance_matrix(
        gt_subject_paths=gt_subject_paths,
        out_path=Path(cli_args.dist_npz).expanduser().resolve(),
        recompute=bool(cli_args.recompute_gt_dist),
        workers=int(cli_args.gt_workers),
    )

    eval_plan = benchmark_base.build_eval_plan(
        subj_map=inference_subject_map,
        eval_subjects=target_subjects,
        max_meshes_per_subject_eval=int(model_args.max_meshes_per_subject_eval),
        seed=int(model_args.seed),
    )
    sample_cache = (
        benchmark_base.preload_eval_samples(
            dataset=dataset,
            eval_plan=eval_plan,
            workers=int(cli_args.preload_workers),
        )
        if cli_args.preload_eval_samples
        else None
    )
    if sample_cache is not None:
        sample_cache, chamfer_cache_stats = benchmark_base.maybe_cache_chamfer_vertices_on_device(
            sample_cache=sample_cache,
            device=device,
            cache_mode=str(cli_args.chamfer_cache_verts),
            max_mb=float(cli_args.chamfer_cache_verts_max_mb),
        )
    else:
        chamfer_cache_stats = {
            "mode": str(cli_args.chamfer_cache_verts),
            "enabled": False,
            "device": "",
            "vertex_bytes": 0,
            "vertex_mb": 0.0,
            "reason": "no_preloaded_samples",
        }

    sample_records = benchmark_base.build_sample_eval_records(
        dataset=dataset,
        eval_plan=eval_plan,
        eval_subjects=target_subjects,
        sample_cache=sample_cache,
    )
    pair_ctx = benchmark_base.build_pair_eval_context(
        sample_records=sample_records,
        name_to_idx=gt_name_to_idx,
        gt_matrix=gt_matrix,
        device=device,
        pair_mode=str(cli_args.pair_mode),
        aggregation_level=str(cli_args.aggregation_level),
    )
    if pair_ctx.pair_count <= 0:
        raise RuntimeError(
            "No valid FaceVerse evaluation pairs found for the selected subset. "
            f"pair_mode={cli_args.pair_mode!r} aggregation_level={cli_args.aggregation_level!r}"
        )

    pairwise_icp_transforms: np.ndarray | None = None
    if bool(cli_args.chamfer_use_icp):
        clean_icp_point_sets: List[np.ndarray] = []
        for record in pair_ctx.sample_records:
            sample = sample_cache[int(record.dataset_idx)] if sample_cache is not None else dataset[int(record.dataset_idx)]
            clean_icp_point_sets.append(
                faceverse_base._sample_vertices_for_icp(
                    sample["verts"],
                    n_points=int(cli_args.icp_points),
                    seed=int(model_args.seed) * 1_000_003 + int(record.dataset_idx),
                )
            )
        pairwise_icp_transforms = faceverse_base._precompute_pairwise_icp_transforms(
            icp_point_sets=clean_icp_point_sets,
            pair_i=pair_ctx.pair_i_cpu,
            pair_j=pair_ctx.pair_j_cpu,
            batch_pairs=int(cli_args.chamfer_batch_pairs),
            icp_workers=int(cli_args.icp_workers),
            icp_max_correspondence_distance=float(cli_args.icp_max_correspondence_distance),
            icp_max_iteration=int(cli_args.icp_max_iteration),
            progress_desc="clean pairwise ICP",
            show_progress=True,
        )

    model = benchmark_base.build_model(args=model_args, device=device)
    checkpoint_bundle = benchmark_base.load_checkpoint_bundle(checkpoint_path)
    model.load_state_dict(checkpoint_bundle["state_dict"], strict=True)
    model.eval()

    root_out_dir = (
        Path(cli_args.out_dir).expanduser().resolve()
        if cli_args.out_dir
        else _build_default_out_dir(
            checkpoint_path=checkpoint_path,
            cli_args=cli_args,
            target_subjects=target_subjects,
            max_meshes_per_subject_eval=int(model_args.max_meshes_per_subject_eval),
            sweep_scenarios=sweep_scenarios,
            sigma_values=sigma_values,
        )
    )
    root_out_dir.mkdir(parents=True, exist_ok=True)

    clean_row = None
    if cli_args.include_clean_once:
        clean_row = _finalize_result(
            faceverse_base._evaluate_scenario(
                model=model,
                dataset=dataset,
                pair_ctx=pair_ctx,
                sample_cache=sample_cache,
                device=device,
                params=params,
                scenario=CLEAN_SPEC,
                scenario_index=0,
                base_seed=int(model_args.seed),
                chamfer_batch_pairs=int(cli_args.chamfer_batch_pairs),
                chamfer_use_icp=bool(cli_args.chamfer_use_icp),
                pairwise_icp_transforms=pairwise_icp_transforms,
                icp_workers=int(cli_args.icp_workers),
                icp_max_correspondence_distance=float(cli_args.icp_max_correspondence_distance),
                icp_max_iteration=int(cli_args.icp_max_iteration),
                show_chamfer_pair_progress=bool(cli_args.chamfer_pair_progress),
            )
        )

    scenario_index_lookup = {name: idx + 1 for idx, name in enumerate(sweep_scenarios)}
    summary_rows: List[dict] = []

    for scenario_name in sweep_scenarios:
        scenario_index = int(scenario_index_lookup[scenario_name])
        for sigma in sigma_values:
            scenario_spec = _build_scenario_spec(scenario_name, sigma)
            if clean_row is not None and _is_zero_sigma(sigma):
                result_row = _clone_clean_as_scenario(clean_row, scenario_name=scenario_name)
            else:
                result_row = _finalize_result(
                    faceverse_base._evaluate_scenario(
                        model=model,
                        dataset=dataset,
                        pair_ctx=pair_ctx,
                        sample_cache=sample_cache,
                        device=device,
                        params=params,
                        scenario=scenario_spec,
                        scenario_index=scenario_index,
                        base_seed=int(model_args.seed),
                        chamfer_batch_pairs=int(cli_args.chamfer_batch_pairs),
                        chamfer_use_icp=bool(cli_args.chamfer_use_icp),
                        pairwise_icp_transforms=pairwise_icp_transforms,
                        icp_workers=int(cli_args.icp_workers),
                        icp_max_correspondence_distance=float(cli_args.icp_max_correspondence_distance),
                        icp_max_iteration=int(cli_args.icp_max_iteration),
                        show_chamfer_pair_progress=bool(cli_args.chamfer_pair_progress),
                    )
                )

            label = f"{scenario_name}_sigma{_sigma_token(sigma)}"
            item_out_dir, json_path, csv_path, md_path = _resolve_item_paths(
                root_out_dir=root_out_dir,
                label=label,
                progressive_output_layout=bool(cli_args.progressive_output_layout),
            )

            rows_for_outputs: List[dict] = []
            scenarios_for_outputs: List[benchmark_base.ScenarioSpec] = []
            if clean_row is not None and not _is_zero_sigma(sigma):
                rows_for_outputs.append(dict(clean_row))
                scenarios_for_outputs.append(CLEAN_SPEC)
            rows_for_outputs.append(dict(result_row))
            scenarios_for_outputs.append(scenario_spec)

            payload = _build_payload(
                checkpoint_path=checkpoint_path,
                run_dir=run_dir,
                cli_args=cli_args,
                model_args=model_args,
                device=device,
                target_subjects=target_subjects,
                inference_filter_summary=inference_filter_summary,
                gt_filter_summary=gt_filter_summary,
                gt_dist_summary=gt_dist_summary,
                eval_plan=eval_plan,
                sample_records=sample_records,
                pair_ctx=pair_ctx,
                chamfer_cache_stats=chamfer_cache_stats,
                params=params,
                scenario_spec=scenario_spec,
                sigma=sigma,
                include_clean_once=bool(cli_args.include_clean_once),
                clean_row=clean_row,
                rows_for_outputs=rows_for_outputs,
            )
            _write_json(json_path, payload)
            faceverse_base._write_summary_csv(csv_path, rows=rows_for_outputs, params=params, scenarios=scenarios_for_outputs)
            faceverse_base._write_summary_md(md_path, rows=rows_for_outputs, params=params, scenarios=scenarios_for_outputs)

            summary_rows.append(
                _build_summary_row(
                    scenario_name=scenario_name,
                    sigma=sigma,
                    scenario_spec=scenario_spec,
                    result_row=result_row,
                    clean_row=clean_row,
                    params=params,
                    item_out_dir=item_out_dir,
                    json_path=json_path,
                    csv_path=csv_path,
                    md_path=md_path,
                )
            )

    summary_csv_path = root_out_dir / cli_args.summary_filename
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    _write_sweep_summary_csv(summary_csv_path, rows=summary_rows)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Root output dir: {root_out_dir}")
    print(f"GT dist matrix: {Path(cli_args.dist_npz).expanduser().resolve()}")
    print(f"Selected subjects: {len(target_subjects)}")
    print(f"Samples kept: {pair_ctx.n_samples}")
    print(f"Pairs kept: {pair_ctx.pair_count}")
    print(f"Aggregated sweep CSV: {summary_csv_path}")
    for row in summary_rows:
        print(
            f"[{row['scenario']} sigma={row['sigma']:.6f}] "
            f"latent_sp={row['latent_spearman']:.4f} "
            f"chamfer_sp={row['chamfer_spearman']:.4f} "
            f"delta={row['delta_spearman']:.4f}"
        )


if __name__ == "__main__":
    main()
