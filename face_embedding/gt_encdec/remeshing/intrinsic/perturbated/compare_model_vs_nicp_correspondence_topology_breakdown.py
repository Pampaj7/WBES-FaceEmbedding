from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

import compare_model_vs_chamfer_rankings as base
import registration_utils as reg_utils


ALLOWED_TOPOLOGY_PAIR_MODES = ("all", "cross_only")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare latent ranking on perturbed REMESH samples against Chamfer after "
            "rigid ICP + correspondence-based non-rigid ICP, broken down by topology-pair."
        )
    )
    p.add_argument("--model_path", type=str, required=True, help="Run directory or checkpoint path")
    p.add_argument(
        "--checkpoint_selector",
        type=str,
        default="best_by_xtopo_mesh_clean",
        choices=("best_by_auc", "best_by_clean", "latest", "best_by_xtopo_mesh_clean"),
        help="Checkpoint selection when --model_path points to a run directory",
    )
    p.add_argument("--config_json", type=str, default="", help="Optional explicit config.json path")
    p.add_argument("--out_dir", type=str, default="", help="Optional output directory")
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--data_dir", type=str, default="", help="Override dataset path")
    p.add_argument("--dist_npz", type=str, default="", help="Override GT distance matrix path")
    p.add_argument("--subject_split", type=str, default="eval", choices=("eval", "train", "all"))
    p.add_argument("--eval_fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=-1, help="Negative => use checkpoint/config seed")
    p.add_argument("--max_subjects", type=int, default=100, help="0 = all overlapping subjects")
    p.add_argument(
        "--topology_labels",
        type=str,
        default="crop,down8k,noisy,original,remesh,up60k",
        help="Optional comma-separated topology labels to keep",
    )
    p.add_argument(
        "--max_meshes_per_subject_eval",
        type=int,
        default=10,
        help="Max meshes per subject for this probe",
    )
    p.add_argument(
        "--preload_eval_samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preload selected meshes before evaluation",
    )
    p.add_argument("--preload_workers", type=int, default=4, help="Workers for sample preloading")

    p.add_argument("--pair_mode", type=str, default="cross_topology", choices=base.ALLOWED_PAIR_MODES)
    p.add_argument(
        "--scenarios",
        type=str,
        default="clean,jitter,translation,rotation,mixed",
        help="Comma-separated scenarios from: clean,jitter,translation,rotation,mixed",
    )
    p.add_argument(
        "--mesh_pair_level",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use mesh-pairs directly instead of aggregating within subject-pairs",
    )

    p.add_argument(
        "--base_sigma",
        type=float,
        default=-1.0,
        help="Default sigma for perturbation scenarios. Negative => use config sigma_max, else 0.05 fallback",
    )
    p.add_argument("--jitter_sigma", type=float, default=-1.0, help="Override jitter sigma")
    p.add_argument("--translation_sigma", type=float, default=-1.0, help="Override translation sigma")
    p.add_argument("--rotation_sigma", type=float, default=-1.0, help="Override rotation sigma")
    p.add_argument("--mixed_jitter_sigma", type=float, default=-1.0, help="Override mixed jitter sigma")
    p.add_argument("--mixed_translation_sigma", type=float, default=-1.0, help="Override mixed translation sigma")
    p.add_argument("--mixed_rotation_sigma", type=float, default=-1.0, help="Override mixed rotation sigma")

    p.add_argument("--rigid_rot_deg", type=float, default=-1.0, help="Override rigid rotation slope in degrees")
    p.add_argument("--rigid_trans_scale", type=float, default=-1.0, help="Override rigid translation slope")
    p.add_argument("--rigid_rot_deg_min", type=float, default=-1.0, help="Override rigid minimum angle")
    p.add_argument("--rigid_trans_scale_min", type=float, default=-1.0, help="Override rigid minimum translation")

    p.add_argument(
        "--topology_pair_mode",
        type=str,
        default="cross_only",
        choices=ALLOWED_TOPOLOGY_PAIR_MODES,
        help="Whether to include all topology-pairs or only cross-topology pairs",
    )
    p.add_argument(
        "--ordered_topology_pairs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat (topology_a, topology_b) as distinct from (topology_b, topology_a)",
    )
    p.add_argument(
        "--write_per_pair_outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write JSON/CSV/MD artifacts for each topology-pair in each scenario",
    )
    p.add_argument(
        "--save_pair_timings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-mesh-pair timing arrays (.npz) for each scenario",
    )

    p.add_argument("--registration_workers", type=int, default=8, help="CPU worker threads for pair registration")
    p.add_argument("--icp_points", type=int, default=128, help="Vertices sampled from each mesh for rigid ICP")
    p.add_argument(
        "--icp_max_correspondence_distance",
        type=float,
        default=0.05,
        help="Open3D rigid ICP max_correspondence_distance",
    )
    p.add_argument("--icp_max_iteration", type=int, default=20, help="Open3D rigid ICP max iterations")
    p.add_argument("--nicp_points", type=int, default=128, help="Vertices sampled from each mesh for non-rigid ICP")
    p.add_argument("--nicp_iterations", type=int, default=8, help="Number of non-rigid ICP iterations")
    p.add_argument("--nicp_neighbors", type=int, default=8, help="KNN neighbors for the nICP smoothness graph")
    p.add_argument(
        "--nicp_smoothness_lambda",
        type=float,
        default=2.0,
        help="Smoothness weight for the nICP displacement field",
    )
    p.add_argument(
        "--nicp_damping",
        type=float,
        default=1.0,
        help="Damping factor applied to the solved non-rigid displacement field",
    )
    return p.parse_args()


def _resolve_run_dir_and_checkpoint(model_path: str, selector: str) -> tuple[Path, Path]:
    requested = Path(model_path).expanduser().resolve()
    if requested.is_file():
        return base.infer_run_dir(requested), requested

    if requested.is_dir() and requested.name == "checkpoints":
        ckpt_dir = requested
        run_dir = requested.parent
    elif requested.is_dir():
        ckpt_dir = requested / "checkpoints"
        run_dir = requested
    else:
        raise FileNotFoundError(f"Model path not found: {requested}")

    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory found under: {run_dir}")

    if selector == "best_by_auc":
        checkpoint_path = ckpt_dir / "best_by_auc.pth"
    elif selector == "best_by_clean":
        checkpoint_path = ckpt_dir / "best_by_clean.pth"
    elif selector == "best_by_xtopo_mesh_clean":
        checkpoint_path = ckpt_dir / "best_by_xtopo_mesh_clean.pth"
    elif selector == "latest":
        epoch_paths = sorted(ckpt_dir.glob("epoch*.pth"))
        if not epoch_paths:
            raise FileNotFoundError(f"No epoch checkpoints found under: {ckpt_dir}")
        checkpoint_path = epoch_paths[-1]
    else:
        raise ValueError(f"Unsupported checkpoint selector: {selector}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return run_dir.resolve(), checkpoint_path.resolve()


def _pair_label(topology_a: str, topology_b: str, ordered: bool) -> str:
    if ordered:
        return f"{topology_a}__to__{topology_b}"
    return f"{topology_a}__and__{topology_b}"


def _topology_pair_key(topology_a: str, topology_b: str, ordered: bool) -> tuple[str, str]:
    if ordered:
        return str(topology_a), str(topology_b)
    return tuple(sorted((str(topology_a), str(topology_b))))


def _enumerate_topology_pairs(
    topology_labels: Sequence[str],
    topology_pair_mode: str,
    ordered: bool,
) -> List[tuple[str, str]]:
    labels = [str(label) for label in sorted(topology_labels)]
    pairs: List[tuple[str, str]] = []
    if ordered:
        for topology_a in labels:
            for topology_b in labels:
                if topology_pair_mode == "cross_only" and topology_a == topology_b:
                    continue
                pairs.append((topology_a, topology_b))
        return pairs

    for idx_a, topology_a in enumerate(labels):
        start_idx = idx_a + 1 if topology_pair_mode == "cross_only" else idx_a
        for idx_b in range(start_idx, len(labels)):
            pairs.append((topology_a, labels[idx_b]))
    return pairs


def _format_float(value: object) -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(value_f):
        return ""
    return f"{value_f:.6f}"


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=True)
        f.write("\n")


def _build_default_out_dir(
    run_dir: Path,
    checkpoint_path: Path,
    cli_args: argparse.Namespace,
    target_subjects: Sequence[str],
    scenarios: Sequence[base.ScenarioSpec],
) -> Path:
    slug = base.slugify_token(
        f"{checkpoint_path.stem}_split-{cli_args.subject_split}_pairs-{cli_args.pair_mode}_"
        f"topopairs-{cli_args.topology_pair_mode}_ordered-{int(bool(cli_args.ordered_topology_pairs))}_"
        f"subjects-{len(target_subjects)}_meshes-{int(cli_args.max_meshes_per_subject_eval)}_"
        f"scenarios-{'-'.join(spec.name for spec in scenarios)}_"
        f"icppts-{int(cli_args.icp_points)}_nicppts-{int(cli_args.nicp_points)}_"
        f"nicpiter-{int(cli_args.nicp_iterations)}"
    )
    return run_dir / "perturbation_ranking_vs_nicp_correspondence_topology_breakdown" / slug


def _subject_pair_key(subject_a: str, subject_b: str) -> tuple[str, str]:
    if str(subject_a) <= str(subject_b):
        return str(subject_a), str(subject_b)
    return str(subject_b), str(subject_a)


def _group_subject_pair_members(
    subject_ids_a: Sequence[str],
    subject_ids_b: Sequence[str],
) -> tuple[List[tuple[str, str]], List[np.ndarray]]:
    group_lookup: Dict[tuple[str, str], int] = {}
    group_keys: List[tuple[str, str]] = []
    group_members: List[List[int]] = []

    for local_idx, (subject_a, subject_b) in enumerate(zip(subject_ids_a, subject_ids_b)):
        key = _subject_pair_key(str(subject_a), str(subject_b))
        group_idx = group_lookup.get(key)
        if group_idx is None:
            group_idx = len(group_keys)
            group_lookup[key] = group_idx
            group_keys.append(key)
            group_members.append([])
        group_members[group_idx].append(int(local_idx))

    return group_keys, [np.asarray(members, dtype=np.int64) for members in group_members]


def _aggregate_grouped_values(values: np.ndarray, group_members: Sequence[np.ndarray]) -> np.ndarray:
    aggregated = np.full((len(group_members),), float("nan"), dtype=np.float64)
    for group_idx, member_indices in enumerate(group_members):
        member_values = np.asarray(values[np.asarray(member_indices, dtype=np.int64)], dtype=np.float64)
        member_values = member_values[np.isfinite(member_values)]
        if member_values.size == 0:
            continue
        aggregated[group_idx] = float(member_values.mean())
    return aggregated


def _finite_stat(values: np.ndarray, reducer: str) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if reducer == "mean":
        return float(arr.mean())
    if reducer == "median":
        return float(np.median(arr))
    raise ValueError(f"Unsupported reducer: {reducer}")


def _summarize_subset(
    *,
    scenario: str,
    topology_a: str,
    topology_b: str,
    ordered_pair_label: str,
    ordered: bool,
    mesh_pair_indices: np.ndarray,
    pair_subject_ids_a: np.ndarray,
    pair_subject_ids_b: np.ndarray,
    pair_gt_mesh_values: np.ndarray,
    latent_mesh_values: np.ndarray,
    nicp_correspondence_mesh_values: np.ndarray,
    timing_arrays: Dict[str, np.ndarray],
    mesh_pair_level: bool,
) -> dict:
    selected_indices = np.asarray(mesh_pair_indices, dtype=np.int64)
    n_mesh_pairs = int(selected_indices.size)
    if n_mesh_pairs == 0:
        return {
            "scenario": str(scenario),
            "topology_a": str(topology_a),
            "topology_b": str(topology_b),
            "ordered_pair_label": str(ordered_pair_label),
            "ordered_topology_pairs": bool(ordered),
            "mesh_pair_level": bool(mesh_pair_level),
            "latent_spearman": float("nan"),
            "latent_pearson": float("nan"),
            "nicp_correspondence_spearman": float("nan"),
            "nicp_correspondence_pearson": float("nan"),
            "delta_spearman": float("nan"),
            "delta_pearson": float("nan"),
            "model_beats_nicp_correspondence": False,
            "n_subjects": 0,
            "n_subject_pairs": 0,
            "n_mesh_pairs": 0,
            "n_observations": 0,
            "rigid_icp_seconds_mean": float("nan"),
            "rigid_icp_seconds_median": float("nan"),
            "nonrigid_nicp_seconds_mean": float("nan"),
            "nonrigid_nicp_seconds_median": float("nan"),
            "metric_seconds_mean": float("nan"),
            "metric_seconds_median": float("nan"),
            "total_pair_seconds_mean": float("nan"),
            "total_pair_seconds_median": float("nan"),
        }

    subject_ids_a = np.asarray(pair_subject_ids_a[selected_indices], dtype=object)
    subject_ids_b = np.asarray(pair_subject_ids_b[selected_indices], dtype=object)
    gt_mesh_values = np.asarray(pair_gt_mesh_values[selected_indices], dtype=np.float64)
    latent_mesh = np.asarray(latent_mesh_values[selected_indices], dtype=np.float64)
    nicp_correspondence_mesh = np.asarray(nicp_correspondence_mesh_values[selected_indices], dtype=np.float64)

    if mesh_pair_level:
        gt_values = gt_mesh_values
        latent_values = latent_mesh
        nicp_correspondence_values = nicp_correspondence_mesh
        subject_pair_count = int(len({_subject_pair_key(str(a), str(b)) for a, b in zip(subject_ids_a, subject_ids_b)}))
    else:
        subject_pair_keys, group_members = _group_subject_pair_members(subject_ids_a=subject_ids_a, subject_ids_b=subject_ids_b)
        gt_values = np.asarray(
            [gt_mesh_values[int(group_members[group_idx][0])] for group_idx in range(len(group_members))],
            dtype=np.float64,
        )
        latent_values = _aggregate_grouped_values(latent_mesh, group_members=group_members)
        nicp_correspondence_values = _aggregate_grouped_values(nicp_correspondence_mesh, group_members=group_members)
        subject_pair_count = int(len(subject_pair_keys))

    latent_spearman = float(base.spearman_corr(gt_values, latent_values))
    latent_pearson = float(base.pearson_corr(gt_values, latent_values))
    nicp_correspondence_spearman = float(base.spearman_corr(gt_values, nicp_correspondence_values))
    nicp_correspondence_pearson = float(base.pearson_corr(gt_values, nicp_correspondence_values))
    delta_spearman = float(latent_spearman - nicp_correspondence_spearman)
    delta_pearson = float(latent_pearson - nicp_correspondence_pearson)
    n_subjects = int(len({str(sid) for sid in subject_ids_a.tolist() + subject_ids_b.tolist()}))

    row = {
        "scenario": str(scenario),
        "topology_a": str(topology_a),
        "topology_b": str(topology_b),
        "ordered_pair_label": str(ordered_pair_label),
        "ordered_topology_pairs": bool(ordered),
        "mesh_pair_level": bool(mesh_pair_level),
        "latent_spearman": latent_spearman,
        "latent_pearson": latent_pearson,
        "nicp_correspondence_spearman": nicp_correspondence_spearman,
        "nicp_correspondence_pearson": nicp_correspondence_pearson,
        "delta_spearman": delta_spearman,
        "delta_pearson": delta_pearson,
        "model_beats_nicp_correspondence": bool(latent_spearman > nicp_correspondence_spearman),
        "n_subjects": n_subjects,
        "n_subject_pairs": int(subject_pair_count),
        "n_mesh_pairs": int(n_mesh_pairs),
        "n_observations": int(gt_values.size),
    }
    for key in (
        "rigid_icp_seconds",
        "nonrigid_refine_seconds",
        "metric_seconds",
        "total_pair_seconds",
    ):
        values = np.asarray(timing_arrays[key][selected_indices], dtype=np.float64)
        mapped_key = {
            "rigid_icp_seconds": "rigid_icp_seconds",
            "nonrigid_refine_seconds": "nonrigid_nicp_seconds",
            "metric_seconds": "metric_seconds",
            "total_pair_seconds": "total_pair_seconds",
        }[key]
        row[f"{mapped_key}_mean"] = _finite_stat(values, reducer="mean")
        row[f"{mapped_key}_median"] = _finite_stat(values, reducer="median")
    return row


def _summary_header() -> List[str]:
    return [
        "scenario",
        "topology_a",
        "topology_b",
        "ordered_pair_label",
        "ordered_topology_pairs",
        "mesh_pair_level",
        "latent_spearman",
        "nicp_correspondence_spearman",
        "delta_spearman",
        "latent_pearson",
        "nicp_correspondence_pearson",
        "delta_pearson",
        "model_beats_nicp_correspondence",
        "n_subjects",
        "n_subject_pairs",
        "n_mesh_pairs",
        "n_observations",
        "rigid_icp_seconds_mean",
        "rigid_icp_seconds_median",
        "nonrigid_nicp_seconds_mean",
        "nonrigid_nicp_seconds_median",
        "metric_seconds_mean",
        "metric_seconds_median",
        "total_pair_seconds_mean",
        "total_pair_seconds_median",
        "output_dir",
        "output_json",
        "output_csv",
        "output_md",
    ]


def _write_summary_csv(path: Path, rows: Sequence[dict]) -> None:
    header = _summary_header()
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "scenario": row["scenario"],
                    "topology_a": row.get("topology_a", ""),
                    "topology_b": row.get("topology_b", ""),
                    "ordered_pair_label": row["ordered_pair_label"],
                    "ordered_topology_pairs": int(bool(row["ordered_topology_pairs"])),
                    "mesh_pair_level": int(bool(row["mesh_pair_level"])),
                    "latent_spearman": _format_float(row["latent_spearman"]),
                    "nicp_correspondence_spearman": _format_float(row["nicp_correspondence_spearman"]),
                    "delta_spearman": _format_float(row["delta_spearman"]),
                    "latent_pearson": _format_float(row["latent_pearson"]),
                    "nicp_correspondence_pearson": _format_float(row["nicp_correspondence_pearson"]),
                    "delta_pearson": _format_float(row["delta_pearson"]),
                    "model_beats_nicp_correspondence": int(bool(row["model_beats_nicp_correspondence"])),
                    "n_subjects": int(row["n_subjects"]),
                    "n_subject_pairs": int(row["n_subject_pairs"]),
                    "n_mesh_pairs": int(row["n_mesh_pairs"]),
                    "n_observations": int(row["n_observations"]),
                    "rigid_icp_seconds_mean": _format_float(row["rigid_icp_seconds_mean"]),
                    "rigid_icp_seconds_median": _format_float(row["rigid_icp_seconds_median"]),
                    "nonrigid_nicp_seconds_mean": _format_float(row["nonrigid_nicp_seconds_mean"]),
                    "nonrigid_nicp_seconds_median": _format_float(row["nonrigid_nicp_seconds_median"]),
                    "metric_seconds_mean": _format_float(row["metric_seconds_mean"]),
                    "metric_seconds_median": _format_float(row["metric_seconds_median"]),
                    "total_pair_seconds_mean": _format_float(row["total_pair_seconds_mean"]),
                    "total_pair_seconds_median": _format_float(row["total_pair_seconds_median"]),
                    "output_dir": row.get("output_dir", ""),
                    "output_json": row.get("output_json", ""),
                    "output_csv": row.get("output_csv", ""),
                    "output_md": row.get("output_md", ""),
                }
            )


def _write_pair_summary_md(path: Path, row: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# nICP Correspondence Topology Pair Breakdown\n\n")
        f.write(
            "| Scenario | Pair Label | Mesh-level | Lat Sp | nICP Corr Sp | Delta Sp | "
            "Lat Pe | nICP Corr Pe | Delta Pe | Rigid mean s | nICP mean s | Total mean s |\n"
        )
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        f.write(
            "| "
            + " | ".join(
                [
                    str(row["scenario"]),
                    str(row["ordered_pair_label"]),
                    "yes" if row["mesh_pair_level"] else "no",
                    _format_float(row["latent_spearman"]),
                    _format_float(row["nicp_correspondence_spearman"]),
                    _format_float(row["delta_spearman"]),
                    _format_float(row["latent_pearson"]),
                    _format_float(row["nicp_correspondence_pearson"]),
                    _format_float(row["delta_pearson"]),
                    _format_float(row["rigid_icp_seconds_mean"]),
                    _format_float(row["nonrigid_nicp_seconds_mean"]),
                    _format_float(row["total_pair_seconds_mean"]),
                ]
            )
            + " |\n"
        )


def _build_registration_index_cache(
    *,
    dataset: base.GTReadyDataset,
    sample_records: Sequence[object],
    sample_cache: Dict[int, Dict[str, torch.Tensor]] | None,
    icp_points: int,
    nicp_points: int,
    base_seed: int,
) -> tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    icp_cache: Dict[int, np.ndarray] = {}
    nicp_cache: Dict[int, np.ndarray] = {}
    for record in sample_records:
        dataset_idx = int(record.dataset_idx)
        if dataset_idx in icp_cache:
            continue
        sample = sample_cache[dataset_idx] if sample_cache is not None else dataset[dataset_idx]
        n_verts = int(sample["verts"].shape[0])
        icp_cache[dataset_idx] = reg_utils.build_sample_vertex_indices(
            vertex_count=n_verts,
            n_points=int(icp_points),
            seed=int(base_seed) * 1_000_003 + dataset_idx * 97 + 11,
        )
        nicp_cache[dataset_idx] = reg_utils.build_sample_vertex_indices(
            vertex_count=n_verts,
            n_points=int(nicp_points),
            seed=int(base_seed) * 1_000_003 + dataset_idx * 97 + 23,
        )
    return icp_cache, nicp_cache


def _build_point_sets_for_scenario(
    *,
    vertex_sets: Sequence[torch.Tensor],
    sample_records: Sequence[object],
    icp_index_cache: Dict[int, np.ndarray],
    nicp_index_cache: Dict[int, np.ndarray],
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    icp_point_sets: List[np.ndarray] = []
    nicp_point_sets: List[np.ndarray] = []
    for local_idx, record in enumerate(sample_records):
        V = vertex_sets[local_idx]
        dataset_idx = int(record.dataset_idx)
        icp_point_sets.append(reg_utils.extract_point_subset(V, indices=icp_index_cache[dataset_idx]))
        nicp_point_sets.append(reg_utils.extract_point_subset(V, indices=nicp_index_cache[dataset_idx]))
    return icp_point_sets, nicp_point_sets


@torch.no_grad()
def _collect_scenario_latents_and_vertices(
    *,
    model: torch.nn.Module,
    dataset: base.GTReadyDataset,
    sample_records: Sequence[object],
    sample_cache: Dict[int, Dict[str, torch.Tensor]] | None,
    device: torch.device,
    params: base.PerturbationParams,
    scenario: base.ScenarioSpec,
    scenario_index: int,
    base_seed: int,
    pair_ctx,
) -> tuple[np.ndarray, List[torch.Tensor]]:
    latent_vectors: List[torch.Tensor] = []
    vertex_sets: List[torch.Tensor] = []
    sample_total = len(sample_records)
    sample_log_step = base._progress_log_step(sample_total)

    for sample_index, record in enumerate(sample_records, start=1):
        sample = sample_cache[int(record.dataset_idx)] if sample_cache is not None else dataset[int(record.dataset_idx)]
        sample_d = base.sample_to_device(sample, device=device)
        pert_seed = base._scenario_seed(
            base_seed=base_seed,
            scenario_index=scenario_index,
            dataset_idx=int(record.dataset_idx),
        )
        V_in = base._apply_scenario(V=sample_d["verts"], params=params, scenario=scenario, seed=pert_seed)
        z, _ = base.forward_model(
            model=model,
            sample_dict=sample_d,
            V_in=V_in,
            return_gate_info=False,
            add_noise=False,
        )
        latent_vectors.append(z.squeeze(0))
        vertex_sets.append(V_in.contiguous())
        if sample_index == 1 or sample_index == sample_total or sample_index % sample_log_step == 0:
            print(
                f"Scenario {scenario.name}: encoded {sample_index}/{sample_total} samples",
                flush=True,
            )

    Z = torch.stack(latent_vectors, dim=0)
    latent_mesh_values = (
        torch.linalg.vector_norm(
            Z.index_select(0, pair_ctx.pair_i) - Z.index_select(0, pair_ctx.pair_j),
            dim=1,
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=False)
    )
    return latent_mesh_values, vertex_sets


def main() -> None:
    cli_args = parse_args()

    run_dir, checkpoint_path = _resolve_run_dir_and_checkpoint(
        cli_args.model_path,
        selector=str(cli_args.checkpoint_selector),
    )
    base_args = base.merge_run_args(checkpoint_path, explicit_config_json=cli_args.config_json)
    model_args = base._resolve_runtime_args(cli_args, base_args)
    params = base.PerturbationParams.from_namespace(model_args)

    base_sigma = float(cli_args.base_sigma)
    if base_sigma < 0.0:
        base_sigma = float(getattr(model_args, "sigma_max", 0.05))
    scenarios = base._parse_scenarios(cli_args.scenarios, base_sigma=base_sigma, cli_args=cli_args)

    base.seed_everything(int(model_args.seed))
    device = base._resolve_device(cli_args.device)

    dataset = base.GTReadyDataset(model_args.data_dir)
    gt_matrix, gt_name_to_idx = base.load_gt_distance_matrix(
        model_args.dist_npz,
        subject_re=base.SUBJECT_RE_ANY,
        dtype=np.float64,
    )
    subject_map = base.build_subject_map(dataset.files, subject_re=base.SUBJECT_RE_ANY)
    subjects = sorted([sid for sid in subject_map.keys() if sid in gt_name_to_idx])
    _, _, target_subjects = base._select_subject_subset(
        subjects=subjects,
        subject_split=str(cli_args.subject_split),
        eval_fraction=float(cli_args.eval_fraction),
        seed=int(model_args.seed),
        max_subjects=int(cli_args.max_subjects),
    )

    eval_plan = base.build_eval_plan(
        subj_map=subject_map,
        eval_subjects=target_subjects,
        max_meshes_per_subject_eval=int(model_args.max_meshes_per_subject_eval),
        seed=int(model_args.seed),
    )
    sample_cache = (
        base.preload_eval_samples(
            dataset=dataset,
            eval_plan=eval_plan,
            workers=int(cli_args.preload_workers),
        )
        if cli_args.preload_eval_samples
        else None
    )
    sample_records = base.build_sample_eval_records(
        dataset=dataset,
        eval_plan=eval_plan,
        eval_subjects=target_subjects,
        sample_cache=sample_cache,
    )
    requested_topology_labels = base._parse_topology_labels(cli_args.topology_labels)
    if requested_topology_labels:
        sample_records = base._filter_sample_records_by_topology_labels(
            sample_records=sample_records,
            topology_labels=requested_topology_labels,
        )
    active_eval_plan = base._eval_plan_from_sample_records(sample_records)
    active_subjects = sorted(active_eval_plan.keys())
    print(
        f"Prepared nICP-correspondence subset: subjects={len(active_subjects)} "
        f"samples={len(sample_records)} "
        f"topologies={sorted({rec.topology_label for rec in sample_records})}",
        flush=True,
    )

    pair_ctx = base.build_pair_eval_context(
        sample_records=sample_records,
        name_to_idx=gt_name_to_idx,
        gt_matrix=gt_matrix,
        device=device,
        pair_mode=str(cli_args.pair_mode),
        aggregation_level="mesh_pair",
    )
    if pair_ctx.mesh_pair_count <= 0:
        raise RuntimeError("No valid mesh-pairs found for the selected subset and pair mode")
    print(
        f"Prepared nICP-correspondence pair context: subject_pairs={pair_ctx.subject_pair_count} "
        f"mesh_pairs={pair_ctx.mesh_pair_count}",
        flush=True,
    )

    icp_index_cache, nicp_index_cache = _build_registration_index_cache(
        dataset=dataset,
        sample_records=pair_ctx.sample_records,
        sample_cache=sample_cache,
        icp_points=int(cli_args.icp_points),
        nicp_points=int(cli_args.nicp_points),
        base_seed=int(model_args.seed),
    )

    model = base.build_model(args=model_args, device=device)
    checkpoint_bundle = base.load_checkpoint_bundle(checkpoint_path)
    model.load_state_dict(checkpoint_bundle["state_dict"], strict=True)
    model.eval()

    sample_topology_labels = np.asarray([rec.topology_label for rec in pair_ctx.sample_records], dtype=object)
    sample_subject_ids = np.asarray([rec.subject_id for rec in pair_ctx.sample_records], dtype=object)
    sample_gt_idx = np.asarray([gt_name_to_idx[rec.subject_id] for rec in pair_ctx.sample_records], dtype=np.int64)
    pair_topology_a = sample_topology_labels[pair_ctx.pair_i_cpu]
    pair_topology_b = sample_topology_labels[pair_ctx.pair_j_cpu]
    pair_subject_ids_a = sample_subject_ids[pair_ctx.pair_i_cpu]
    pair_subject_ids_b = sample_subject_ids[pair_ctx.pair_j_cpu]
    pair_gt_mesh_values = np.asarray(
        gt_matrix[sample_gt_idx[pair_ctx.pair_i_cpu], sample_gt_idx[pair_ctx.pair_j_cpu]],
        dtype=np.float64,
    )

    topology_pair_members: Dict[tuple[str, str], List[int]] = {}
    for mesh_pair_idx in range(int(pair_ctx.mesh_pair_count)):
        key = _topology_pair_key(
            topology_a=str(pair_topology_a[mesh_pair_idx]),
            topology_b=str(pair_topology_b[mesh_pair_idx]),
            ordered=bool(cli_args.ordered_topology_pairs),
        )
        topology_pair_members.setdefault(key, []).append(int(mesh_pair_idx))

    topology_pairs = _enumerate_topology_pairs(
        topology_labels=pair_ctx.topology_labels,
        topology_pair_mode=str(cli_args.topology_pair_mode),
        ordered=bool(cli_args.ordered_topology_pairs),
    )

    root_out_dir = (
        Path(cli_args.out_dir).expanduser().resolve()
        if cli_args.out_dir
        else _build_default_out_dir(
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            cli_args=cli_args,
            target_subjects=target_subjects,
            scenarios=scenarios,
        )
    )
    root_out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "checkpoint": str(checkpoint_path),
        "run_dir": str(run_dir),
        "checkpoint_selector": str(cli_args.checkpoint_selector),
        "device": str(device),
        "data_dir": str(model_args.data_dir),
        "dist_npz": str(model_args.dist_npz),
        "subject_split": str(cli_args.subject_split),
        "eval_fraction": float(cli_args.eval_fraction),
        "seed": int(model_args.seed),
        "selected_subjects": list(active_subjects),
        "eval_plan_summary": base.summarize_eval_plan(active_eval_plan),
        "sample_eval_summary": base.summarize_sample_records(sample_records),
        "pair_context": {
            "n_subjects": int(pair_ctx.n_subjects),
            "n_samples": int(pair_ctx.n_samples),
            "n_pairs": int(pair_ctx.pair_count),
            "n_mesh_pairs": int(pair_ctx.mesh_pair_count),
            "n_subject_pairs": int(pair_ctx.subject_pair_count),
            "n_topology_labels": int(pair_ctx.n_topology_labels),
        },
        "registration": {
            "icp_points": int(cli_args.icp_points),
            "icp_max_correspondence_distance": float(cli_args.icp_max_correspondence_distance),
            "icp_max_iteration": int(cli_args.icp_max_iteration),
            "registration_workers": int(cli_args.registration_workers),
            "nicp_points": int(cli_args.nicp_points),
            "nicp_iterations": int(cli_args.nicp_iterations),
            "nicp_neighbors": int(cli_args.nicp_neighbors),
            "nicp_smoothness_lambda": float(cli_args.nicp_smoothness_lambda),
            "nicp_damping": float(cli_args.nicp_damping),
        },
        "scenarios": [spec.name for spec in scenarios],
    }
    _write_json(root_out_dir / "metadata.json", metadata)

    overall_rows: List[dict] = []
    overall_scenario_rows: List[dict] = []
    for scenario_index, scenario in enumerate(scenarios):
        print(
            f"Starting nICP-correspondence scenario {scenario_index + 1}/{len(scenarios)}: {scenario.name}",
            flush=True,
        )
        latent_mesh_values, vertex_sets = _collect_scenario_latents_and_vertices(
            model=model,
            dataset=dataset,
            sample_records=pair_ctx.sample_records,
            sample_cache=sample_cache,
            device=device,
            params=params,
            scenario=scenario,
            scenario_index=scenario_index,
            base_seed=int(model_args.seed),
            pair_ctx=pair_ctx,
        )
        icp_point_sets, nicp_point_sets = _build_point_sets_for_scenario(
            vertex_sets=vertex_sets,
            sample_records=pair_ctx.sample_records,
            icp_index_cache=icp_index_cache,
            nicp_index_cache=nicp_index_cache,
        )
        print(
            f"Scenario {scenario.name}: computing nICP-correspondence metric over {pair_ctx.mesh_pair_count} mesh pairs "
            f"(nicp_points={int(cli_args.nicp_points)})",
            flush=True,
        )
        nicp_correspondence_mesh_values, timing_arrays = reg_utils.compute_pairwise_nicp_correspondence_values(
            icp_point_sets=icp_point_sets,
            nicp_point_sets=nicp_point_sets,
            pair_i=pair_ctx.pair_i_cpu,
            pair_j=pair_ctx.pair_j_cpu,
            registration_workers=int(cli_args.registration_workers),
            icp_max_correspondence_distance=float(cli_args.icp_max_correspondence_distance),
            icp_max_iteration=int(cli_args.icp_max_iteration),
            nicp_iterations=int(cli_args.nicp_iterations),
            nicp_neighbors=int(cli_args.nicp_neighbors),
            nicp_smoothness_lambda=float(cli_args.nicp_smoothness_lambda),
            nicp_damping=float(cli_args.nicp_damping),
            progress_desc=f"{scenario.name} nicp correspondence pairs",
            show_progress=True,
        )
        print(
            f"Scenario {scenario.name}: finished nICP-correspondence metric over {pair_ctx.mesh_pair_count} mesh pairs",
            flush=True,
        )

        scenario_dir = root_out_dir / scenario.name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        if cli_args.save_pair_timings:
            np.savez(
                scenario_dir / "pair_timings.npz",
                pair_i=pair_ctx.pair_i_cpu.astype(np.int64, copy=False),
                pair_j=pair_ctx.pair_j_cpu.astype(np.int64, copy=False),
                rigid_icp_seconds=np.asarray(timing_arrays["rigid_icp_seconds"], dtype=np.float64),
                nonrigid_nicp_seconds=np.asarray(timing_arrays["nonrigid_refine_seconds"], dtype=np.float64),
                metric_seconds=np.asarray(timing_arrays["metric_seconds"], dtype=np.float64),
                total_pair_seconds=np.asarray(timing_arrays["total_pair_seconds"], dtype=np.float64),
            )

        scenario_overall = _summarize_subset(
            scenario=scenario.name,
            topology_a="all",
            topology_b="all",
            ordered_pair_label="all_pairs",
            ordered=False,
            mesh_pair_indices=np.arange(int(pair_ctx.mesh_pair_count), dtype=np.int64),
            pair_subject_ids_a=pair_subject_ids_a,
            pair_subject_ids_b=pair_subject_ids_b,
            pair_gt_mesh_values=pair_gt_mesh_values,
            latent_mesh_values=latent_mesh_values,
            nicp_correspondence_mesh_values=nicp_correspondence_mesh_values,
            timing_arrays=timing_arrays,
            mesh_pair_level=bool(cli_args.mesh_pair_level),
        )
        scenario_overall["output_dir"] = str(scenario_dir)
        scenario_overall["output_json"] = ""
        scenario_overall["output_csv"] = ""
        scenario_overall["output_md"] = ""
        overall_scenario_rows.append(scenario_overall)

        scenario_rows: List[dict] = []
        for topology_a, topology_b in topology_pairs:
            pair_key = _topology_pair_key(
                topology_a=topology_a,
                topology_b=topology_b,
                ordered=bool(cli_args.ordered_topology_pairs),
            )
            ordered_pair_label = _pair_label(
                topology_a=topology_a,
                topology_b=topology_b,
                ordered=bool(cli_args.ordered_topology_pairs),
            )
            mesh_pair_indices = np.asarray(topology_pair_members.get(pair_key, []), dtype=np.int64)
            row = _summarize_subset(
                scenario=scenario.name,
                topology_a=topology_a,
                topology_b=topology_b,
                ordered_pair_label=ordered_pair_label,
                ordered=bool(cli_args.ordered_topology_pairs),
                mesh_pair_indices=mesh_pair_indices,
                pair_subject_ids_a=pair_subject_ids_a,
                pair_subject_ids_b=pair_subject_ids_b,
                pair_gt_mesh_values=pair_gt_mesh_values,
                latent_mesh_values=latent_mesh_values,
                nicp_correspondence_mesh_values=nicp_correspondence_mesh_values,
                timing_arrays=timing_arrays,
                mesh_pair_level=bool(cli_args.mesh_pair_level),
            )
            if cli_args.write_per_pair_outputs:
                pair_dir = scenario_dir / base.slugify_token(ordered_pair_label)
                pair_dir.mkdir(parents=True, exist_ok=True)
                json_path = pair_dir / "ranking_summary.json"
                csv_path = pair_dir / "ranking_summary.csv"
                md_path = pair_dir / "ranking_summary.md"
                payload = {
                    "metadata": metadata,
                    "scenario": scenario.name,
                    "scenario_description": base._describe_scenario(scenario, params=params),
                    "row": dict(row),
                }
                _write_json(json_path, payload)
                _write_summary_csv(csv_path, [dict(row, output_dir="", output_json="", output_csv="", output_md="")])
                _write_pair_summary_md(md_path, row=row)
                row["output_dir"] = str(pair_dir)
                row["output_json"] = str(json_path)
                row["output_csv"] = str(csv_path)
                row["output_md"] = str(md_path)
            else:
                row["output_dir"] = ""
                row["output_json"] = ""
                row["output_csv"] = ""
                row["output_md"] = ""
            scenario_rows.append(row)
            overall_rows.append(row)

        _write_summary_csv(scenario_dir / "topology_breakdown_summary.csv", scenario_rows)
        _write_summary_csv(scenario_dir / "scenario_summary.csv", [scenario_overall])
        _write_json(
            scenario_dir / "scenario_summary.json",
            {
                "scenario": scenario.name,
                "scenario_description": base._describe_scenario(scenario, params=params),
                "overall": scenario_overall,
            },
        )
        print(
            f"Finished scenario {scenario.name}: "
            f"latent_sp={scenario_overall['latent_spearman']:.4f} "
            f"nicp_corr_sp={scenario_overall['nicp_correspondence_spearman']:.4f} "
            f"delta={scenario_overall['delta_spearman']:.4f} "
            f"total_pair_s_mean={_format_float(scenario_overall['total_pair_seconds_mean']) or 'nan'}",
            flush=True,
        )

    _write_summary_csv(root_out_dir / "overall_topology_breakdown_summary.csv", overall_rows)
    _write_summary_csv(root_out_dir / "overall_scenario_summary.csv", overall_scenario_rows)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Root output dir: {root_out_dir}")
    print(f"Selected subjects: {len(target_subjects)}")
    print(f"Samples kept: {pair_ctx.n_samples}")
    print(f"Mesh-pairs kept by pair_mode: {pair_ctx.mesh_pair_count}")
    print(f"Overall scenario summary: {root_out_dir / 'overall_scenario_summary.csv'}")
    print(f"Overall topology summary: {root_out_dir / 'overall_topology_breakdown_summary.csv'}")


if __name__ == "__main__":
    main()
