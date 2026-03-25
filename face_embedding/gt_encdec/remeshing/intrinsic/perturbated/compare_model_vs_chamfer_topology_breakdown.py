from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

import compare_model_vs_chamfer_rankings as base


ALLOWED_TOPOLOGY_PAIR_MODES = ("all", "cross_only")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Break down the model-vs-Chamfer ranking benchmark by topology-pair "
            "without modifying the existing ranking scripts."
        )
    )
    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Run directory or explicit checkpoint path for the model to evaluate",
    )
    p.add_argument(
        "--checkpoint_selector",
        type=str,
        default="best_by_auc",
        choices=("best_by_auc", "best_by_clean", "latest"),
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
    p.add_argument("--max_subjects", type=int, default=16, help="0 = all overlapping subjects")
    p.add_argument(
        "--max_meshes_per_subject_eval",
        type=int,
        default=2,
        help="Max meshes per subject for this ranking probe",
    )
    p.add_argument(
        "--preload_eval_samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preload selected meshes before evaluation",
    )
    p.add_argument("--preload_workers", type=int, default=4, help="Workers for sample preloading")

    p.add_argument("--pair_mode", type=str, default="cross_topology", choices=base.ALLOWED_PAIR_MODES)
    p.add_argument("--rigid_rot_deg", type=float, default=-1.0, help="Override rigid rotation slope in degrees")
    p.add_argument("--rigid_trans_scale", type=float, default=-1.0, help="Override rigid translation slope")
    p.add_argument("--rigid_rot_deg_min", type=float, default=-1.0, help="Override rigid minimum angle")
    p.add_argument("--rigid_trans_scale_min", type=float, default=-1.0, help="Override rigid minimum translation")
    p.add_argument("--chamfer_batch_pairs", type=int, default=64, help="Pair batch size for Chamfer evaluation")
    p.add_argument(
        "--chamfer_cache_verts",
        type=str,
        default="auto",
        choices=("off", "auto", "force"),
        help="Cache Chamfer vertices on eval device when beneficial",
    )
    p.add_argument(
        "--chamfer_cache_verts_max_mb",
        type=float,
        default=256.0,
        help="Device-cache limit in auto mode",
    )

    p.add_argument(
        "--topology_pair_mode",
        type=str,
        default="all",
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
        "--summary_filename",
        type=str,
        default="topology_breakdown_summary.csv",
        help="Filename for the aggregated topology-breakdown CSV",
    )
    p.add_argument(
        "--write_per_pair_outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write JSON/CSV/MD artifacts for each topology-pair",
    )
    p.add_argument(
        "--mesh_pair_level",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use each mesh-pair as an observation instead of aggregating within topology-pair "
            "at subject-pair level."
        ),
    )
    return p.parse_args()


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
            topology_b = labels[idx_b]
            pairs.append((topology_a, topology_b))
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
    evaluation_level: str,
) -> Path:
    slug = base.slugify_token(
        f"{checkpoint_path.stem}_split-{cli_args.subject_split}_pairs-{cli_args.pair_mode}_"
        f"topopairs-{cli_args.topology_pair_mode}_ordered-{int(bool(cli_args.ordered_topology_pairs))}_"
        f"subjects-{len(target_subjects)}_meshes-{int(cli_args.max_meshes_per_subject_eval)}_"
        f"level-{evaluation_level}"
    )
    return run_dir / "perturbation_ranking_vs_chamfer_topology_breakdown" / slug


def _collect_clean_embeddings_and_vertices(
    model: torch.nn.Module,
    dataset: base.GTReadyDataset,
    sample_records: Sequence[object],
    sample_cache: Dict[int, Dict[str, torch.Tensor]] | None,
    device: torch.device,
) -> tuple[torch.Tensor, List[torch.Tensor]]:
    latent_vectors: List[torch.Tensor] = []
    vertex_sets: List[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for record in sample_records:
            sample = sample_cache[int(record.dataset_idx)] if sample_cache is not None else dataset[int(record.dataset_idx)]
            sample_d = base.sample_to_device(sample, device=device)
            V_in = sample_d["verts"]
            z, _ = base.forward_model(
                model=model,
                sample_dict=sample_d,
                V_in=V_in,
                return_gate_info=False,
                add_noise=False,
            )
            latent_vectors.append(z.squeeze(0))
            vertex_sets.append(V_in.contiguous())

    return torch.stack(latent_vectors, dim=0), vertex_sets


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


def _summarize_topology_pair(
    *,
    topology_a: str,
    topology_b: str,
    ordered_pair_label: str,
    ordered: bool,
    mesh_pair_indices: np.ndarray,
    pair_topology_a: np.ndarray,
    pair_topology_b: np.ndarray,
    pair_subject_ids_a: np.ndarray,
    pair_subject_ids_b: np.ndarray,
    pair_gt_mesh_values: np.ndarray,
    latent_mesh_values: np.ndarray,
    chamfer_mesh_values: np.ndarray,
    mesh_pair_level: bool,
) -> dict:
    selected_indices = np.asarray(mesh_pair_indices, dtype=np.int64)
    n_mesh_pairs = int(selected_indices.size)
    if n_mesh_pairs == 0:
        return {
            "topology_a": str(topology_a),
            "topology_b": str(topology_b),
            "ordered_pair_label": str(ordered_pair_label),
            "ordered_topology_pairs": bool(ordered),
            "mesh_pair_level": bool(mesh_pair_level),
            "latent_spearman": float("nan"),
            "latent_pearson": float("nan"),
            "chamfer_spearman": float("nan"),
            "chamfer_pearson": float("nan"),
            "delta_spearman": float("nan"),
            "delta_pearson": float("nan"),
            "model_beats_chamfer": False,
            "n_subjects": 0,
            "n_subject_pairs": 0,
            "n_mesh_pairs": 0,
            "n_observations": 0,
            "selected_topology_a": str(topology_a),
            "selected_topology_b": str(topology_b),
        }

    subject_ids_a = np.asarray(pair_subject_ids_a[selected_indices], dtype=object)
    subject_ids_b = np.asarray(pair_subject_ids_b[selected_indices], dtype=object)
    gt_mesh_values = np.asarray(pair_gt_mesh_values[selected_indices], dtype=np.float64)
    latent_mesh = np.asarray(latent_mesh_values[selected_indices], dtype=np.float64)
    chamfer_mesh = np.asarray(chamfer_mesh_values[selected_indices], dtype=np.float64)

    if mesh_pair_level:
        gt_values = gt_mesh_values
        latent_values = latent_mesh
        chamfer_values = chamfer_mesh
        subject_pair_count = int(len({_subject_pair_key(str(a), str(b)) for a, b in zip(subject_ids_a, subject_ids_b)}))
    else:
        subject_pair_keys, group_members = _group_subject_pair_members(subject_ids_a=subject_ids_a, subject_ids_b=subject_ids_b)
        gt_values = np.asarray(
            [gt_mesh_values[int(group_members[group_idx][0])] for group_idx in range(len(group_members))],
            dtype=np.float64,
        )
        latent_values = _aggregate_grouped_values(latent_mesh, group_members=group_members)
        chamfer_values = _aggregate_grouped_values(chamfer_mesh, group_members=group_members)
        subject_pair_count = int(len(subject_pair_keys))

    latent_spearman = float(base.spearman_corr(gt_values, latent_values))
    latent_pearson = float(base.pearson_corr(gt_values, latent_values))
    chamfer_spearman = float(base.spearman_corr(gt_values, chamfer_values))
    chamfer_pearson = float(base.pearson_corr(gt_values, chamfer_values))
    delta_spearman = float(latent_spearman - chamfer_spearman)
    delta_pearson = float(latent_pearson - chamfer_pearson)
    n_subjects = int(len({str(sid) for sid in subject_ids_a.tolist() + subject_ids_b.tolist()}))

    return {
        "topology_a": str(topology_a),
        "topology_b": str(topology_b),
        "ordered_pair_label": str(ordered_pair_label),
        "ordered_topology_pairs": bool(ordered),
        "mesh_pair_level": bool(mesh_pair_level),
        "latent_spearman": latent_spearman,
        "latent_pearson": latent_pearson,
        "chamfer_spearman": chamfer_spearman,
        "chamfer_pearson": chamfer_pearson,
        "delta_spearman": delta_spearman,
        "delta_pearson": delta_pearson,
        "model_beats_chamfer": bool(latent_spearman > chamfer_spearman),
        "n_subjects": n_subjects,
        "n_subject_pairs": int(subject_pair_count),
        "n_mesh_pairs": int(n_mesh_pairs),
        "n_observations": int(gt_values.size),
        "selected_topology_a": str(topology_a),
        "selected_topology_b": str(topology_b),
    }


def _write_pair_summary_csv(path: Path, row: dict) -> None:
    header = [
        "topology_a",
        "topology_b",
        "ordered_pair_label",
        "ordered_topology_pairs",
        "mesh_pair_level",
        "latent_spearman",
        "chamfer_spearman",
        "delta_spearman",
        "latent_pearson",
        "chamfer_pearson",
        "delta_pearson",
        "model_beats_chamfer",
        "n_subjects",
        "n_subject_pairs",
        "n_mesh_pairs",
        "n_observations",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerow(
            {
                "topology_a": row["topology_a"],
                "topology_b": row["topology_b"],
                "ordered_pair_label": row["ordered_pair_label"],
                "ordered_topology_pairs": int(bool(row["ordered_topology_pairs"])),
                "mesh_pair_level": int(bool(row["mesh_pair_level"])),
                "latent_spearman": _format_float(row["latent_spearman"]),
                "chamfer_spearman": _format_float(row["chamfer_spearman"]),
                "delta_spearman": _format_float(row["delta_spearman"]),
                "latent_pearson": _format_float(row["latent_pearson"]),
                "chamfer_pearson": _format_float(row["chamfer_pearson"]),
                "delta_pearson": _format_float(row["delta_pearson"]),
                "model_beats_chamfer": int(bool(row["model_beats_chamfer"])),
                "n_subjects": int(row["n_subjects"]),
                "n_subject_pairs": int(row["n_subject_pairs"]),
                "n_mesh_pairs": int(row["n_mesh_pairs"]),
                "n_observations": int(row["n_observations"]),
            }
        )


def _write_pair_summary_md(path: Path, row: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Topology Pair Breakdown\n\n")
        f.write(
            "| Topology A | Topology B | Pair Label | Ordered | Mesh-pair level | "
            "Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | "
            "Model > Chamfer | N subjects | N subject pairs | N mesh pairs | N observations |\n"
        )
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        f.write(
            "| "
            + " | ".join(
                [
                    str(row["topology_a"]),
                    str(row["topology_b"]),
                    str(row["ordered_pair_label"]),
                    "yes" if row["ordered_topology_pairs"] else "no",
                    "yes" if row["mesh_pair_level"] else "no",
                    _format_float(row["latent_spearman"]),
                    _format_float(row["chamfer_spearman"]),
                    _format_float(row["delta_spearman"]),
                    _format_float(row["latent_pearson"]),
                    _format_float(row["chamfer_pearson"]),
                    _format_float(row["delta_pearson"]),
                    "yes" if row["model_beats_chamfer"] else "no",
                    str(int(row["n_subjects"])),
                    str(int(row["n_subject_pairs"])),
                    str(int(row["n_mesh_pairs"])),
                    str(int(row["n_observations"])),
                ]
            )
            + " |\n"
        )


def _write_summary_csv(path: Path, rows: Sequence[dict]) -> None:
    header = [
        "topology_a",
        "topology_b",
        "ordered_pair_label",
        "ordered_topology_pairs",
        "mesh_pair_level",
        "latent_spearman",
        "chamfer_spearman",
        "delta_spearman",
        "latent_pearson",
        "chamfer_pearson",
        "delta_pearson",
        "model_beats_chamfer",
        "n_subjects",
        "n_subject_pairs",
        "n_mesh_pairs",
        "n_observations",
        "output_dir",
        "output_json",
        "output_csv",
        "output_md",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "topology_a": row["topology_a"],
                    "topology_b": row["topology_b"],
                    "ordered_pair_label": row["ordered_pair_label"],
                    "ordered_topology_pairs": int(bool(row["ordered_topology_pairs"])),
                    "mesh_pair_level": int(bool(row["mesh_pair_level"])),
                    "latent_spearman": _format_float(row["latent_spearman"]),
                    "chamfer_spearman": _format_float(row["chamfer_spearman"]),
                    "delta_spearman": _format_float(row["delta_spearman"]),
                    "latent_pearson": _format_float(row["latent_pearson"]),
                    "chamfer_pearson": _format_float(row["chamfer_pearson"]),
                    "delta_pearson": _format_float(row["delta_pearson"]),
                    "model_beats_chamfer": int(bool(row["model_beats_chamfer"])),
                    "n_subjects": int(row["n_subjects"]),
                    "n_subject_pairs": int(row["n_subject_pairs"]),
                    "n_mesh_pairs": int(row["n_mesh_pairs"]),
                    "n_observations": int(row["n_observations"]),
                    "output_dir": row.get("output_dir", ""),
                    "output_json": row.get("output_json", ""),
                    "output_csv": row.get("output_csv", ""),
                    "output_md": row.get("output_md", ""),
                }
            )


def main() -> None:
    cli_args = parse_args()

    run_dir, checkpoint_path = base._resolve_run_dir_and_checkpoint(
        cli_args.model_path,
        selector=str(cli_args.checkpoint_selector),
    )
    base_args = base.merge_run_args(checkpoint_path, explicit_config_json=cli_args.config_json)
    model_args = base._resolve_runtime_args(cli_args, base_args)

    base.seed_everything(int(model_args.seed))
    device = base._resolve_device(cli_args.device)

    dataset = base.GTReadyDataset(model_args.data_dir)
    gt_matrix, gt_name_to_idx = base.load_gt_distance_matrix(
        model_args.dist_npz,
        subject_re=base.SUBJECT_RE_ANY,
        dtype=base.np.float64,
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
    if sample_cache is not None:
        sample_cache, chamfer_cache_stats = base.maybe_cache_chamfer_vertices_on_device(
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

    sample_records = base.build_sample_eval_records(
        dataset=dataset,
        eval_plan=eval_plan,
        eval_subjects=target_subjects,
        sample_cache=sample_cache,
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

    model = base.build_model(args=model_args, device=device)
    checkpoint_bundle = base.load_checkpoint_bundle(checkpoint_path)
    model.load_state_dict(checkpoint_bundle["state_dict"], strict=True)
    model.eval()

    Z, vertex_sets = _collect_clean_embeddings_and_vertices(
        model=model,
        dataset=dataset,
        sample_records=pair_ctx.sample_records,
        sample_cache=sample_cache,
        device=device,
    )
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
    chamfer_mesh_values = base.compute_pairwise_chamfer_values(
        vertex_sets=vertex_sets,
        pair_i=pair_ctx.pair_i_cpu,
        pair_j=pair_ctx.pair_j_cpu,
        batch_pairs=int(cli_args.chamfer_batch_pairs),
        progress_desc="topology breakdown chamfer pairs",
        show_progress=False,
    )

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
    evaluation_level = "mesh_pair" if cli_args.mesh_pair_level else "subject_pair_mean"
    root_out_dir = (
        Path(cli_args.out_dir).expanduser().resolve()
        if cli_args.out_dir
        else _build_default_out_dir(
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            cli_args=cli_args,
            target_subjects=target_subjects,
            evaluation_level=evaluation_level,
        )
    )
    root_out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[dict] = []
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

        row = _summarize_topology_pair(
            topology_a=topology_a,
            topology_b=topology_b,
            ordered_pair_label=ordered_pair_label,
            ordered=bool(cli_args.ordered_topology_pairs),
            mesh_pair_indices=mesh_pair_indices,
            pair_topology_a=pair_topology_a,
            pair_topology_b=pair_topology_b,
            pair_subject_ids_a=pair_subject_ids_a,
            pair_subject_ids_b=pair_subject_ids_b,
            pair_gt_mesh_values=pair_gt_mesh_values,
            latent_mesh_values=latent_mesh_values,
            chamfer_mesh_values=chamfer_mesh_values,
            mesh_pair_level=bool(cli_args.mesh_pair_level),
        )

        if cli_args.write_per_pair_outputs:
            pair_dir = root_out_dir / base.slugify_token(ordered_pair_label)
            pair_dir.mkdir(parents=True, exist_ok=True)
            json_path = pair_dir / "ranking_summary.json"
            csv_path = pair_dir / "ranking_summary.csv"
            md_path = pair_dir / "ranking_summary.md"

            payload = {
                "checkpoint": str(checkpoint_path),
                "run_dir": str(run_dir),
                "checkpoint_selector": str(cli_args.checkpoint_selector),
                "device": str(device),
                "data_dir": str(model_args.data_dir),
                "dist_npz": str(model_args.dist_npz),
                "subject_split": str(cli_args.subject_split),
                "eval_fraction": float(cli_args.eval_fraction),
                "seed": int(model_args.seed),
                "max_subjects_requested": int(cli_args.max_subjects),
                "max_meshes_per_subject_eval": int(model_args.max_meshes_per_subject_eval),
                "pair_mode": str(cli_args.pair_mode),
                "topology_pair_mode": str(cli_args.topology_pair_mode),
                "ordered_topology_pairs": bool(cli_args.ordered_topology_pairs),
                "mesh_pair_level": bool(cli_args.mesh_pair_level),
                "selected_subjects": list(target_subjects),
                "eval_plan_summary": base.summarize_eval_plan(eval_plan),
                "sample_eval_summary": base.summarize_sample_records(sample_records),
                "pair_context": {
                    "n_subjects": int(pair_ctx.n_subjects),
                    "n_samples": int(pair_ctx.n_samples),
                    "n_pairs": int(pair_ctx.pair_count),
                    "n_mesh_pairs": int(pair_ctx.mesh_pair_count),
                    "n_subject_pairs": int(pair_ctx.subject_pair_count),
                    "n_topology_labels": int(pair_ctx.n_topology_labels),
                },
                "chamfer_cache_verts": dict(chamfer_cache_stats),
                "topology_pair": {
                    "topology_a": str(topology_a),
                    "topology_b": str(topology_b),
                    "ordered_pair_label": str(ordered_pair_label),
                },
                "row": dict(row),
            }
            _write_json(json_path, payload)
            _write_pair_summary_csv(csv_path, row=row)
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

        summary_rows.append(row)

    summary_csv_path = root_out_dir / cli_args.summary_filename
    _write_summary_csv(summary_csv_path, rows=summary_rows)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Root output dir: {root_out_dir}")
    print(f"Selected subjects: {len(target_subjects)}")
    print(f"Samples kept: {pair_ctx.n_samples}")
    print(f"Mesh-pairs kept by pair_mode: {pair_ctx.mesh_pair_count}")
    print(f"Aggregated summary CSV: {summary_csv_path}")
    for row in summary_rows:
        print(
            f"[{row['ordered_pair_label']}] "
            f"latent_sp={_format_float(row['latent_spearman']) or 'nan'} "
            f"chamfer_sp={_format_float(row['chamfer_spearman']) or 'nan'} "
            f"n_mesh_pairs={int(row['n_mesh_pairs'])}"
        )


if __name__ == "__main__":
    main()
