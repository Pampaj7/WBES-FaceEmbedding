from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
import torch
from tqdm import tqdm

from .data_utils import (
    GTReadyDataset,
    build_eval_plan,
    build_sample_eval_records,
    maybe_cache_chamfer_vertices_on_device,
    preload_eval_samples,
    rebuild_subject_split,
    summarize_sample_records,
)
from .eval_utils import (
    ALLOWED_AGGREGATION_LEVELS,
    ALLOWED_METRICS,
    ALLOWED_PAIR_MODES,
    build_pair_eval_context,
    build_perturbation_reference,
    build_sigma_grid,
    evaluate_robustness_grid_tqdm,
    parse_ratio_thresholds,
    summarize_eval_plan,
    summarize_pack,
    threshold_label,
    write_grid_csv,
    write_summary_md,
)
from .model_helpers import build_model
from .noise import PerturbationParams, parse_noise_modes
from .paths import DEFAULT_DATA_DIR, DEFAULT_DIST_NPZ, ensure_autoencoder_dir_on_syspath


ensure_autoencoder_dir_on_syspath()

from intrinsic_utils import SUBJECT_RE_ANY, build_subject_map, load_gt_distance_matrix, seed_everything, slugify_token  # noqa: E402


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    noise_modes: List[str]
    eval_mode: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate robustness breakdown of an existing checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    p.add_argument("--config_json", type=str, default="", help="Optional explicit config.json path")
    p.add_argument("--out_dir", type=str, default="", help="Optional output directory")
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--data_dir", type=str, default="", help="Override dataset path")
    p.add_argument("--dist_npz", type=str, default="", help="Override GT distance matrix path")
    p.add_argument("--metric", type=str, default="latent", choices=ALLOWED_METRICS)
    p.add_argument("--pair_mode", type=str, default="all", choices=ALLOWED_PAIR_MODES)
    p.add_argument(
        "--aggregation_level",
        type=str,
        default="mesh_pair",
        choices=ALLOWED_AGGREGATION_LEVELS,
        help=(
            "mesh_pair: each valid mesh-mesh pair is one observation; "
            "subject_pair_mean/subject_pair_median: aggregate valid mesh-mesh distances per subject pair"
        ),
    )

    p.add_argument("--subject_split", type=str, default="eval", choices=("eval", "train", "all"))
    p.add_argument("--eval_fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=-1, help="Negative => use checkpoint/config seed")
    p.add_argument("--max_subjects", type=int, default=0, help="0 = all overlapping subjects")
    p.add_argument(
        "--max_meshes_per_subject_eval",
        "--eval_meshes_per_subject",
        dest="max_meshes_per_subject_eval",
        type=int,
        default=-1,
        help="Max eval meshes per subject. Negative => use checkpoint/config",
    )
    p.add_argument(
        "--preload_eval_samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preload selected eval meshes once and reuse them across all sigma/mode evaluations",
    )
    p.add_argument(
        "--preload_workers",
        type=int,
        default=4,
        help="Workers for sample preloading; 0 = auto, 1 = sequential",
    )

    p.add_argument(
        "--eval_noise_modes",
        type=str,
        default="jitter,rigid,outliers",
        help="Comma-separated perturbation modes used only at evaluation",
    )
    p.add_argument(
        "--aggregate_eval_mode",
        type=str,
        default="average",
        choices=("fixed", "random", "average"),
        help="Aggregate behavior when multiple eval modes are provided",
    )
    p.add_argument("--report_per_mode", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--sigma_min_eval", type=float, default=1e-4)
    p.add_argument("--sigma_max_eval", type=float, default=2e-1)
    p.add_argument("--n_sigma_eval", type=int, default=16)

    p.add_argument("--outlier_frac", type=float, default=-1.0, help="Negative => use checkpoint/config")
    p.add_argument("--outlier_scale", type=float, default=-1.0, help="Negative => use checkpoint/config")
    p.add_argument("--rigid_rot_deg", type=float, default=-1.0, help="Negative => use checkpoint/config")
    p.add_argument("--rigid_trans_scale", type=float, default=-1.0, help="Negative => use checkpoint/config")
    p.add_argument("--rigid_rot_deg_min", type=float, default=-1.0, help="Negative => use checkpoint/config")
    p.add_argument("--rigid_trans_scale_min", type=float, default=-1.0, help="Negative => use checkpoint/config")
    p.add_argument("--chamfer_batch_pairs", type=int, default=64, help="Pair batch size for Chamfer evaluation")
    p.add_argument(
        "--chamfer_pair_progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show inner pair-batch progress bars during Chamfer evaluation",
    )
    p.add_argument(
        "--chamfer_cache_verts",
        type=str,
        default="auto",
        choices=("off", "auto", "force"),
        help="Cache preloaded Chamfer vertex tensors on the eval device once to reduce repeated CPU->device copies",
    )
    p.add_argument(
        "--chamfer_cache_verts_max_mb",
        type=float,
        default=256.0,
        help="In auto mode, cache Chamfer verts on device only if their total size is at most this many MB",
    )

    p.add_argument(
        "--ratio_thresholds",
        type=str,
        default="0.9,0.8,0.5",
        help="Comma-separated thresholds for sigma_xx metrics",
    )
    return p.parse_args()


def infer_run_dir(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint_bundle(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def list_epoch_checkpoints(checkpoint_dir: Path) -> List[tuple[int, Path]]:
    out: List[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("epoch*.pth"):
        match = re.fullmatch(r"epoch(\d+)\.pth", path.name)
        if match is None:
            continue
        out.append((int(match.group(1)), path.resolve()))
    out.sort(key=lambda item: item[0])
    return out


def choose_default_checkpoint(run_dir: Path) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    best_auc = ckpt_dir / "best_by_auc.pth"
    if best_auc.exists():
        return best_auc.resolve()

    epochs = list_epoch_checkpoints(ckpt_dir)
    if epochs:
        return epochs[-1][1]
    raise FileNotFoundError(f"No checkpoints found under: {ckpt_dir}")


def resolve_checkpoint_path(raw_path: str) -> tuple[Path, str]:
    requested = Path(raw_path).expanduser().resolve()
    if requested.exists():
        if requested.is_dir():
            resolved = choose_default_checkpoint(requested)
            return resolved, f"Input is a run directory; using {resolved.name}"
        return requested, ""

    if requested.suffix == "" and requested.parent.exists():
        ckpt_dir = requested / "checkpoints"
        if ckpt_dir.exists():
            resolved = choose_default_checkpoint(requested)
            return resolved, f"Input directory has no direct file; using {resolved.name}"

    match = re.fullmatch(r"epoch(\d+)\.pth", requested.name)
    if match is not None and requested.parent.exists():
        target_epoch = int(match.group(1))
        candidates = list_epoch_checkpoints(requested.parent)
        if candidates:
            nearest_epoch, nearest_path = min(candidates, key=lambda item: (abs(item[0] - target_epoch), item[0]))
            return nearest_path, f"Requested {requested.name} not found; using nearest saved checkpoint {nearest_path.name}"

    raise FileNotFoundError(f"Checkpoint not found: {requested}")


def merge_run_args(checkpoint_path: Path, explicit_config_json: str) -> Dict[str, object]:
    pack = load_checkpoint_bundle(checkpoint_path)
    if not isinstance(pack, dict) or "state_dict" not in pack:
        raise RuntimeError(f"{checkpoint_path} is not a valid checkpoint bundle with state_dict")

    run_dir = infer_run_dir(checkpoint_path)
    config_path = Path(explicit_config_json) if explicit_config_json else (run_dir / "config.json")

    merged: Dict[str, object] = {}
    if config_path.exists():
        cfg = load_json(config_path)
        if isinstance(cfg, dict) and "args" in cfg and isinstance(cfg["args"], dict):
            merged.update(cfg["args"])
        elif isinstance(cfg, dict):
            merged.update(cfg)

    ckpt_args = pack.get("args", {})
    if isinstance(ckpt_args, dict):
        merged.update(ckpt_args)
    return merged


def resolve_runtime_args(cli_args: argparse.Namespace, base_args: Dict[str, object]) -> SimpleNamespace:
    args = dict(base_args)

    if cli_args.data_dir:
        args["data_dir"] = cli_args.data_dir
    if cli_args.dist_npz:
        args["dist_npz"] = cli_args.dist_npz
    if cli_args.seed >= 0:
        args["seed"] = int(cli_args.seed)
    if cli_args.max_meshes_per_subject_eval >= 0:
        args["max_meshes_per_subject_eval"] = int(cli_args.max_meshes_per_subject_eval)
    if cli_args.outlier_frac >= 0.0:
        args["outlier_frac"] = float(cli_args.outlier_frac)
    if cli_args.outlier_scale >= 0.0:
        args["outlier_scale"] = float(cli_args.outlier_scale)
    if cli_args.rigid_rot_deg >= 0.0:
        args["rigid_rot_deg"] = float(cli_args.rigid_rot_deg)
    if cli_args.rigid_trans_scale >= 0.0:
        args["rigid_trans_scale"] = float(cli_args.rigid_trans_scale)
    if cli_args.rigid_rot_deg_min >= 0.0:
        args["rigid_rot_deg_min"] = float(cli_args.rigid_rot_deg_min)
    if cli_args.rigid_trans_scale_min >= 0.0:
        args["rigid_trans_scale_min"] = float(cli_args.rigid_trans_scale_min)

    args.setdefault("seed", 1234)
    args.setdefault("data_dir", str(DEFAULT_DATA_DIR))
    args.setdefault("dist_npz", str(DEFAULT_DIST_NPZ))
    args.setdefault("max_meshes_per_subject_eval", 0)
    args.setdefault("outlier_frac", 0.02)
    args.setdefault("outlier_scale", 6.0)
    args.setdefault("rigid_rot_deg", 5.0)
    args.setdefault("rigid_trans_scale", 0.02)
    args.setdefault("rigid_rot_deg_min", 0.0)
    args.setdefault("rigid_trans_scale_min", 0.0)
    return SimpleNamespace(**args)


def _resolve_device(device_name: str) -> torch.device:
    return torch.device(device_name if (device_name == "cuda" and torch.cuda.is_available()) else "cpu")


def _select_target_subjects(
    dataset: GTReadyDataset,
    gt_name_to_idx: Dict[str, int],
    cli_args: argparse.Namespace,
    model_args: SimpleNamespace,
) -> tuple[Dict[str, List[int]], list[str], list[str], list[str]]:
    subject_map = build_subject_map(dataset.files, subject_re=SUBJECT_RE_ANY)
    subjects = sorted([sid for sid in subject_map.keys() if sid in gt_name_to_idx])
    train_subjects, eval_subjects = rebuild_subject_split(
        subjects=subjects,
        eval_fraction=cli_args.eval_fraction,
        seed=int(model_args.seed),
        max_subjects=int(cli_args.max_subjects),
    )
    if cli_args.subject_split == "eval":
        target_subjects = eval_subjects
    elif cli_args.subject_split == "train":
        target_subjects = train_subjects
    else:
        target_subjects = sorted(train_subjects + eval_subjects)
    return subject_map, train_subjects, eval_subjects, target_subjects


def _load_model_for_metric(cli_args: argparse.Namespace, model_args: SimpleNamespace, checkpoint_path: Path, device: torch.device) -> torch.nn.Module | None:
    if cli_args.metric != "latent":
        return None
    model = build_model(args=model_args, device=device)
    pack = load_checkpoint_bundle(checkpoint_path)
    model.load_state_dict(pack["state_dict"], strict=True)
    model.eval()
    return model


def _build_scenario_specs(eval_noise_modes: Sequence[str], report_per_mode: bool, aggregate_eval_mode: str) -> List[ScenarioSpec]:
    scenario_specs: List[ScenarioSpec] = []
    if report_per_mode:
        for mode in eval_noise_modes:
            scenario_specs.append(ScenarioSpec(name=mode, noise_modes=[mode], eval_mode="fixed"))

    if len(eval_noise_modes) == 1:
        scenario_specs.append(
            ScenarioSpec(
                name=f"aggregate_{eval_noise_modes[0]}",
                noise_modes=list(eval_noise_modes),
                eval_mode="fixed",
            )
        )
    else:
        scenario_specs.append(
            ScenarioSpec(
                name=f"aggregate_{aggregate_eval_mode}",
                noise_modes=list(eval_noise_modes),
                eval_mode=aggregate_eval_mode,
            )
        )
    return scenario_specs


def _evaluate_scenarios(
    scenario_specs: Sequence[ScenarioSpec],
    model: torch.nn.Module | None,
    dataset: GTReadyDataset,
    pair_ctx,
    sample_cache,
    device: torch.device,
    cli_args: argparse.Namespace,
    perturbation: PerturbationParams,
    sigma_grid: Sequence[float],
    seed: int,
) -> Dict[str, Dict[str, object]]:
    scenario_packs: Dict[str, Dict[str, object]] = {}
    shared_clean: Dict[str, float] | None = None

    for spec in tqdm(scenario_specs, desc="Scenarios", dynamic_ncols=True):
        if len(spec.noise_modes) == 1 and spec.name.startswith("aggregate_") and spec.noise_modes[0] in scenario_packs:
            scenario_packs[spec.name] = scenario_packs[spec.noise_modes[0]]
            continue

        scenario_packs[spec.name] = evaluate_robustness_grid_tqdm(
            model=model,
            dataset=dataset,
            pair_ctx=pair_ctx,
            sample_cache=sample_cache,
            device=device,
            metric=cli_args.metric,
            chamfer_batch_pairs=int(cli_args.chamfer_batch_pairs),
            sigma_grid=sigma_grid,
            noise_modes=spec.noise_modes,
            params=perturbation,
            seed=seed,
            eval_mode=spec.eval_mode,
            progress_desc=f"{spec.name} sigma",
            show_chamfer_pair_progress=bool(cli_args.chamfer_pair_progress and cli_args.metric == "chamfer"),
            precomputed_clean=shared_clean,
        )
        if shared_clean is None:
            shared_clean = dict(scenario_packs[spec.name]["clean"])
        scenario_packs[spec.name]["metric"] = str(cli_args.metric)
        scenario_packs[spec.name]["pair_mode"] = str(cli_args.pair_mode)
        scenario_packs[spec.name]["aggregation_level"] = str(cli_args.aggregation_level)
        scenario_packs[spec.name]["noise_modes"] = list(spec.noise_modes)
        scenario_packs[spec.name]["eval_mode"] = str(spec.eval_mode)
        scenario_packs[spec.name]["outlier_frac"] = float(perturbation.outlier_frac)
        scenario_packs[spec.name]["outlier_scale"] = float(perturbation.outlier_scale)
        scenario_packs[spec.name]["rigid_rot_deg"] = float(perturbation.rigid_rot_deg)
        scenario_packs[spec.name]["rigid_trans_scale"] = float(perturbation.rigid_trans_scale)
        scenario_packs[spec.name]["rigid_rot_deg_min"] = float(perturbation.rigid_rot_deg_min)
        scenario_packs[spec.name]["rigid_trans_scale_min"] = float(perturbation.rigid_trans_scale_min)
    return scenario_packs


def _build_output_dir(
    cli_args: argparse.Namespace,
    checkpoint_path: Path,
    run_dir: Path,
    model_args: SimpleNamespace,
    eval_noise_modes: Sequence[str],
) -> Path:
    out_dir = (
        Path(cli_args.out_dir).resolve()
        if cli_args.out_dir
        else run_dir
        / "posthoc_breakdown"
        / slugify_token(
            f"{checkpoint_path.stem}_metric-{cli_args.metric}_pairs-{cli_args.pair_mode}_"
            f"agglvl-{cli_args.aggregation_level}_"
            f"split-{cli_args.subject_split}_modes-{'-'.join(eval_noise_modes)}_"
            f"meval-{int(model_args.max_meshes_per_subject_eval)}_"
            f"agg-{cli_args.aggregate_eval_mode}_s{cli_args.sigma_min_eval:.1e}-{cli_args.sigma_max_eval:.1e}_"
            f"n{cli_args.n_sigma_eval}"
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _print_summary(
    checkpoint_path: Path,
    checkpoint_note: str,
    out_dir: Path,
    cli_args: argparse.Namespace,
    pair_ctx,
    sample_cache,
    chamfer_cache_stats: Dict[str, object],
    eval_plan_summary: Dict[str, float],
    target_subjects: Sequence[str],
    summaries: Sequence[Dict[str, object]],
) -> None:
    print(f"Checkpoint: {checkpoint_path}")
    if checkpoint_note:
        print(f"Note: {checkpoint_note}")
    print(f"Output dir: {out_dir}")
    print(f"Metric: {cli_args.metric}")
    print(f"Pair mode: {cli_args.pair_mode}")
    print(f"Aggregation level: {cli_args.aggregation_level}")
    print(f"Subjects evaluated: {len(target_subjects)}")
    print(f"Subjects kept: {pair_ctx.n_subjects}")
    print(f"Samples kept: {pair_ctx.n_samples}")
    print(f"Observations kept: {pair_ctx.pair_count}")
    print(f"Mesh pairs kept: {pair_ctx.mesh_pair_count}")
    print(f"Subject pairs kept: {pair_ctx.subject_pair_count}")
    print(f"Topology labels: {pair_ctx.n_topology_labels}")
    print(f"Cached eval meshes: {len(sample_cache) if sample_cache is not None else 0}")
    if cli_args.metric == "chamfer":
        print("Chamfer vertices: full mesh vertices (no subsampling)")
        print(
            "Chamfer vertex cache: "
            f"mode={chamfer_cache_stats['mode']} "
            f"enabled={chamfer_cache_stats['enabled']} "
            f"device={chamfer_cache_stats['device'] or 'cpu'} "
            f"verts_mb={float(chamfer_cache_stats['vertex_mb']):.1f} "
            f"reason={chamfer_cache_stats['reason']}"
        )
    print(
        "Eval plan: "
        f"subjects_with_meshes={eval_plan_summary['n_subjects_with_meshes']} "
        f"total_meshes={eval_plan_summary['total_meshes']} "
        f"min_per_subject={eval_plan_summary['min_meshes_per_subject']} "
        f"max_per_subject={eval_plan_summary['max_meshes_per_subject']} "
        f"mean_per_subject={eval_plan_summary['mean_meshes_per_subject']:.2f}"
    )
    for summary in summaries:
        thr_map = summary["thresholds"]
        thr_txt = " ".join(
            f"{key}={'nan' if not np.isfinite(float(val)) else f'{float(val):.2e}'}"
            for key, val in thr_map.items()
        )
        print(
            f"[{summary['scenario']}] sp_clean={float(summary['spearman_clean']):.4f} "
            f"aucR={float(summary['auc_r']):.4f} obs={int(summary['n_pairs'])} "
            f"mesh_pairs={int(summary.get('n_mesh_pairs', 0))} "
            f"subject_pairs={int(summary.get('n_subject_pairs', 0))} "
            f"worst_sp={float(summary['worst_spearman']):.4f} "
            f"worst_ratio={float(summary['worst_ratio']):.4f} {thr_txt}"
        )


def run_posthoc_evaluation(cli_args: argparse.Namespace) -> None:
    checkpoint_path, checkpoint_note = resolve_checkpoint_path(cli_args.checkpoint)
    base_args = merge_run_args(checkpoint_path, explicit_config_json=cli_args.config_json)
    model_args = resolve_runtime_args(cli_args, base_args)
    perturbation = PerturbationParams.from_namespace(model_args)
    seed_everything(int(model_args.seed))

    device = _resolve_device(cli_args.device)
    dataset = GTReadyDataset(model_args.data_dir)
    gt_matrix, gt_name_to_idx = load_gt_distance_matrix(model_args.dist_npz, subject_re=SUBJECT_RE_ANY, dtype=np.float64)
    subject_map, _, _, target_subjects = _select_target_subjects(
        dataset=dataset,
        gt_name_to_idx=gt_name_to_idx,
        cli_args=cli_args,
        model_args=model_args,
    )

    eval_noise_modes = parse_noise_modes(cli_args.eval_noise_modes)
    sigma_grid = build_sigma_grid(cli_args.sigma_min_eval, cli_args.sigma_max_eval, cli_args.n_sigma_eval)
    thresholds = parse_ratio_thresholds(cli_args.ratio_thresholds)

    eval_plan = build_eval_plan(
        subj_map=subject_map,
        eval_subjects=target_subjects,
        max_meshes_per_subject_eval=int(model_args.max_meshes_per_subject_eval),
        seed=int(model_args.seed),
    )
    sample_cache = (
        preload_eval_samples(dataset=dataset, eval_plan=eval_plan, workers=int(cli_args.preload_workers))
        if cli_args.preload_eval_samples
        else None
    )
    sample_records = build_sample_eval_records(
        dataset=dataset,
        eval_plan=eval_plan,
        eval_subjects=target_subjects,
        sample_cache=sample_cache,
    )
    sample_eval_summary = summarize_sample_records(sample_records)
    pair_ctx = build_pair_eval_context(
        sample_records=sample_records,
        name_to_idx=gt_name_to_idx,
        gt_matrix=gt_matrix,
        device=device,
        pair_mode=cli_args.pair_mode,
        aggregation_level=cli_args.aggregation_level,
    )

    chamfer_cache_stats: Dict[str, object] = {
        "mode": str(cli_args.chamfer_cache_verts),
        "enabled": False,
        "device": "",
        "vertex_bytes": 0,
        "vertex_mb": 0.0,
        "reason": "not_requested",
    }
    if cli_args.metric == "chamfer":
        sample_cache, chamfer_cache_stats = maybe_cache_chamfer_vertices_on_device(
            sample_cache=sample_cache,
            device=device,
            cache_mode=str(cli_args.chamfer_cache_verts),
            max_mb=float(cli_args.chamfer_cache_verts_max_mb),
        )

    model = _load_model_for_metric(cli_args=cli_args, model_args=model_args, checkpoint_path=checkpoint_path, device=device)
    run_dir = infer_run_dir(checkpoint_path)
    out_dir = _build_output_dir(
        cli_args=cli_args,
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
        model_args=model_args,
        eval_noise_modes=eval_noise_modes,
    )

    scenario_specs = _build_scenario_specs(
        eval_noise_modes=eval_noise_modes,
        report_per_mode=bool(cli_args.report_per_mode),
        aggregate_eval_mode=cli_args.aggregate_eval_mode,
    )
    scenario_packs = _evaluate_scenarios(
        scenario_specs=scenario_specs,
        model=model,
        dataset=dataset,
        pair_ctx=pair_ctx,
        sample_cache=sample_cache,
        device=device,
        cli_args=cli_args,
        perturbation=perturbation,
        sigma_grid=sigma_grid,
        seed=int(model_args.seed),
    )

    summaries = [summarize_pack(name=name, eval_pack=pack, thresholds=thresholds) for name, pack in scenario_packs.items()]
    summaries.sort(key=lambda item: float(item["spearman_clean"]), reverse=True)
    eval_plan_summary = summarize_eval_plan(eval_plan)
    chamfer_vertex_mode = "full_vertices" if cli_args.metric == "chamfer" else ""

    summary_json = {
        "checkpoint": str(checkpoint_path),
        "run_dir": str(run_dir),
        "device": str(device),
        "data_dir": str(model_args.data_dir),
        "dist_npz": str(model_args.dist_npz),
        "metric": str(cli_args.metric),
        "pair_mode": str(cli_args.pair_mode),
        "aggregation_level": str(cli_args.aggregation_level),
        "subject_split": cli_args.subject_split,
        "seed": int(model_args.seed),
        "max_meshes_per_subject_eval": int(model_args.max_meshes_per_subject_eval),
        "n_subjects_target": int(len(target_subjects)),
        "n_subjects_kept": int(pair_ctx.n_subjects),
        "n_eval_samples": int(pair_ctx.n_samples),
        "n_pairs": int(pair_ctx.pair_count),
        "n_mesh_pairs": int(pair_ctx.mesh_pair_count),
        "n_subject_pairs": int(pair_ctx.subject_pair_count),
        "n_cached_samples": int(len(sample_cache) if sample_cache is not None else 0),
        "chamfer_vertex_mode": chamfer_vertex_mode,
        "chamfer_pair_progress": bool(cli_args.chamfer_pair_progress),
        "chamfer_cache_verts": dict(chamfer_cache_stats),
        "ratio_definition": "spearman / spearman_clean",
        "target_subjects_preview": list(target_subjects[:10]),
        "kept_subjects_preview": list(pair_ctx.kept_subjects[:10]),
        "pair_counts_by_mode": dict(pair_ctx.pair_counts_by_mode),
        "subject_pair_counts_by_mode": dict(pair_ctx.subject_pair_counts_by_mode),
        "eval_plan_summary": eval_plan_summary,
        "sample_eval_summary": sample_eval_summary,
        "pair_context_summary": {
            "pair_mode": str(pair_ctx.pair_mode),
            "aggregation_level": str(pair_ctx.aggregation_level),
            "n_pairs": int(pair_ctx.pair_count),
            "n_mesh_pairs": int(pair_ctx.mesh_pair_count),
            "n_subject_pairs": int(pair_ctx.subject_pair_count),
            "n_eval_samples": int(pair_ctx.n_samples),
            "n_subjects": int(pair_ctx.n_subjects),
            "n_topology_labels": int(pair_ctx.n_topology_labels),
        },
        "preload_eval_samples": bool(cli_args.preload_eval_samples),
        "eval_noise_modes": eval_noise_modes,
        "aggregate_eval_mode": cli_args.aggregate_eval_mode,
        "sigma_grid": sigma_grid,
        "ratio_thresholds": list(thresholds),
        "topology_label_resolution": {
            "priority": "filename-derived label after removing subject id and generic GTready tokens",
            "fallback": "faces topology signature",
        },
        "perturbation_reference": build_perturbation_reference(perturbation),
        "scenarios": {name: pack for name, pack in scenario_packs.items()},
        "summaries": summaries,
    }

    with open(out_dir / "breakdown_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    write_grid_csv(
        out_dir / "breakdown_grid.csv",
        scenario_packs,
        max_meshes_per_subject_eval=int(model_args.max_meshes_per_subject_eval),
        pair_ctx=pair_ctx,
        eval_plan_summary=eval_plan_summary,
    )
    write_summary_md(out_dir / "breakdown_summary.md", summaries, thresholds)

    _print_summary(
        checkpoint_path=checkpoint_path,
        checkpoint_note=checkpoint_note,
        out_dir=out_dir,
        cli_args=cli_args,
        pair_ctx=pair_ctx,
        sample_cache=sample_cache,
        chamfer_cache_stats=chamfer_cache_stats,
        eval_plan_summary=eval_plan_summary,
        target_subjects=target_subjects,
        summaries=summaries,
    )


def main() -> None:
    run_posthoc_evaluation(parse_args())


if __name__ == "__main__":
    main()
