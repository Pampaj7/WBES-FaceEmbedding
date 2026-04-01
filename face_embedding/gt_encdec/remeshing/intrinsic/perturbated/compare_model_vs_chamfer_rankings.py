from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
import torch


THIS_FILE = Path(__file__).resolve()
INTRINSIC_DIR = THIS_FILE.parent.parent
if str(INTRINSIC_DIR) not in sys.path:
    sys.path.append(str(INTRINSIC_DIR))

from robustness.data_utils import (  # noqa: E402
    GTReadyDataset,
    build_eval_plan,
    build_sample_eval_records,
    maybe_cache_chamfer_vertices_on_device,
    preload_eval_samples,
    rebuild_subject_split,
    sample_to_device,
    summarize_sample_records,
)
from robustness.eval_utils import (  # noqa: E402
    ALLOWED_AGGREGATION_LEVELS,
    ALLOWED_PAIR_MODES,
    aggregate_pair_observations,
    build_pair_eval_context,
    compute_pairwise_chamfer_values,
    summarize_eval_plan,
)
from robustness.model_helpers import build_model, forward_model  # noqa: E402
from robustness.noise import (  # noqa: E402
    PerturbationParams,
    apply_xyz_perturbation_with_params,
    rigid_angle_max_deg_from_sigma,
    rigid_trans_axis_std_from_sigma,
)
from robustness.paths import DEFAULT_DATA_DIR, DEFAULT_DIST_NPZ, ensure_autoencoder_dir_on_syspath  # noqa: E402
from robustness.posthoc_runner import infer_run_dir, load_checkpoint_bundle, merge_run_args  # noqa: E402


ensure_autoencoder_dir_on_syspath()

from intrinsic_utils import (  # noqa: E402
    SUBJECT_RE_ANY,
    build_subject_map,
    load_gt_distance_matrix,
    pearson_corr,
    seed_everything,
    slugify_token,
    spearman_corr,
)


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    jitter_sigma: float
    rotation_sigma: float
    translation_sigma: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare a model ranking against Chamfer ranking on the same subset under "
            "clean/jitter/translation/rotation/mixed perturbations."
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
        "--topology_labels",
        type=str,
        default="",
        help="Optional comma-separated topology labels to keep, e.g. crop,noisy,original,remesh",
    )
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

    p.add_argument("--pair_mode", type=str, default="cross_topology", choices=ALLOWED_PAIR_MODES)
    p.add_argument(
        "--aggregation_level",
        type=str,
        default="subject_pair_mean",
        choices=ALLOWED_AGGREGATION_LEVELS,
    )
    p.add_argument(
        "--scenarios",
        type=str,
        default="clean,jitter,translation,rotation,mixed",
        help="Comma-separated scenarios from: clean,jitter,translation,rotation,mixed",
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
    return p.parse_args()


def _resolve_device(device_name: str) -> torch.device:
    return torch.device(device_name if (device_name == "cuda" and torch.cuda.is_available()) else "cpu")


def _resolve_run_dir_and_checkpoint(model_path: str, selector: str) -> tuple[Path, Path]:
    requested = Path(model_path).expanduser().resolve()
    if requested.is_file():
        return infer_run_dir(requested), requested

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


def _resolve_runtime_args(cli_args: argparse.Namespace, base_args: Dict[str, object]) -> SimpleNamespace:
    args = dict(base_args)

    if cli_args.data_dir:
        args["data_dir"] = cli_args.data_dir
    if cli_args.dist_npz:
        args["dist_npz"] = cli_args.dist_npz
    if cli_args.seed >= 0:
        args["seed"] = int(cli_args.seed)
    if cli_args.max_meshes_per_subject_eval > 0:
        args["max_meshes_per_subject_eval"] = int(cli_args.max_meshes_per_subject_eval)
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
    args.setdefault("max_meshes_per_subject_eval", 2)
    args.setdefault("rigid_rot_deg", 5.0)
    args.setdefault("rigid_trans_scale", 0.02)
    args.setdefault("rigid_rot_deg_min", 0.0)
    args.setdefault("rigid_trans_scale_min", 0.0)
    args.setdefault("sigma_max", 0.05)
    return SimpleNamespace(**args)


def _parse_topology_labels(text: str) -> List[str]:
    labels: List[str] = []
    seen = set()
    for raw in str(text).split(","):
        label = raw.strip().lower()
        if not label:
            continue
        if label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def _filter_sample_records_by_topology_labels(
    sample_records: Sequence[object],
    topology_labels: Sequence[str],
) -> List[object]:
    allowed = {str(label).strip().lower() for label in topology_labels if str(label).strip()}
    if not allowed:
        return list(sample_records)

    present = sorted({str(rec.topology_label).lower() for rec in sample_records})
    missing = [label for label in sorted(allowed) if label not in present]
    if missing:
        raise ValueError(
            f"Requested topology labels not found in selected sample set: {missing}. "
            f"Available={present}"
        )

    filtered = [rec for rec in sample_records if str(rec.topology_label).lower() in allowed]
    if not filtered:
        raise RuntimeError(f"No samples left after topology filter {sorted(allowed)}")
    return filtered


def _eval_plan_from_sample_records(sample_records: Sequence[object]) -> Dict[str, List[int]]:
    plan: Dict[str, List[int]] = {}
    for rec in sample_records:
        sid = str(rec.subject_id)
        plan.setdefault(sid, []).append(int(rec.dataset_idx))
    return {sid: sorted(set(indices)) for sid, indices in sorted(plan.items())}


def _select_subject_subset(
    subjects: Sequence[str],
    subject_split: str,
    eval_fraction: float,
    seed: int,
    max_subjects: int,
) -> tuple[List[str], List[str], List[str]]:
    arr = np.asarray(sorted(subjects), dtype=object)
    rng = np.random.default_rng(seed)

    if max_subjects > 0 and len(arr) > max_subjects:
        pick = rng.choice(len(arr), size=max_subjects, replace=False)
        arr = arr[np.sort(pick)]

    selected_subjects = sorted(arr.tolist())
    if subject_split == "all":
        return [], [], selected_subjects

    train_subjects, eval_subjects = rebuild_subject_split(
        subjects=selected_subjects,
        eval_fraction=float(eval_fraction),
        seed=int(seed),
        max_subjects=0,
    )
    if subject_split == "eval":
        target_subjects = eval_subjects
    else:
        target_subjects = train_subjects
    return train_subjects, eval_subjects, target_subjects


def _parse_scenarios(text: str, base_sigma: float, cli_args: argparse.Namespace) -> List[ScenarioSpec]:
    allowed = {"clean", "jitter", "translation", "rotation", "mixed"}
    names = [tok.strip().lower() for tok in str(text).split(",") if tok.strip()]
    if not names:
        raise ValueError("scenarios must contain at least one scenario")
    bad = [name for name in names if name not in allowed]
    if bad:
        raise ValueError(f"Unsupported scenarios: {bad}. Allowed={sorted(allowed)}")

    def choose(override: float) -> float:
        return float(override if override >= 0.0 else base_sigma)

    specs: List[ScenarioSpec] = []
    for name in names:
        if name == "clean":
            specs.append(ScenarioSpec(name=name, jitter_sigma=0.0, rotation_sigma=0.0, translation_sigma=0.0))
        elif name == "jitter":
            specs.append(ScenarioSpec(name=name, jitter_sigma=choose(cli_args.jitter_sigma), rotation_sigma=0.0, translation_sigma=0.0))
        elif name == "translation":
            specs.append(
                ScenarioSpec(name=name, jitter_sigma=0.0, rotation_sigma=0.0, translation_sigma=choose(cli_args.translation_sigma))
            )
        elif name == "rotation":
            specs.append(
                ScenarioSpec(name=name, jitter_sigma=0.0, rotation_sigma=choose(cli_args.rotation_sigma), translation_sigma=0.0)
            )
        elif name == "mixed":
            specs.append(
                ScenarioSpec(
                    name=name,
                    jitter_sigma=choose(cli_args.mixed_jitter_sigma),
                    rotation_sigma=choose(cli_args.mixed_rotation_sigma),
                    translation_sigma=choose(cli_args.mixed_translation_sigma),
                )
            )
    return specs


def _scenario_seed(base_seed: int, scenario_index: int, dataset_idx: int) -> int:
    return int((int(base_seed) * 1_000_003 + int(scenario_index) * 10_007 + int(dataset_idx) * 97) % (2**31 - 1))


def _rotation_only_params(params: PerturbationParams) -> PerturbationParams:
    return PerturbationParams(
        outlier_frac=float(params.outlier_frac),
        outlier_scale=float(params.outlier_scale),
        rigid_rot_deg=float(params.rigid_rot_deg),
        rigid_trans_scale=0.0,
        rigid_rot_deg_min=float(params.rigid_rot_deg_min),
        rigid_trans_scale_min=0.0,
    )


def _translation_only_params(params: PerturbationParams) -> PerturbationParams:
    return PerturbationParams(
        outlier_frac=float(params.outlier_frac),
        outlier_scale=float(params.outlier_scale),
        rigid_rot_deg=0.0,
        rigid_trans_scale=float(params.rigid_trans_scale),
        rigid_rot_deg_min=0.0,
        rigid_trans_scale_min=float(params.rigid_trans_scale_min),
    )


def _apply_scenario(
    V: torch.Tensor,
    params: PerturbationParams,
    scenario: ScenarioSpec,
    seed: int,
) -> torch.Tensor:
    if scenario.name == "clean":
        return V

    devices = [V.device] if V.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if V.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))

        out = V
        if scenario.jitter_sigma > 0.0:
            out = apply_xyz_perturbation_with_params(out, mode="jitter", sigma=float(scenario.jitter_sigma), params=params)
        if scenario.rotation_sigma > 0.0:
            out = apply_xyz_perturbation_with_params(
                out,
                mode="rigid",
                sigma=float(scenario.rotation_sigma),
                params=_rotation_only_params(params),
            )
        if scenario.translation_sigma > 0.0:
            out = apply_xyz_perturbation_with_params(
                out,
                mode="rigid",
                sigma=float(scenario.translation_sigma),
                params=_translation_only_params(params),
            )
        return out


def _describe_scenario(scenario: ScenarioSpec, params: PerturbationParams) -> Dict[str, float]:
    rot_sigma = float(scenario.rotation_sigma)
    trans_sigma = float(scenario.translation_sigma)
    return {
        "jitter_sigma": float(scenario.jitter_sigma),
        "rotation_sigma": rot_sigma,
        "translation_sigma": trans_sigma,
        "rotation_angle_max_deg": rigid_angle_max_deg_from_sigma(
            sigma=rot_sigma,
            rigid_rot_deg=params.rigid_rot_deg,
            rigid_rot_deg_min=params.rigid_rot_deg_min,
        )
        if rot_sigma > 0.0
        else 0.0,
        "translation_axis_std": rigid_trans_axis_std_from_sigma(
            sigma=trans_sigma,
            rigid_trans_scale=params.rigid_trans_scale,
            rigid_trans_scale_min=params.rigid_trans_scale_min,
        )
        if trans_sigma > 0.0
        else 0.0,
    }


def _progress_log_step(total: int, target_updates: int = 10) -> int:
    total_i = max(1, int(total))
    target_i = max(1, int(target_updates))
    return max(1, math.ceil(total_i / target_i))


@torch.no_grad()
def _evaluate_scenario(
    model: torch.nn.Module,
    dataset: GTReadyDataset,
    pair_ctx,
    sample_cache: Dict[int, Dict[str, torch.Tensor]] | None,
    device: torch.device,
    params: PerturbationParams,
    scenario: ScenarioSpec,
    scenario_index: int,
    base_seed: int,
    chamfer_batch_pairs: int,
) -> Dict[str, object]:
    latent_vectors: List[torch.Tensor] = []
    full_vertex_sets: List[torch.Tensor] = []
    sample_total = len(pair_ctx.sample_records)
    sample_log_step = _progress_log_step(sample_total)

    for sample_index, record in enumerate(pair_ctx.sample_records, start=1):
        sample = sample_cache[int(record.dataset_idx)] if sample_cache is not None else dataset[int(record.dataset_idx)]
        sample_d = sample_to_device(sample, device=device)
        pert_seed = _scenario_seed(base_seed=base_seed, scenario_index=scenario_index, dataset_idx=int(record.dataset_idx))
        V_in = _apply_scenario(V=sample_d["verts"], params=params, scenario=scenario, seed=pert_seed)
        z, _ = forward_model(
            model=model,
            sample_dict=sample_d,
            V_in=V_in,
            return_gate_info=False,
            add_noise=False,
        )
        latent_vectors.append(z.squeeze(0))
        full_vertex_sets.append(V_in.contiguous())
        if sample_index == 1 or sample_index == sample_total or sample_index % sample_log_step == 0:
            print(
                f"Scenario {scenario.name}: encoded {sample_index}/{sample_total} samples",
                flush=True,
            )

    Z = torch.stack(latent_vectors, dim=0)
    latent_values = torch.linalg.vector_norm(
        Z.index_select(0, pair_ctx.pair_i) - Z.index_select(0, pair_ctx.pair_j),
        dim=1,
    )
    latent_values_np = aggregate_pair_observations(
        latent_values.detach().cpu().numpy().astype(np.float64, copy=False),
        pair_ctx=pair_ctx,
    )

    print(
        f"Scenario {scenario.name}: computing Chamfer over {pair_ctx.mesh_pair_count} mesh pairs "
        f"(batch_pairs={int(chamfer_batch_pairs)})",
        flush=True,
    )
    chamfer_values_np = aggregate_pair_observations(
        compute_pairwise_chamfer_values(
            vertex_sets=full_vertex_sets,
            pair_i=pair_ctx.pair_i_cpu,
            pair_j=pair_ctx.pair_j_cpu,
            batch_pairs=int(chamfer_batch_pairs),
            progress_desc=f"{scenario.name} chamfer pairs",
            show_progress=True,
        ),
        pair_ctx=pair_ctx,
    )
    print(
        f"Scenario {scenario.name}: finished Chamfer over {pair_ctx.mesh_pair_count} mesh pairs",
        flush=True,
    )

    gt_vals = np.asarray(pair_ctx.gt_vals, dtype=np.float64)
    return {
        "scenario": scenario.name,
        "latent_spearman": float(spearman_corr(gt_vals, latent_values_np)),
        "latent_pearson": float(pearson_corr(gt_vals, latent_values_np)),
        "chamfer_spearman": float(spearman_corr(gt_vals, chamfer_values_np)),
        "chamfer_pearson": float(pearson_corr(gt_vals, chamfer_values_np)),
        "n_pairs": int(pair_ctx.pair_count),
        "n_mesh_pairs": int(pair_ctx.mesh_pair_count),
        "n_subject_pairs": int(pair_ctx.subject_pair_count),
        "n_subjects": int(pair_ctx.n_subjects),
        "n_samples": int(pair_ctx.n_samples),
    }


def _write_summary_csv(path: Path, rows: Sequence[dict], params: PerturbationParams, scenarios: Sequence[ScenarioSpec]) -> None:
    scenario_lookup = {spec.name: spec for spec in scenarios}
    header = [
        "scenario",
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
        "n_subjects",
        "n_samples",
        "n_pairs",
        "n_mesh_pairs",
        "n_subject_pairs",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            desc = _describe_scenario(scenario_lookup[row["scenario"]], params=params)
            writer.writerow(
                {
                    "scenario": row["scenario"],
                    "jitter_sigma": f"{desc['jitter_sigma']:.6e}",
                    "rotation_sigma": f"{desc['rotation_sigma']:.6e}",
                    "translation_sigma": f"{desc['translation_sigma']:.6e}",
                    "rotation_angle_max_deg": f"{desc['rotation_angle_max_deg']:.6f}",
                    "translation_axis_std": f"{desc['translation_axis_std']:.6f}",
                    "latent_spearman": f"{row['latent_spearman']:.6f}",
                    "chamfer_spearman": f"{row['chamfer_spearman']:.6f}",
                    "delta_spearman": f"{row['delta_spearman']:.6f}",
                    "latent_pearson": f"{row['latent_pearson']:.6f}",
                    "chamfer_pearson": f"{row['chamfer_pearson']:.6f}",
                    "delta_pearson": f"{row['delta_pearson']:.6f}",
                    "model_beats_chamfer": int(bool(row["model_beats_chamfer"])),
                    "n_subjects": int(row["n_subjects"]),
                    "n_samples": int(row["n_samples"]),
                    "n_pairs": int(row["n_pairs"]),
                    "n_mesh_pairs": int(row["n_mesh_pairs"]),
                    "n_subject_pairs": int(row["n_subject_pairs"]),
                }
            )


def _write_summary_md(path: Path, rows: Sequence[dict], params: PerturbationParams, scenarios: Sequence[ScenarioSpec]) -> None:
    scenario_lookup = {spec.name: spec for spec in scenarios}
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Model vs Chamfer Ranking Summary\n\n")
        f.write(
            "| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | "
            "Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |\n"
        )
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            desc = _describe_scenario(scenario_lookup[row["scenario"]], params=params)
            f.write(
                "| "
                + " | ".join(
                    [
                        str(row["scenario"]),
                        f"{desc['jitter_sigma']:.6e}",
                        f"{desc['rotation_sigma']:.6e}",
                        f"{desc['translation_sigma']:.6e}",
                        f"{desc['rotation_angle_max_deg']:.6f}",
                        f"{desc['translation_axis_std']:.6f}",
                        f"{row['latent_spearman']:.6f}",
                        f"{row['chamfer_spearman']:.6f}",
                        f"{row['delta_spearman']:.6f}",
                        f"{row['latent_pearson']:.6f}",
                        f"{row['chamfer_pearson']:.6f}",
                        f"{row['delta_pearson']:.6f}",
                        "yes" if row["model_beats_chamfer"] else "no",
                    ]
                )
                + " |\n"
            )


def main() -> None:
    cli_args = parse_args()

    run_dir, checkpoint_path = _resolve_run_dir_and_checkpoint(cli_args.model_path, selector=str(cli_args.checkpoint_selector))
    base_args = merge_run_args(checkpoint_path, explicit_config_json=cli_args.config_json)
    model_args = _resolve_runtime_args(cli_args, base_args)
    params = PerturbationParams.from_namespace(model_args)

    base_sigma = float(cli_args.base_sigma)
    if base_sigma < 0.0:
        base_sigma = float(getattr(model_args, "sigma_max", 0.05))
    scenarios = _parse_scenarios(cli_args.scenarios, base_sigma=base_sigma, cli_args=cli_args)

    seed_everything(int(model_args.seed))
    device = _resolve_device(cli_args.device)

    dataset = GTReadyDataset(model_args.data_dir)
    gt_matrix, gt_name_to_idx = load_gt_distance_matrix(model_args.dist_npz, subject_re=SUBJECT_RE_ANY, dtype=np.float64)
    subject_map = build_subject_map(dataset.files, subject_re=SUBJECT_RE_ANY)
    subjects = sorted([sid for sid in subject_map.keys() if sid in gt_name_to_idx])
    train_subjects, eval_subjects, target_subjects = _select_subject_subset(
        subjects=subjects,
        subject_split=str(cli_args.subject_split),
        eval_fraction=float(cli_args.eval_fraction),
        seed=int(model_args.seed),
        max_subjects=int(cli_args.max_subjects),
    )

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
    if sample_cache is not None:
        sample_cache, chamfer_cache_stats = maybe_cache_chamfer_vertices_on_device(
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

    sample_records = build_sample_eval_records(
        dataset=dataset,
        eval_plan=eval_plan,
        eval_subjects=target_subjects,
        sample_cache=sample_cache,
    )
    requested_topology_labels = _parse_topology_labels(cli_args.topology_labels)
    if requested_topology_labels:
        sample_records = _filter_sample_records_by_topology_labels(
            sample_records=sample_records,
            topology_labels=requested_topology_labels,
        )
    active_eval_plan = _eval_plan_from_sample_records(sample_records)
    active_subjects = sorted(active_eval_plan.keys())
    print(
        f"Prepared sample subset: subjects={len(active_subjects)} "
        f"samples={len(sample_records)} "
        f"topologies={sorted({rec.topology_label for rec in sample_records})}",
        flush=True,
    )
    pair_ctx = build_pair_eval_context(
        sample_records=sample_records,
        name_to_idx=gt_name_to_idx,
        gt_matrix=gt_matrix,
        device=device,
        pair_mode=str(cli_args.pair_mode),
        aggregation_level=str(cli_args.aggregation_level),
    )
    if pair_ctx.pair_count <= 0:
        raise RuntimeError("No valid evaluation pairs found for the selected subset and pair mode")
    print(
        f"Prepared pair context: subject_pairs={pair_ctx.subject_pair_count} "
        f"mesh_pairs={pair_ctx.mesh_pair_count} "
        f"aggregation={cli_args.aggregation_level}",
        flush=True,
    )

    model = build_model(args=model_args, device=device)
    checkpoint_bundle = load_checkpoint_bundle(checkpoint_path)
    model.load_state_dict(checkpoint_bundle["state_dict"], strict=True)
    model.eval()

    rows: List[dict] = []
    for scenario_index, scenario in enumerate(scenarios):
        print(
            f"Starting scenario {scenario_index + 1}/{len(scenarios)}: {scenario.name}",
            flush=True,
        )
        result = _evaluate_scenario(
            model=model,
            dataset=dataset,
            pair_ctx=pair_ctx,
            sample_cache=sample_cache,
            device=device,
            params=params,
            scenario=scenario,
            scenario_index=scenario_index,
            base_seed=int(model_args.seed),
            chamfer_batch_pairs=int(cli_args.chamfer_batch_pairs),
        )
        result["delta_spearman"] = float(result["latent_spearman"] - result["chamfer_spearman"])
        result["delta_pearson"] = float(result["latent_pearson"] - result["chamfer_pearson"])
        result["model_beats_chamfer"] = bool(result["latent_spearman"] > result["chamfer_spearman"])
        rows.append(result)
        print(
            f"Finished scenario {scenario.name}: "
            f"latent_sp={result['latent_spearman']:.4f} "
            f"chamfer_sp={result['chamfer_spearman']:.4f} "
            f"delta={result['delta_spearman']:.4f}",
            flush=True,
        )

    out_dir = (
        Path(cli_args.out_dir).expanduser().resolve()
        if cli_args.out_dir
        else run_dir
        / "perturbation_ranking_vs_chamfer"
        / slugify_token(
            f"{checkpoint_path.stem}_split-{cli_args.subject_split}_pairs-{cli_args.pair_mode}_"
            f"agglvl-{cli_args.aggregation_level}_subjects-{len(target_subjects)}_"
            f"meshes-{int(model_args.max_meshes_per_subject_eval)}_scenarios-{'-'.join(spec.name for spec in scenarios)}"
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

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
        "aggregation_level": str(cli_args.aggregation_level),
        "selected_subjects": list(active_subjects),
        "eval_plan_summary": summarize_eval_plan(active_eval_plan),
        "sample_eval_summary": summarize_sample_records(sample_records),
        "pair_context": {
            "n_subjects": int(pair_ctx.n_subjects),
            "n_samples": int(pair_ctx.n_samples),
            "n_pairs": int(pair_ctx.pair_count),
            "n_mesh_pairs": int(pair_ctx.mesh_pair_count),
            "n_subject_pairs": int(pair_ctx.subject_pair_count),
            "n_topology_labels": int(pair_ctx.n_topology_labels),
        },
        "chamfer_cache_verts": dict(chamfer_cache_stats),
        "perturbation_params": {
            "rigid_rot_deg": float(params.rigid_rot_deg),
            "rigid_trans_scale": float(params.rigid_trans_scale),
            "rigid_rot_deg_min": float(params.rigid_rot_deg_min),
            "rigid_trans_scale_min": float(params.rigid_trans_scale_min),
        },
        "scenarios": [spec.__dict__ for spec in scenarios],
        "rows": rows,
    }

    with open(out_dir / "ranking_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=True)
        f.write("\n")
    _write_summary_csv(out_dir / "ranking_summary.csv", rows=rows, params=params, scenarios=scenarios)
    _write_summary_md(out_dir / "ranking_summary.md", rows=rows, params=params, scenarios=scenarios)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {out_dir}")
    print(f"Selected subjects: {len(target_subjects)}")
    print(f"Samples kept: {pair_ctx.n_samples}")
    print(f"Pairs kept: {pair_ctx.pair_count}")
    for row in rows:
        print(
            f"[{row['scenario']}] latent_sp={row['latent_spearman']:.4f} "
            f"chamfer_sp={row['chamfer_spearman']:.4f} delta={row['delta_spearman']:.4f}"
        )


if __name__ == "__main__":
    main()
