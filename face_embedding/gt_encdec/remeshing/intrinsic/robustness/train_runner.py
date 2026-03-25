from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .data_utils import GTReadyDataset, build_eval_plan, preload_eval_samples, rebuild_subject_split, sample_to_device
from .eval_utils import SubjectEvalContext, build_sigma_grid, evaluate_subject_robustness_grid, summarize_eval_plan
from .model_helpers import build_model, forward_model, smooth_term_from_model
from .noise import (
    PerturbationParams,
    apply_xyz_perturbation_with_params,
    parse_noise_mode_weights,
    parse_noise_modes,
    sample_log_uniform_sigma,
)
from .paths import DEFAULT_DATA_DIR, DEFAULT_DIST_NPZ, TRAIN_RUNS_ROOT, ensure_autoencoder_dir_on_syspath


ensure_autoencoder_dir_on_syspath()

from intrinsic_utils import (  # noqa: E402
    SUBJECT_RE_ANY,
    build_subject_map,
    load_gt_distance_matrix,
    sample_mesh_indices,
    seed_everything,
    slugify_token,
)
from latent_loss import stress_loss  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train robustness-aware intrinsic encoders.")

    p.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    p.add_argument("--dist_npz", type=str, default=str(DEFAULT_DIST_NPZ))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--runs_root",
        type=str,
        default="",
        help="Optional output root directory. Default: intrinsic/perturbated",
    )

    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_subjects", type=int, default=4)
    p.add_argument("--init_checkpoint", type=str, default="", help="Optional checkpoint or run dir used to initialize model weights")
    p.add_argument("--teacher_checkpoint", type=str, default="", help="Optional frozen teacher checkpoint or run dir for teacher-student distillation")
    p.add_argument("--lambda_teacher", type=float, default=0.0, help="Weight for teacher-student embedding distillation loss")

    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--n_blocks", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument(
        "--model",
        type=str,
        default="gated_twotower",
        choices=["xyz_dn", "intrinsic_dn", "xyz_spec_dn", "gated_twotower", "concat_twotower", "spec_mlp"],
        help=(
            "Encoder model: "
            "xyz_dn=DiffusionEncoderOnly, "
            "intrinsic_dn=DiffusionEncoderOnlyIntrinsec, "
            "xyz_spec_dn=DiffusionEncoderXYZSpectrum, "
            "gated_twotower=TwoTowerGatedRobust, "
            "concat_twotower=TwoTowerConcat, "
            "spec_mlp=Spectrum-only baseline MLP"
        ),
    )

    p.add_argument("--k_spec", type=int, default=100)
    p.add_argument("--log_spec", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--eps", type=float, default=1e-8)

    p.add_argument("--use_xyz", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_spectrum", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--n_hks", type=int, default=0)
    p.add_argument("--n_wks", type=int, default=0)
    p.add_argument("--eig_k", type=int, default=300)
    p.add_argument("--pool_mode", type=str, default="meanmax", choices=["mean", "meanmax"])

    p.add_argument("--xyz_feature_dropout", type=float, default=0.0)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument(
        "--max_subjects_eval_train",
        type=int,
        default=24,
        help="Cap subjects used by the online training-time eval only. 0 = all eval subjects.",
    )
    p.add_argument(
        "--preload_eval_samples_train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preload the fixed online-eval meshes once at startup to avoid repeated dataset I/O during training eval.",
    )
    p.add_argument(
        "--preload_eval_workers_train",
        type=int,
        default=2,
        help="Workers used to preload online-eval samples once at startup. 0 = auto, 1 = sequential.",
    )

    p.add_argument("--max_meshes_per_subject_train", type=int, default=0, help="0 = all")
    p.add_argument("--max_meshes_per_subject_eval", type=int, default=0, help="0 = all")

    p.add_argument("--use_id_loss", action="store_true")
    p.add_argument("--lambda_id", type=float, default=0.1)
    p.add_argument("--use_smooth_loss", action="store_true")
    p.add_argument("--lambda_smooth", type=float, default=0.01)
    p.add_argument(
        "--train_latent_noise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the encoder's internal latent Gaussian noise during training forwards",
    )

    p.add_argument("--p_noise", type=float, default=0.8)
    p.add_argument("--sigma_min", type=float, default=1e-4)
    p.add_argument("--sigma_max", type=float, default=3e-2)
    p.add_argument(
        "--noise_modes",
        type=str,
        default="jitter,rigid,outliers",
        help="Comma-separated perturbation modes from: jitter,rigid,rotation,translation,outliers",
    )
    p.add_argument(
        "--noise_mode_weights",
        type=str,
        default="",
        help=(
            "Optional comma-separated mode weights for noisy training batches, "
            "e.g. 'translation=6,rotation=2,jitter=1,outliers=1'. Unspecified modes default to 1."
        ),
    )
    p.add_argument("--outlier_frac", type=float, default=0.02)
    p.add_argument("--outlier_scale", type=float, default=6.0)
    p.add_argument("--rigid_rot_deg", type=float, default=5.0)
    p.add_argument("--rigid_trans_scale", type=float, default=0.02)
    p.add_argument("--rigid_rot_deg_min", type=float, default=0.0)
    p.add_argument("--rigid_trans_scale_min", type=float, default=0.0)

    p.add_argument("--sigma_min_eval", type=float, default=1e-4)
    p.add_argument("--sigma_max_eval", type=float, default=3e-2)
    p.add_argument("--n_sigma_eval", type=int, default=8)
    p.add_argument(
        "--eval_mode",
        type=str,
        default="fixed",
        choices=["fixed", "random", "average"],
        help="fixed: deterministic first mode, random: random mode per sample, average: average over all modes",
    )
    return p.parse_args()


def make_run_dir(args: argparse.Namespace, noise_modes: Sequence[str], noise_mode_probs: Sequence[float]) -> Path:
    fp = {
        "model": str(args.model),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_subjects": int(args.batch_subjects),
        "init_checkpoint": str(Path(args.init_checkpoint).expanduser().resolve()) if str(args.init_checkpoint).strip() else "",
        "teacher_checkpoint": str(Path(args.teacher_checkpoint).expanduser().resolve()) if str(args.teacher_checkpoint).strip() else "",
        "lambda_teacher": float(args.lambda_teacher),
        "latent_dim": int(args.latent_dim),
        "width": int(args.width),
        "n_blocks": int(args.n_blocks),
        "dropout": float(args.dropout),
        "use_xyz": bool(args.use_xyz),
        "use_spectrum": bool(args.use_spectrum),
        "n_hks": int(args.n_hks),
        "n_wks": int(args.n_wks),
        "eig_k": int(args.eig_k),
        "k_spec": int(args.k_spec),
        "log_spec": bool(args.log_spec),
        "pool_mode": str(args.pool_mode),
        "xyz_feature_dropout": float(args.xyz_feature_dropout),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "max_meshes_per_subject_train": int(args.max_meshes_per_subject_train),
        "max_meshes_per_subject_eval": int(args.max_meshes_per_subject_eval),
        "max_subjects_eval_train": int(args.max_subjects_eval_train),
        "preload_eval_samples_train": bool(args.preload_eval_samples_train),
        "preload_eval_workers_train": int(args.preload_eval_workers_train),
        "p_noise": float(args.p_noise),
        "sigma_min": float(args.sigma_min),
        "sigma_max": float(args.sigma_max),
        "noise_modes": list(noise_modes),
        "noise_mode_probs": {str(mode): float(prob) for mode, prob in zip(noise_modes, noise_mode_probs)},
        "outlier_frac": float(args.outlier_frac),
        "outlier_scale": float(args.outlier_scale),
        "rigid_rot_deg": float(args.rigid_rot_deg),
        "rigid_trans_scale": float(args.rigid_trans_scale),
        "rigid_rot_deg_min": float(args.rigid_rot_deg_min),
        "rigid_trans_scale_min": float(args.rigid_trans_scale_min),
        "sigma_min_eval": float(args.sigma_min_eval),
        "sigma_max_eval": float(args.sigma_max_eval),
        "n_sigma_eval": int(args.n_sigma_eval),
        "eval_mode": str(args.eval_mode),
        "use_id_loss": bool(args.use_id_loss),
        "lambda_id": float(args.lambda_id),
        "use_smooth_loss": bool(args.use_smooth_loss),
        "lambda_smooth": float(args.lambda_smooth),
        "train_latent_noise": bool(args.train_latent_noise),
    }

    fp_json = json.dumps(fp, sort_keys=True, separators=(",", ":"))
    short_hash = hashlib.sha1(fp_json.encode("utf-8")).hexdigest()[:8]

    teacher_tag = f"_ts{fp['lambda_teacher']:.2f}" if fp["lambda_teacher"] > 0.0 else ""
    cfg = (
        f"m{fp['model']}_z{fp['latent_dim']}_w{fp['width']}_b{fp['n_blocks']}_ks{fp['k_spec']}"
        f"_pool{fp['pool_mode']}_bs{fp['batch_subjects']}"
        f"_tn{int(fp['train_latent_noise'])}"
        f"_pn{fp['p_noise']:.2f}_s{fp['sigma_min']:.1e}-{fp['sigma_max']:.1e}"
        f"{teacher_tag}"
        f"_seed{fp['seed']}"
    )
    runs_root = _resolve_runs_root(args)
    run_dir = runs_root / f"{slugify_token(cfg)}__{short_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "noise_modes": list(noise_modes), "fingerprint": fp}, f, indent=2, sort_keys=True)

    (run_dir / "run_started.txt").write_text(datetime.now().isoformat(), encoding="utf-8")
    return run_dir


def _format_launch_command() -> str:
    argv0 = str(Path(sys.argv[0]).resolve()) if sys.argv else ""
    parts = [sys.executable]
    if argv0:
        parts.append(argv0)
    parts.extend(str(arg) for arg in sys.argv[1:])
    return shlex.join(parts)


def _write_launch_summary(run_dir: Path, args: argparse.Namespace) -> None:
    now = datetime.now().isoformat()
    env_name = os.environ.get("CONDA_DEFAULT_ENV", "")
    lines = [
        f"started_at: {now}",
        f"cwd: {Path.cwd()}",
        f"python: {sys.executable}",
    ]
    if env_name:
        lines.append(f"conda_env: {env_name}")
    lines.extend(
        [
            "",
            "command:",
            _format_launch_command(),
            "",
            "parsed_args:",
            json.dumps(vars(args), indent=2, sort_keys=True),
            "",
        ]
    )
    (run_dir / "launch_command.txt").write_text("\n".join(lines), encoding="utf-8")


def _select_online_eval_subjects(
    eval_subjects: Sequence[str],
    max_subjects_eval_train: int,
    seed: int,
) -> List[str]:
    subjects = sorted(str(sid) for sid in eval_subjects)
    cap = int(max_subjects_eval_train)
    if cap <= 0 or len(subjects) <= cap:
        return subjects

    rng = np.random.default_rng(int(seed) + 31_337)
    picked = rng.choice(len(subjects), size=cap, replace=False)
    selected = [subjects[int(i)] for i in np.sort(picked)]
    return sorted(selected)


def _configure_device(device_name: str) -> torch.device:
    device = torch.device(device_name if (device_name == "cuda" and torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            torch.backends.cudnn.conv.fp32_precision = "tf32"
        except Exception:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    return device


def _resolve_runs_root(args: argparse.Namespace) -> Path:
    if str(args.runs_root).strip():
        return Path(args.runs_root).expanduser().resolve()
    if str(getattr(args, "teacher_checkpoint", "")).strip() and float(getattr(args, "lambda_teacher", 0.0)) > 0.0:
        return (TRAIN_RUNS_ROOT / "teacher_student").resolve()
    return TRAIN_RUNS_ROOT.resolve()


def _load_checkpoint_bundle(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _list_epoch_checkpoints(checkpoint_dir: Path) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    for path in checkpoint_dir.glob("epoch*.pth"):
        stem = path.stem
        if not stem.startswith("epoch"):
            continue
        suffix = stem[len("epoch") :]
        if not suffix.isdigit():
            continue
        out.append((int(suffix), path.resolve()))
    out.sort(key=lambda item: item[0])
    return out


def _resolve_init_checkpoint(raw_path: str) -> Path:
    requested = Path(raw_path).expanduser().resolve()
    if requested.is_file():
        return requested
    if requested.is_dir():
        ckpt_dir = requested / "checkpoints"
        if ckpt_dir.exists():
            best_auc = ckpt_dir / "best_by_auc.pth"
            if best_auc.exists():
                return best_auc.resolve()
            epochs = _list_epoch_checkpoints(ckpt_dir)
            if epochs:
                return epochs[-1][1]
        best_auc = requested / "best_by_auc.pth"
        if best_auc.exists():
            return best_auc.resolve()
        epochs = _list_epoch_checkpoints(requested)
        if epochs:
            return epochs[-1][1]
    raise FileNotFoundError(f"Init checkpoint not found: {requested}")


def _infer_run_dir_from_checkpoint(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def _merge_checkpoint_args(checkpoint_path: Path, pack: dict) -> Dict[str, object]:
    merged: Dict[str, object] = {}
    run_dir = _infer_run_dir_from_checkpoint(checkpoint_path)
    config_path = run_dir / "config.json"
    if config_path.exists():
        cfg = _load_json(config_path)
        if isinstance(cfg, dict) and "args" in cfg and isinstance(cfg["args"], dict):
            merged.update(cfg["args"])
        elif isinstance(cfg, dict):
            merged.update(cfg)

    ckpt_args = pack.get("args", {})
    if isinstance(ckpt_args, dict):
        merged.update(ckpt_args)
    return merged


def _load_frozen_teacher(
    raw_path: str,
    device: torch.device,
) -> tuple[nn.Module, Path, SimpleNamespace, int]:
    teacher_checkpoint_path = _resolve_init_checkpoint(raw_path)
    teacher_pack = _load_checkpoint_bundle(teacher_checkpoint_path)
    state_dict = teacher_pack["state_dict"] if isinstance(teacher_pack, dict) and "state_dict" in teacher_pack else teacher_pack
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"{teacher_checkpoint_path} does not contain a valid teacher state_dict")

    teacher_args = SimpleNamespace(**_merge_checkpoint_args(teacher_checkpoint_path, teacher_pack))
    teacher_model = build_model(args=teacher_args, device=device)
    teacher_model.load_state_dict(state_dict, strict=True)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad_(False)

    teacher_epoch = int(teacher_pack.get("epoch", -1)) if isinstance(teacher_pack, dict) else -1
    return teacher_model, teacher_checkpoint_path, teacher_args, teacher_epoch


def _teacher_distillation_loss(student_z: torch.Tensor, teacher_z: torch.Tensor) -> torch.Tensor:
    if student_z.shape != teacher_z.shape:
        raise RuntimeError(f"Teacher/student latent shape mismatch: student={tuple(student_z.shape)} teacher={tuple(teacher_z.shape)}")
    return ((student_z - teacher_z) ** 2).mean()


def _write_log_headers(log_csv: Path, robust_csv: Path) -> None:
    with open(log_csv, "w", encoding="utf-8") as f:
        f.write(
            "epoch,loss,stress,id,smooth,teacher,lr,"
            "spearman_clean,pearson_clean,auc_r,"
            "gate_mean_clean,gate_mean_noisy_max,spearman_noisy_max,ratio_noisy_max,"
            "intra_clean,n_eval\n"
        )
    with open(robust_csv, "w", encoding="utf-8") as f:
        f.write("epoch,sigma,is_clean,spearman,pearson,ratio,gate_mean,intra_mean,n_eval\n")


def _train_epoch(
    model: nn.Module,
    dataset: GTReadyDataset,
    subj_map: Dict[str, List[int]],
    train_subjects: Sequence[str],
    name_to_idx: Dict[str, int],
    gt_matrix: np.ndarray,
    device: torch.device,
    optimizer: optim.Optimizer,
    teacher_model: nn.Module | None,
    epoch: int,
    args: argparse.Namespace,
    noise_modes: Sequence[str],
    noise_mode_probs: Sequence[float],
    perturbation: PerturbationParams,
) -> Dict[str, float]:
    model.train()
    rng = np.random.default_rng(args.seed + 777 + epoch)
    train_perm = rng.permutation(np.array(train_subjects, dtype=object))

    epoch_loss = 0.0
    epoch_stress = 0.0
    epoch_id = 0.0
    epoch_smooth = 0.0
    epoch_teacher = 0.0
    n_steps = 0
    noise_mode_probs_np = np.asarray(noise_mode_probs, dtype=np.float64)

    pbar = tqdm(
        range(0, len(train_perm), args.batch_subjects),
        desc=f"Epoch {epoch}/{args.epochs}",
        dynamic_ncols=True,
    )

    for start in pbar:
        batch_subjects = train_perm[start : start + args.batch_subjects].tolist()
        if len(batch_subjects) < 2:
            continue

        do_noise = bool(rng.uniform() < float(args.p_noise))
        sigma_batch = sample_log_uniform_sigma(args.sigma_min, args.sigma_max, rng) if do_noise else 0.0

        optimizer.zero_grad(set_to_none=True)

        subj_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        smooth_terms: List[torch.Tensor] = []
        teacher_terms: List[torch.Tensor] = []

        for sid in batch_subjects:
            idxs = sample_mesh_indices(
                subj_map[sid],
                max_meshes=args.max_meshes_per_subject_train,
                seed=int(rng.integers(0, 2_000_000_000)),
            )
            if not idxs:
                continue

            z_list: List[torch.Tensor] = []
            for idx in idxs:
                sample = dataset[int(idx)]
                sample_d = sample_to_device(sample, device=device)
                V_in = sample_d["verts"]
                if sigma_batch > 0.0:
                    if len(noise_modes) == 1:
                        mode = noise_modes[0]
                    else:
                        mode = noise_modes[int(rng.choice(len(noise_modes), p=noise_mode_probs_np))]
                    V_in = apply_xyz_perturbation_with_params(
                        V=V_in,
                        mode=mode,
                        sigma=sigma_batch,
                        params=perturbation,
                    )

                z, _ = forward_model(
                    model=model,
                    sample_dict=sample_d,
                    V_in=V_in,
                    return_gate_info=False,
                    add_noise=bool(args.train_latent_noise),
                )
                z_student = z.squeeze(0)
                z_list.append(z_student)

                if teacher_model is not None:
                    with torch.no_grad():
                        z_teacher, _ = forward_model(
                            model=teacher_model,
                            sample_dict=sample_d,
                            V_in=sample_d["verts"],
                            return_gate_info=False,
                            add_noise=False,
                        )
                    teacher_terms.append(_teacher_distillation_loss(z_student, z_teacher.squeeze(0)))

                if args.use_smooth_loss:
                    smooth_term = smooth_term_from_model(
                        model=model,
                        sample_dict=sample_d,
                        V_in=V_in,
                    )
                    if smooth_term is not None:
                        smooth_terms.append(smooth_term)

            if not z_list:
                continue

            Zs = torch.stack(z_list, dim=0)
            subj_stats[sid] = (Zs, Zs.mean(dim=0))

        if len(subj_stats) < 2:
            continue

        subj_means: List[torch.Tensor] = []
        gt_idx: List[int] = []
        for sid in batch_subjects:
            packed = subj_stats.get(sid)
            gt_i = name_to_idx.get(sid)
            if packed is None or gt_i is None:
                continue
            _, zm = packed
            subj_means.append(zm)
            gt_idx.append(gt_i)

        if len(subj_means) < 2:
            continue

        Z_batch = torch.stack(subj_means, dim=0)
        idx_np = np.asarray(gt_idx, dtype=int)
        D_batch = torch.tensor(gt_matrix[np.ix_(idx_np, idx_np)], device=device, dtype=Z_batch.dtype)
        loss_stress = stress_loss(Z_batch, D_batch)

        if args.use_id_loss:
            id_terms: List[torch.Tensor] = []
            for Zs, Zm in subj_stats.values():
                if Zs.shape[0] < 2:
                    continue
                id_terms.append(((Zs - Zm.unsqueeze(0)) ** 2).mean())
            loss_id = torch.stack(id_terms).mean() if id_terms else torch.tensor(0.0, device=device)
        else:
            loss_id = torch.tensor(0.0, device=device)

        if args.use_smooth_loss:
            loss_smooth = torch.stack(smooth_terms).mean() if smooth_terms else torch.tensor(0.0, device=device)
        else:
            loss_smooth = torch.tensor(0.0, device=device)

        if teacher_model is not None:
            loss_teacher = torch.stack(teacher_terms).mean() if teacher_terms else torch.tensor(0.0, device=device)
        else:
            loss_teacher = torch.tensor(0.0, device=device)

        loss = loss_stress + args.lambda_id * loss_id + args.lambda_smooth * loss_smooth + args.lambda_teacher * loss_teacher
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        li = float(loss.item())
        ls = float(loss_stress.item())
        lid = float(loss_id.item())
        lsm = float(loss_smooth.item())
        lteach = float(loss_teacher.item())

        epoch_loss += li
        epoch_stress += ls
        epoch_id += lid
        epoch_smooth += lsm
        epoch_teacher += lteach
        n_steps += 1

        pbar.set_postfix(loss=f"{li:.4f}", stress=f"{ls:.4f}", teacher=f"{lteach:.4f}", sigma=f"{sigma_batch:.2e}")

    if n_steps == 0:
        raise RuntimeError("No valid optimization step in epoch. Check split/settings.")

    return {
        "loss": epoch_loss / n_steps,
        "stress": epoch_stress / n_steps,
        "id": epoch_id / n_steps,
        "smooth": epoch_smooth / n_steps,
        "teacher": epoch_teacher / n_steps,
    }


def _append_eval_outputs(
    run_dir: Path,
    robust_csv: Path,
    epoch: int,
    eval_pack: Dict[str, object],
) -> None:
    detail_path = run_dir / f"robustness_epoch{epoch:03d}.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(eval_pack, f, indent=2)

    with open(robust_csv, "a", encoding="utf-8") as f:
        clean = eval_pack["clean"]
        f.write(
            f"{epoch},0.0,1,{float(clean['spearman']):.6f},{float(clean['pearson']):.6f},"
            f"1.0,{float(clean['gate_mean']):.6f},{float(clean['intra_mean']):.6e},{int(clean['n_eval'])}\n"
        )
        for row in eval_pack["noisy"]:
            f.write(
                f"{epoch},{float(row['sigma']):.8e},0,{float(row['spearman']):.6f},"
                f"{float(row['pearson']):.6f},{float(row['ratio']):.6f},"
                f"{float(row['gate_mean']):.6f},{float(row['intra_mean']):.6e},{int(row['n_eval'])}\n"
            )


def run_training(args: argparse.Namespace) -> None:
    teacher_enabled = bool(str(args.teacher_checkpoint).strip()) and float(args.lambda_teacher) > 0.0
    if str(args.teacher_checkpoint).strip() and float(args.lambda_teacher) <= 0.0:
        raise ValueError("--teacher_checkpoint requires --lambda_teacher > 0")
    if float(args.lambda_teacher) > 0.0 and not str(args.teacher_checkpoint).strip():
        raise ValueError("--lambda_teacher > 0 requires --teacher_checkpoint")

    noise_modes = parse_noise_modes(args.noise_modes)
    noise_mode_probs = parse_noise_mode_weights(args.noise_mode_weights, noise_modes)
    perturbation = PerturbationParams.from_namespace(args)
    seed_everything(args.seed)

    device = _configure_device(args.device)
    run_dir = make_run_dir(args, noise_modes=noise_modes, noise_mode_probs=noise_mode_probs)
    _write_launch_summary(run_dir, args)
    log_csv = run_dir / "train_log.csv"
    robust_csv = run_dir / "robustness_grid.csv"
    _write_log_headers(log_csv, robust_csv)

    print(f"Device={device}")
    print(f"Run dir: {run_dir}")
    print(f"Model: {args.model}")
    print(f"Noise modes: {noise_modes}")
    print("Noise mode probs: {" + ", ".join(f"{mode}={prob:.3f}" for mode, prob in zip(noise_modes, noise_mode_probs)) + "}")

    dataset = GTReadyDataset(args.data_dir)
    subj_map = build_subject_map(dataset.files, subject_re=SUBJECT_RE_ANY)
    gt_matrix, name_to_idx = load_gt_distance_matrix(args.dist_npz, dtype=np.float64)

    subjects = sorted([sid for sid in subj_map.keys() if sid in name_to_idx])
    train_subjects, eval_subjects = rebuild_subject_split(
        subjects=subjects,
        eval_fraction=0.2,
        seed=args.seed,
        max_subjects=0,
    )
    online_eval_subjects = _select_online_eval_subjects(
        eval_subjects=eval_subjects,
        max_subjects_eval_train=args.max_subjects_eval_train,
        seed=args.seed,
    )
    sigma_grid = build_sigma_grid(args.sigma_min_eval, args.sigma_max_eval, args.n_sigma_eval)
    online_eval_plan = build_eval_plan(
        subj_map=subj_map,
        eval_subjects=online_eval_subjects,
        max_meshes_per_subject_eval=int(args.max_meshes_per_subject_eval),
        seed=int(args.seed) + 91_000,
    )
    online_eval_sample_cache = (
        preload_eval_samples(
            dataset=dataset,
            eval_plan=online_eval_plan,
            workers=int(args.preload_eval_workers_train),
        )
        if bool(args.preload_eval_samples_train)
        else None
    )
    online_eval_summary = summarize_eval_plan(online_eval_plan)
    with open(run_dir / "online_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_subjects_selected": int(len(online_eval_subjects)),
                "selected_subjects": list(online_eval_subjects),
                "plan_summary": online_eval_summary,
                "preloaded_samples": int(len(online_eval_sample_cache or {})),
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(
        "Online eval: "
        f"subjects={len(online_eval_subjects)}/{len(eval_subjects)} "
        f"meshes={online_eval_summary['total_meshes']} "
        f"mean_meshes={online_eval_summary['mean_meshes_per_subject']:.2f} "
        f"preloaded={len(online_eval_sample_cache or {})}"
    )

    model = build_model(args=args, device=device)
    if str(args.init_checkpoint).strip():
        init_checkpoint_path = _resolve_init_checkpoint(args.init_checkpoint)
        init_pack = _load_checkpoint_bundle(init_checkpoint_path)
        state_dict = init_pack["state_dict"] if isinstance(init_pack, dict) and "state_dict" in init_pack else init_pack
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"{init_checkpoint_path} does not contain a valid state_dict")
        model.load_state_dict(state_dict, strict=True)
        init_epoch = int(init_pack.get("epoch", -1)) if isinstance(init_pack, dict) else -1
        init_msg = f"Initialized weights from {init_checkpoint_path}"
        if init_epoch >= 0:
            init_msg += f" (epoch {init_epoch})"
        print(init_msg)
        (run_dir / "init_checkpoint.txt").write_text(init_msg + "\n", encoding="utf-8")

    teacher_model: nn.Module | None = None
    if teacher_enabled:
        teacher_model, teacher_checkpoint_path, teacher_args, teacher_epoch = _load_frozen_teacher(args.teacher_checkpoint, device=device)
        teacher_latent_dim = getattr(teacher_args, "latent_dim", None)
        if teacher_latent_dim is not None and int(teacher_latent_dim) != int(args.latent_dim):
            raise RuntimeError(
                f"Teacher latent_dim={int(teacher_latent_dim)} does not match student latent_dim={int(args.latent_dim)}"
            )
        teacher_msg = (
            f"Teacher checkpoint: {teacher_checkpoint_path}\n"
            f"Teacher model: {getattr(teacher_args, 'model', 'unknown')}\n"
            f"Teacher epoch: {teacher_epoch}\n"
            f"lambda_teacher: {float(args.lambda_teacher):.6f}\n"
        )
        print(teacher_msg.strip())
        (run_dir / "teacher_checkpoint.txt").write_text(teacher_msg, encoding="utf-8")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    eval_ctx = SubjectEvalContext(
        dataset=dataset,
        subj_map=subj_map,
        eval_subjects=online_eval_subjects,
        name_to_idx=name_to_idx,
        gt_matrix=gt_matrix,
        device=device,
        max_meshes_per_subject_eval=args.max_meshes_per_subject_eval,
        eval_plan=online_eval_plan,
        sample_cache=online_eval_sample_cache,
    )

    best_auc = -1e9
    best_epoch = -1
    best_ckpt_path = run_dir / "checkpoints" / "best_by_auc.pth"
    best_clean_sp = -1e9
    best_clean_epoch = -1
    best_clean_ckpt_path = run_dir / "checkpoints" / "best_by_clean.pth"
    last_eval = {
        "spearman_clean": float("nan"),
        "pearson_clean": float("nan"),
        "auc_r": float("nan"),
        "gate_mean_clean": float("nan"),
        "gate_mean_noisy_max": float("nan"),
        "spearman_noisy_max": float("nan"),
        "ratio_noisy_max": float("nan"),
        "intra_clean": float("nan"),
        "n_eval": 0,
    }

    for epoch in range(1, args.epochs + 1):
        epoch_stats = _train_epoch(
            model=model,
            dataset=dataset,
            subj_map=subj_map,
            train_subjects=train_subjects,
            name_to_idx=name_to_idx,
            gt_matrix=gt_matrix,
            device=device,
            optimizer=optimizer,
            teacher_model=teacher_model,
            epoch=epoch,
            args=args,
            noise_modes=noise_modes,
            noise_mode_probs=noise_mode_probs,
            perturbation=perturbation,
        )
        scheduler.step(epoch_stats["loss"])
        lr_now = float(optimizer.param_groups[0]["lr"])

        do_eval = args.eval_every > 0 and (epoch % args.eval_every == 0 or epoch == args.epochs)
        if do_eval:
            eval_pack = evaluate_subject_robustness_grid(
                model=model,
                eval_ctx=eval_ctx,
                sigma_grid=sigma_grid,
                noise_modes=noise_modes,
                params=perturbation,
                seed=args.seed + 50_000 + epoch,
                eval_mode=args.eval_mode,
            )
            last_eval = {
                "spearman_clean": float(eval_pack["spearman_clean"]),
                "pearson_clean": float(eval_pack["pearson_clean"]),
                "auc_r": float(eval_pack["auc_r"]),
                "gate_mean_clean": float(eval_pack["gate_mean_clean"]),
                "gate_mean_noisy_max": float(eval_pack["gate_mean_noisy_max"]),
                "spearman_noisy_max": float(eval_pack["spearman_noisy_max"]),
                "ratio_noisy_max": float(eval_pack["ratio_noisy_max"]),
                "intra_clean": float(eval_pack["clean"]["intra_mean"]),
                "n_eval": int(eval_pack["n_eval"]),
            }
            _append_eval_outputs(run_dir=run_dir, robust_csv=robust_csv, epoch=epoch, eval_pack=eval_pack)

            auc_now = float(last_eval["auc_r"])
            if np.isfinite(auc_now) and auc_now > best_auc:
                best_auc = auc_now
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": vars(args),
                        "best_auc_r": best_auc,
                        "sigma_grid": sigma_grid,
                    },
                    best_ckpt_path,
                )
                (run_dir / "best_by_auc.txt").write_text(
                    f"best_epoch={best_epoch}\nbest_auc_r={best_auc}\n",
                    encoding="utf-8",
                )

            clean_sp_now = float(last_eval["spearman_clean"])
            if np.isfinite(clean_sp_now) and clean_sp_now > best_clean_sp:
                best_clean_sp = clean_sp_now
                best_clean_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": vars(args),
                        "best_clean_spearman": best_clean_sp,
                        "sigma_grid": sigma_grid,
                    },
                    best_clean_ckpt_path,
                )
                (run_dir / "best_by_clean.txt").write_text(
                    f"best_epoch={best_clean_epoch}\nbest_spearman_clean={best_clean_sp}\n",
                    encoding="utf-8",
                )

        print(
            f"Epoch {epoch:03d} | loss={epoch_stats['loss']:.4f} stress={epoch_stats['stress']:.4f} "
            f"id={epoch_stats['id']:.4f} smooth={epoch_stats['smooth']:.4f} teacher={epoch_stats['teacher']:.4f} lr={lr_now:.2e} "
            f"sp_clean={last_eval['spearman_clean']:.4f} aucR={last_eval['auc_r']:.4f} "
            f"g0={last_eval['gate_mean_clean']:.3f} gmax={last_eval['gate_mean_noisy_max']:.3f} "
            f"n_eval={int(last_eval['n_eval'])}"
        )

        with open(log_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{epoch_stats['loss']:.6f},{epoch_stats['stress']:.6f},"
                f"{epoch_stats['id']:.6f},{epoch_stats['smooth']:.6f},{epoch_stats['teacher']:.6f},{lr_now:.2e},"
                f"{last_eval['spearman_clean']:.6f},{last_eval['pearson_clean']:.6f},{last_eval['auc_r']:.6f},"
                f"{last_eval['gate_mean_clean']:.6f},{last_eval['gate_mean_noisy_max']:.6f},"
                f"{last_eval['spearman_noisy_max']:.6f},{last_eval['ratio_noisy_max']:.6f},"
                f"{last_eval['intra_clean']:.6e},{int(last_eval['n_eval'])}\n"
            )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = run_dir / "checkpoints" / f"epoch{epoch:03d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                },
                ckpt,
            )

    print("DONE")
    print(f"Saved in: {run_dir}")
    print(f"Best AUC(R)={best_auc:.4f} at epoch {best_epoch} -> {best_ckpt_path}")
    print(f"Best clean Spearman={best_clean_sp:.4f} at epoch {best_clean_epoch} -> {best_clean_ckpt_path}")


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
