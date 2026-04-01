from __future__ import annotations

import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .data_utils import EvalSampleRecord, GTReadyDataset, sample_to_device
from .model_helpers import forward_model
from .noise import (
    PerturbationParams,
    apply_xyz_perturbation_with_params,
    rigid_angle_max_deg_from_sigma,
    rigid_trans_axis_std_from_sigma,
)
from .paths import ensure_autoencoder_dir_on_syspath


ensure_autoencoder_dir_on_syspath()

from intrinsic_utils import pearson_corr, sample_mesh_indices, spearman_corr  # noqa: E402


ALLOWED_METRICS = ("latent", "chamfer")
ALLOWED_PAIR_MODES = ("all", "within_topology", "cross_topology")
ALLOWED_AGGREGATION_LEVELS = ("mesh_pair", "subject_pair_mean", "subject_pair_median")


@dataclass(frozen=True)
class SubjectEvalContext:
    dataset: GTReadyDataset
    subj_map: Dict[str, List[int]]
    eval_subjects: Sequence[str]
    name_to_idx: Dict[str, int]
    gt_matrix: np.ndarray
    device: torch.device
    max_meshes_per_subject_eval: int
    eval_plan: Dict[str, List[int]] | None = None
    sample_cache: Dict[int, Dict[str, torch.Tensor]] | None = None


@dataclass(frozen=True)
class PairEvalContext:
    sample_records: List[EvalSampleRecord]
    kept_subjects: List[str]
    topology_labels: List[str]
    pair_i_cpu: np.ndarray
    pair_j_cpu: np.ndarray
    pair_i: torch.Tensor
    pair_j: torch.Tensor
    gt_vals: np.ndarray
    aggregation_level: str
    pair_mode: str
    pair_counts_by_mode: Dict[str, int]
    subject_pair_counts_by_mode: Dict[str, int]
    pair_count: int
    mesh_pair_count: int
    subject_pair_count: int
    pair_group_members: List[np.ndarray]
    n_samples: int
    n_subjects: int
    n_topology_labels: int


def build_sigma_grid(sigma_min: float, sigma_max: float, n_levels: int) -> List[float]:
    n_levels = max(int(n_levels), 1)
    smin = max(float(sigma_min), 1e-12)
    smax = max(float(sigma_max), smin)
    if n_levels == 1 or smax <= smin:
        return [smin]
    return [float(x) for x in np.logspace(np.log10(smin), np.log10(smax), num=n_levels)]


def _curve_integral(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def ratio_auc(sigmas: Sequence[float], ratios: Sequence[float]) -> float:
    s = np.asarray(sigmas, dtype=np.float64)
    r = np.asarray(ratios, dtype=np.float64)
    mask = np.isfinite(s) & np.isfinite(r)
    if int(mask.sum()) < 2:
        return float("nan")

    x = np.log10(np.clip(s[mask], 1e-12, None))
    y = r[mask]

    x_range = float(x.max() - x.min())
    if x_range < 1e-12:
        x_norm = np.linspace(0.0, 1.0, num=x.shape[0], dtype=np.float64)
    else:
        x_norm = (x - x.min()) / x_range
    return _curve_integral(y, x_norm)


def first_sigma_below(noisy_rows: Sequence[Dict[str, float]], threshold: float) -> float:
    thr = float(threshold)
    for row in noisy_rows:
        ratio = float(row.get("ratio", float("nan")))
        sigma = float(row.get("sigma", float("nan")))
        if math.isfinite(ratio) and math.isfinite(sigma) and ratio <= thr:
            return sigma
    return float("nan")


def finite_nanmean(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def summarize_eval_plan(eval_plan: Dict[str, List[int]]) -> Dict[str, float]:
    counts = [len(v) for v in eval_plan.values() if v]
    if not counts:
        return {
            "n_subjects_with_meshes": 0,
            "total_meshes": 0,
            "min_meshes_per_subject": 0,
            "max_meshes_per_subject": 0,
            "mean_meshes_per_subject": 0.0,
        }
    arr = np.asarray(counts, dtype=np.float64)
    return {
        "n_subjects_with_meshes": int(arr.size),
        "total_meshes": int(arr.sum()),
        "min_meshes_per_subject": int(arr.min()),
        "max_meshes_per_subject": int(arr.max()),
        "mean_meshes_per_subject": float(arr.mean()),
    }


def _apply_optional_perturbation(
    V: torch.Tensor,
    sigma: float,
    noise_modes: Sequence[str],
    params: PerturbationParams,
    rng: np.random.Generator,
    eval_mode: str,
    fixed_mode: str,
) -> torch.Tensor:
    if sigma <= 0.0:
        return V
    if eval_mode == "random":
        mode = noise_modes[int(rng.integers(0, len(noise_modes)))]
    else:
        mode = fixed_mode
    return apply_xyz_perturbation_with_params(V=V, mode=mode, sigma=sigma, params=params)


def _eval_indices_for_subject(
    eval_ctx: SubjectEvalContext,
    sid: str,
    rng: np.random.Generator,
) -> List[int]:
    if eval_ctx.eval_plan is not None:
        return [int(idx) for idx in eval_ctx.eval_plan.get(sid, [])]
    return sample_mesh_indices(
        eval_ctx.subj_map[sid],
        max_meshes=eval_ctx.max_meshes_per_subject_eval,
        seed=int(rng.integers(0, 2_000_000_000)),
    )


def _load_eval_sample(
    eval_ctx: SubjectEvalContext,
    idx: int,
) -> Dict[str, torch.Tensor]:
    if eval_ctx.sample_cache is not None and int(idx) in eval_ctx.sample_cache:
        return eval_ctx.sample_cache[int(idx)]
    return eval_ctx.dataset[int(idx)]


@torch.inference_mode()
def evaluate_subjects_at_sigma(
    model: nn.Module,
    eval_ctx: SubjectEvalContext,
    sigma: float,
    noise_modes: Sequence[str],
    params: PerturbationParams,
    seed: int,
    eval_mode: str,
    fixed_mode: str,
) -> Dict[str, float]:
    model.eval()

    rng = np.random.default_rng(seed)
    subj_means: Dict[str, torch.Tensor] = {}
    intra_vals: List[float] = []
    gate_vals: List[float] = []

    for sid in eval_ctx.eval_subjects:
        idxs = _eval_indices_for_subject(eval_ctx=eval_ctx, sid=sid, rng=rng)
        if not idxs:
            continue

        z_list: List[torch.Tensor] = []
        for idx in idxs:
            sample = _load_eval_sample(eval_ctx=eval_ctx, idx=int(idx))
            sample_d = sample_to_device(sample, device=eval_ctx.device)
            V_in = _apply_optional_perturbation(
                V=sample_d["verts"],
                sigma=sigma,
                noise_modes=noise_modes,
                params=params,
                rng=rng,
                eval_mode=eval_mode,
                fixed_mode=fixed_mode,
            )
            z, gate_info = forward_model(
                model=model,
                sample_dict=sample_d,
                V_in=V_in,
                return_gate_info=True,
                add_noise=False,
            )
            z_list.append(z.squeeze(0))
            gate_vals.append(float(gate_info["g_mean"].item()))

        if not z_list:
            continue

        Zs = torch.stack(z_list, dim=0)
        Zm = Zs.mean(dim=0)
        subj_means[sid] = Zm
        intra_vals.append(float(((Zs - Zm) ** 2).mean().item()))

    kept = [sid for sid in eval_ctx.eval_subjects if sid in eval_ctx.name_to_idx and sid in subj_means]
    if len(kept) < 3:
        return {
            "sigma": float(sigma),
            "spearman": float("nan"),
            "pearson": float("nan"),
            "intra_mean": float("nan"),
            "gate_mean": float(np.mean(gate_vals)) if gate_vals else float("nan"),
            "n_eval": int(len(kept)),
        }

    Z = torch.stack([subj_means[sid] for sid in kept], dim=0)
    gt_idx = np.asarray([eval_ctx.name_to_idx[sid] for sid in kept], dtype=int)

    D_gt = torch.tensor(eval_ctx.gt_matrix[np.ix_(gt_idx, gt_idx)], device=eval_ctx.device, dtype=Z.dtype)
    D_lat = torch.cdist(Z, Z, p=2)

    iu = torch.triu_indices(D_gt.shape[0], D_gt.shape[1], offset=1, device=eval_ctx.device)
    gt_vals = D_gt[iu[0], iu[1]].detach().cpu().numpy()
    lat_vals = D_lat[iu[0], iu[1]].detach().cpu().numpy()

    return {
        "sigma": float(sigma),
        "spearman": float(spearman_corr(gt_vals, lat_vals)),
        "pearson": float(pearson_corr(gt_vals, lat_vals)),
        "intra_mean": float(np.mean(intra_vals)) if intra_vals else float("nan"),
        "gate_mean": float(np.mean(gate_vals)) if gate_vals else float("nan"),
        "n_eval": int(len(kept)),
    }


@torch.inference_mode()
def evaluate_subject_robustness_grid(
    model: nn.Module,
    eval_ctx: SubjectEvalContext,
    sigma_grid: Sequence[float],
    noise_modes: Sequence[str],
    params: PerturbationParams,
    seed: int,
    eval_mode: str,
) -> Dict[str, object]:
    clean = evaluate_subjects_at_sigma(
        model=model,
        eval_ctx=eval_ctx,
        sigma=0.0,
        noise_modes=noise_modes,
        params=params,
        seed=seed,
        eval_mode="fixed",
        fixed_mode=noise_modes[0],
    )

    clean_sp = float(clean["spearman"])
    noisy_rows: List[Dict[str, float]] = []
    ratios: List[float] = []
    fixed_mode = noise_modes[0]

    for i, sigma in enumerate(sigma_grid):
        if eval_mode == "average":
            per_mode = []
            for j, mode in enumerate(noise_modes):
                per_mode.append(
                    evaluate_subjects_at_sigma(
                        model=model,
                        eval_ctx=eval_ctx,
                        sigma=float(sigma),
                        noise_modes=noise_modes,
                        params=params,
                        seed=seed + 10_000 * (i + 1) + j,
                        eval_mode="fixed",
                        fixed_mode=mode,
                    )
                )
            row = {
                "sigma": float(sigma),
                "spearman": float(np.nanmean([float(r["spearman"]) for r in per_mode])),
                "pearson": float(np.nanmean([float(r["pearson"]) for r in per_mode])),
                "intra_mean": float(np.nanmean([float(r["intra_mean"]) for r in per_mode])),
                "gate_mean": float(np.nanmean([float(r["gate_mean"]) for r in per_mode])),
                "n_eval": int(np.nanmin([float(r["n_eval"]) for r in per_mode])),
            }
        else:
            row = evaluate_subjects_at_sigma(
                model=model,
                eval_ctx=eval_ctx,
                sigma=float(sigma),
                noise_modes=noise_modes,
                params=params,
                seed=seed + 1000 + i,
                eval_mode=eval_mode,
                fixed_mode=fixed_mode,
            )

        sp = float(row["spearman"])
        ratio = float(sp / clean_sp) if np.isfinite(clean_sp) and abs(clean_sp) > 1e-12 and np.isfinite(sp) else float("nan")
        row["ratio"] = ratio
        noisy_rows.append(row)
        ratios.append(ratio)

    gate_mean_noisy_max = float(noisy_rows[-1]["gate_mean"]) if noisy_rows else float("nan")
    spearman_noisy_max = float(noisy_rows[-1]["spearman"]) if noisy_rows else float("nan")
    ratio_noisy_max = float(noisy_rows[-1]["ratio"]) if noisy_rows else float("nan")
    auc_r = ratio_noisy_max if len(sigma_grid) < 2 else ratio_auc(sigmas=sigma_grid, ratios=ratios)

    return {
        "clean": clean,
        "noisy": noisy_rows,
        "auc_r": float(auc_r),
        "spearman_clean": float(clean_sp),
        "pearson_clean": float(clean["pearson"]),
        "gate_mean_clean": float(clean["gate_mean"]),
        "gate_mean_noisy_max": gate_mean_noisy_max,
        "spearman_noisy_max": spearman_noisy_max,
        "ratio_noisy_max": ratio_noisy_max,
        "n_eval": int(clean["n_eval"]),
    }


def subject_pair_key(subject_a: str, subject_b: str) -> tuple[str, str]:
    if str(subject_a) <= str(subject_b):
        return str(subject_a), str(subject_b)
    return str(subject_b), str(subject_a)


def count_unique_subject_pairs(subject_ids_a: Sequence[object], subject_ids_b: Sequence[object]) -> int:
    return int(len({subject_pair_key(str(a), str(b)) for a, b in zip(subject_ids_a, subject_ids_b)}))


def build_subject_pair_group_members(
    subject_ids_a: Sequence[object],
    subject_ids_b: Sequence[object],
) -> tuple[list[tuple[str, str]], list[np.ndarray]]:
    group_lookup: Dict[tuple[str, str], int] = {}
    group_keys: list[tuple[str, str]] = []
    group_members: list[list[int]] = []

    for mesh_pair_idx, (subject_a, subject_b) in enumerate(zip(subject_ids_a, subject_ids_b)):
        key = subject_pair_key(str(subject_a), str(subject_b))
        group_idx = group_lookup.get(key)
        if group_idx is None:
            group_idx = len(group_keys)
            group_lookup[key] = group_idx
            group_keys.append(key)
            group_members.append([])
        group_members[group_idx].append(int(mesh_pair_idx))

    return group_keys, [np.asarray(members, dtype=np.int64) for members in group_members]


def build_pair_eval_context(
    sample_records: Sequence[EvalSampleRecord],
    name_to_idx: Dict[str, int],
    gt_matrix: np.ndarray,
    device: torch.device,
    pair_mode: str,
    aggregation_level: str,
) -> PairEvalContext:
    if pair_mode not in ALLOWED_PAIR_MODES:
        raise ValueError(f"Unsupported pair_mode={pair_mode}. Expected one of {list(ALLOWED_PAIR_MODES)}")
    if aggregation_level not in ALLOWED_AGGREGATION_LEVELS:
        raise ValueError(
            f"Unsupported aggregation_level={aggregation_level}. Expected one of {list(ALLOWED_AGGREGATION_LEVELS)}"
        )

    kept_records = [rec for rec in sample_records if rec.subject_id in name_to_idx]
    kept_subjects = sorted({rec.subject_id for rec in kept_records})
    kept_labels = sorted({rec.topology_label for rec in kept_records})

    if len(kept_records) < 2:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return PairEvalContext(
            sample_records=kept_records,
            kept_subjects=kept_subjects,
            topology_labels=kept_labels,
            pair_i_cpu=np.empty((0,), dtype=np.int64),
            pair_j_cpu=np.empty((0,), dtype=np.int64),
            pair_i=empty,
            pair_j=empty,
            gt_vals=np.empty((0,), dtype=np.float64),
            aggregation_level=str(aggregation_level),
            pair_mode=str(pair_mode),
            pair_counts_by_mode={mode: 0 for mode in ALLOWED_PAIR_MODES},
            subject_pair_counts_by_mode={mode: 0 for mode in ALLOWED_PAIR_MODES},
            pair_count=0,
            mesh_pair_count=0,
            subject_pair_count=0,
            pair_group_members=[],
            n_samples=int(len(kept_records)),
            n_subjects=int(len(kept_subjects)),
            n_topology_labels=int(len(kept_labels)),
        )

    subject_ids = np.asarray([rec.subject_id for rec in kept_records], dtype=object)
    topology_labels = np.asarray([rec.topology_label for rec in kept_records], dtype=object)
    tri_i, tri_j = np.triu_indices(len(kept_records), k=1)

    cross_subject_mask = subject_ids[tri_i] != subject_ids[tri_j]
    same_topology_mask = topology_labels[tri_i] == topology_labels[tri_j]
    mask_by_mode = {
        "all": cross_subject_mask,
        "within_topology": cross_subject_mask & same_topology_mask,
        "cross_topology": cross_subject_mask & (~same_topology_mask),
    }
    selected_mask = mask_by_mode[pair_mode]

    subject_ids_a_all = subject_ids[tri_i]
    subject_ids_b_all = subject_ids[tri_j]

    pair_i_np = tri_i[selected_mask]
    pair_j_np = tri_j[selected_mask]
    mesh_pair_count = int(pair_i_np.size)
    sample_gt_idx = np.asarray([name_to_idx[rec.subject_id] for rec in kept_records], dtype=int)

    selected_subject_ids_a = subject_ids[pair_i_np]
    selected_subject_ids_b = subject_ids[pair_j_np]
    if aggregation_level == "mesh_pair":
        gt_vals = np.asarray(gt_matrix[sample_gt_idx[pair_i_np], sample_gt_idx[pair_j_np]], dtype=np.float64)
        pair_count = mesh_pair_count
        subject_pair_count = count_unique_subject_pairs(selected_subject_ids_a, selected_subject_ids_b)
        pair_group_members: list[np.ndarray] = []
    else:
        subject_pair_keys, pair_group_members = build_subject_pair_group_members(
            subject_ids_a=selected_subject_ids_a,
            subject_ids_b=selected_subject_ids_b,
        )
        gt_vals = np.asarray(
            [gt_matrix[name_to_idx[sid_a], name_to_idx[sid_b]] for sid_a, sid_b in subject_pair_keys],
            dtype=np.float64,
        )
        pair_count = int(len(subject_pair_keys))
        subject_pair_count = int(len(subject_pair_keys))

    return PairEvalContext(
        sample_records=kept_records,
        kept_subjects=kept_subjects,
        topology_labels=kept_labels,
        pair_i_cpu=pair_i_np.astype(np.int64, copy=False),
        pair_j_cpu=pair_j_np.astype(np.int64, copy=False),
        pair_i=torch.as_tensor(pair_i_np, device=device, dtype=torch.long),
        pair_j=torch.as_tensor(pair_j_np, device=device, dtype=torch.long),
        gt_vals=gt_vals,
        aggregation_level=str(aggregation_level),
        pair_mode=str(pair_mode),
        pair_counts_by_mode={mode: int(mask.sum()) for mode, mask in mask_by_mode.items()},
        subject_pair_counts_by_mode={
            mode: count_unique_subject_pairs(subject_ids_a_all[mask], subject_ids_b_all[mask])
            for mode, mask in mask_by_mode.items()
        },
        pair_count=int(pair_count),
        mesh_pair_count=int(mesh_pair_count),
        subject_pair_count=int(subject_pair_count),
        pair_group_members=pair_group_members,
        n_samples=int(len(kept_records)),
        n_subjects=int(len(kept_subjects)),
        n_topology_labels=int(len(kept_labels)),
    )


def symmetric_chamfer_two_sets(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != 3 or Y.shape[1] != 3:
        raise ValueError(f"Expected point sets [N,3] and [M,3], got {tuple(X.shape)} and {tuple(Y.shape)}")
    if int(X.shape[0]) == 0 or int(Y.shape[0]) == 0:
        return torch.tensor(float("nan"), device=X.device, dtype=X.dtype)

    n_x = int(X.shape[0])
    n_y = int(Y.shape[0])
    chunk_y = max(32, min(1024, n_y))
    chunk_x = max(32, min(1024, n_x))

    min_xy = torch.full((n_x,), float("inf"), device=X.device, dtype=X.dtype)
    min_yx = torch.full((n_y,), float("inf"), device=Y.device, dtype=Y.dtype)

    for start in range(0, n_y, chunk_y):
        stop = min(start + chunk_y, n_y)
        d2 = torch.cdist(X, Y[start:stop], p=2) ** 2
        min_xy = torch.minimum(min_xy, d2.min(dim=1).values)

    for start in range(0, n_x, chunk_x):
        stop = min(start + chunk_x, n_x)
        d2 = torch.cdist(Y, X[start:stop], p=2) ** 2
        min_yx = torch.minimum(min_yx, d2.min(dim=1).values)

    return min_xy.mean() + min_yx.mean()


def symmetric_chamfer_same_shape_batch(
    X: torch.Tensor,
    Y: torch.Tensor,
    max_cdist_mb: float = 512.0,
) -> torch.Tensor:
    if X.ndim != 3 or Y.ndim != 3 or X.shape[0] != Y.shape[0] or X.shape[2] != 3 or Y.shape[2] != 3:
        raise ValueError(f"Expected batched point sets [B,N,3] and [B,M,3], got {tuple(X.shape)} and {tuple(Y.shape)}")
    if int(X.shape[0]) == 0:
        return torch.empty((0,), device=X.device, dtype=X.dtype)
    if int(X.shape[1]) == 0 or int(Y.shape[1]) == 0:
        return torch.full((int(X.shape[0]),), float("nan"), device=X.device, dtype=X.dtype)

    batch_size = int(X.shape[0])
    n_x = int(X.shape[1])
    n_y = int(Y.shape[1])
    chunk_y = max(32, min(1024, n_y))
    chunk_x = max(32, min(1024, n_x))

    bytes_per_dist_forward = max(1, n_x * chunk_y * int(X.element_size()))
    bytes_per_dist_reverse = max(1, n_y * chunk_x * int(X.element_size()))
    bytes_per_pair = max(bytes_per_dist_forward, bytes_per_dist_reverse)
    max_cdist_bytes = max(int(64 * (1024**2)), int(float(max_cdist_mb) * (1024**2)))
    max_pairs_per_call = max(1, min(batch_size, max_cdist_bytes // bytes_per_pair))
    if max_pairs_per_call < batch_size:
        parts = []
        for start in range(0, batch_size, max_pairs_per_call):
            stop = min(start + max_pairs_per_call, batch_size)
            parts.append(symmetric_chamfer_same_shape_batch(X[start:stop], Y[start:stop], max_cdist_mb=max_cdist_mb))
        return torch.cat(parts, dim=0)

    min_xy = torch.full((batch_size, n_x), float("inf"), device=X.device, dtype=X.dtype)
    min_yx = torch.full((batch_size, n_y), float("inf"), device=Y.device, dtype=Y.dtype)

    for start in range(0, n_y, chunk_y):
        stop = min(start + chunk_y, n_y)
        d2 = torch.cdist(X, Y[:, start:stop], p=2) ** 2
        min_xy = torch.minimum(min_xy, d2.min(dim=2).values)

    for start in range(0, n_x, chunk_x):
        stop = min(start + chunk_x, n_x)
        d2 = torch.cdist(Y, X[:, start:stop], p=2) ** 2
        min_yx = torch.minimum(min_yx, d2.min(dim=2).values)

    return min_xy.mean(dim=1) + min_yx.mean(dim=1)


def compute_pairwise_chamfer_values(
    vertex_sets: Sequence[torch.Tensor],
    pair_i: Sequence[int] | np.ndarray,
    pair_j: Sequence[int] | np.ndarray,
    batch_pairs: int,
    progress_desc: str = "",
    show_progress: bool = False,
) -> np.ndarray:
    n_pairs = int(len(pair_i))
    if n_pairs == 0:
        return np.empty((0,), dtype=np.float64)

    pair_batch = max(1, int(batch_pairs))
    pair_i_np = np.asarray(pair_i, dtype=np.int64)
    pair_j_np = np.asarray(pair_j, dtype=np.int64)
    values = np.empty((n_pairs,), dtype=np.float64)

    shape_groups: Dict[tuple[int, int], List[int]] = {}
    for pair_idx in range(n_pairs):
        Xi = vertex_sets[int(pair_i_np[pair_idx])]
        Yj = vertex_sets[int(pair_j_np[pair_idx])]
        key = (int(Xi.shape[0]), int(Yj.shape[0]))
        shape_groups.setdefault(key, []).append(int(pair_idx))

    total_batches = int(sum(math.ceil(len(indices) / pair_batch) for indices in shape_groups.values()))
    progress_label = str(progress_desc).strip() or "Chamfer pairs"
    processed_pairs = 0
    processed_batches = 0
    progress_start_time = time.time()
    progress_interval_s = 10.0
    next_progress_time = progress_start_time + progress_interval_s
    use_tqdm = bool(show_progress and total_batches > 1 and sys.stdout.isatty() and sys.stderr.isatty())
    if show_progress:
        print(
            f"{progress_label}: starting {n_pairs} pairs in {total_batches} batches",
            flush=True,
        )
    pair_pbar = (
        tqdm(
            total=total_batches,
            desc=progress_desc,
            dynamic_ncols=True,
            leave=False,
            position=2,
        )
        if use_tqdm
        else None
    )

    for shape_key, group_pair_indices in shape_groups.items():
        group_total = len(group_pair_indices)
        for start in range(0, group_total, pair_batch):
            stop = min(start + pair_batch, group_total)
            batch_pair_indices = group_pair_indices[start:stop]
            X_batch = torch.stack(
                [vertex_sets[int(pair_i_np[pair_idx])] for pair_idx in batch_pair_indices],
                dim=0,
            )
            Y_batch = torch.stack(
                [vertex_sets[int(pair_j_np[pair_idx])] for pair_idx in batch_pair_indices],
                dim=0,
            )
            batch_values = symmetric_chamfer_same_shape_batch(X=X_batch, Y=Y_batch)
            values[np.asarray(batch_pair_indices, dtype=np.int64)] = batch_values.detach().cpu().numpy()
            processed_pairs += len(batch_pair_indices)
            processed_batches += 1

            if pair_pbar is not None:
                pair_pbar.update(1)
                pair_pbar.set_postfix(shape=f"{shape_key[0]}x{shape_key[1]}", done=f"{stop}/{group_total}")
            if show_progress:
                now = time.time()
                should_log = (
                    processed_batches == 1
                    or processed_batches == total_batches
                    or now >= next_progress_time
                )
                if should_log:
                    elapsed_s = max(now - progress_start_time, 1.0e-9)
                    pairs_per_s = float(processed_pairs) / elapsed_s
                    eta_s = (
                        float(n_pairs - processed_pairs) / pairs_per_s
                        if pairs_per_s > 0.0
                        else float("inf")
                    )
                    if eta_s >= 3600.0:
                        eta_text = f"{eta_s / 3600.0:.1f}h"
                    elif eta_s >= 60.0:
                        eta_text = f"{eta_s / 60.0:.1f}m"
                    else:
                        eta_text = f"{eta_s:.1f}s"
                    if elapsed_s >= 3600.0:
                        elapsed_text = f"{elapsed_s / 3600.0:.1f}h"
                    elif elapsed_s >= 60.0:
                        elapsed_text = f"{elapsed_s / 60.0:.1f}m"
                    else:
                        elapsed_text = f"{elapsed_s:.1f}s"
                    while next_progress_time <= now:
                        next_progress_time += progress_interval_s
                    progress_pct = 100.0 * (float(processed_pairs) / float(n_pairs))
                    batch_pct = 100.0 * (float(processed_batches) / float(total_batches))
                    shape_text = f"{shape_key[0]}x{shape_key[1]}"
                    print(
                        f"{progress_label}: {progress_pct:.1f}% pairs "
                        f"({processed_pairs}/{n_pairs}) | "
                        f"{batch_pct:.1f}% batches ({processed_batches}/{total_batches}) | "
                        f"shape={shape_text} | rate={pairs_per_s:.1f} pairs/s | "
                        f"eta={eta_text} | elapsed={elapsed_text}",
                        flush=True,
                    )

    if pair_pbar is not None:
        pair_pbar.close()
    if show_progress:
        elapsed_s = time.time() - progress_start_time
        if elapsed_s >= 3600.0:
            elapsed_text = f"{elapsed_s / 3600.0:.1f}h"
        elif elapsed_s >= 60.0:
            elapsed_text = f"{elapsed_s / 60.0:.1f}m"
        else:
            elapsed_text = f"{elapsed_s:.1f}s"
        print(
            f"{progress_label}: completed {n_pairs} pairs in {elapsed_text}",
            flush=True,
        )
    return values


def parse_ratio_thresholds(text: str) -> List[float]:
    vals = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    if not vals:
        raise ValueError("ratio_thresholds must contain at least one value")
    return vals


def aggregate_pair_observations(mesh_pair_values: Sequence[float] | np.ndarray, pair_ctx: PairEvalContext) -> np.ndarray:
    values = np.asarray(mesh_pair_values, dtype=np.float64)
    if pair_ctx.aggregation_level == "mesh_pair":
        return values

    aggregated = np.full((len(pair_ctx.pair_group_members),), float("nan"), dtype=np.float64)
    for group_idx, member_indices in enumerate(pair_ctx.pair_group_members):
        member_values = values[np.asarray(member_indices, dtype=np.int64)]
        member_values = member_values[np.isfinite(member_values)]
        if member_values.size == 0:
            continue
        if pair_ctx.aggregation_level == "subject_pair_mean":
            aggregated[group_idx] = float(member_values.mean())
        elif pair_ctx.aggregation_level == "subject_pair_median":
            aggregated[group_idx] = float(np.median(member_values))
        else:
            raise ValueError(
                f"Unsupported aggregation_level={pair_ctx.aggregation_level}. "
                f"Expected one of {list(ALLOWED_AGGREGATION_LEVELS)}"
            )
    return aggregated


def _collect_metric_inputs(
    model: torch.nn.Module | None,
    dataset: GTReadyDataset,
    pair_ctx: PairEvalContext,
    sample_cache: Dict[int, Dict[str, torch.Tensor]] | None,
    device: torch.device,
    metric: str,
    sigma: float,
    noise_modes: Sequence[str],
    params: PerturbationParams,
    seed: int,
    eval_mode: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[float]]:
    rng = np.random.default_rng(seed)
    fixed_mode = noise_modes[0]
    latent_vectors: list[torch.Tensor] = []
    full_vertex_sets: list[torch.Tensor] = []
    gate_vals: list[float] = []

    for record in pair_ctx.sample_records:
        sample = sample_cache[int(record.dataset_idx)] if sample_cache is not None else dataset[int(record.dataset_idx)]
        if metric == "latent":
            sample_d = sample_to_device(sample, device=device)
            V = sample_d["verts"]
        else:
            sample_d = None
            verts = sample["verts"]
            V = verts if verts.device == device else verts.to(device, non_blocking=True)

        V_in = _apply_optional_perturbation(
            V=V,
            sigma=sigma,
            noise_modes=noise_modes,
            params=params,
            rng=rng,
            eval_mode=eval_mode,
            fixed_mode=fixed_mode,
        )

        if metric == "latent":
            z, gate_info = forward_model(
                model=model,  # type: ignore[arg-type]
                sample_dict=sample_d,
                V_in=V_in,
                return_gate_info=True,
                add_noise=False,
            )
            latent_vectors.append(z.squeeze(0))
            gate_vals.append(float(gate_info["g_mean"].item()))
        else:
            full_vertex_sets.append(V_in.contiguous())

    return latent_vectors, full_vertex_sets, gate_vals


def _compute_posthoc_metric_values(
    metric: str,
    pair_ctx: PairEvalContext,
    latent_vectors: Sequence[torch.Tensor],
    full_vertex_sets: Sequence[torch.Tensor],
    chamfer_batch_pairs: int,
    show_chamfer_pair_progress: bool,
    chamfer_progress_desc: str,
) -> tuple[np.ndarray, float]:
    if metric == "latent":
        if not latent_vectors:
            return np.empty((0,), dtype=np.float64), float("nan")
        Z = torch.stack(list(latent_vectors), dim=0)
        metric_vals = torch.linalg.vector_norm(
            Z.index_select(0, pair_ctx.pair_i) - Z.index_select(0, pair_ctx.pair_j),
            dim=1,
        )
        return (
            aggregate_pair_observations(metric_vals.detach().cpu().numpy().astype(np.float64, copy=False), pair_ctx=pair_ctx),
            float("nan"),
        )

    if not full_vertex_sets:
        return np.empty((0,), dtype=np.float64), float("nan")
    return (
        aggregate_pair_observations(
            compute_pairwise_chamfer_values(
                vertex_sets=full_vertex_sets,
                pair_i=pair_ctx.pair_i_cpu,
                pair_j=pair_ctx.pair_j_cpu,
                batch_pairs=chamfer_batch_pairs,
                progress_desc=chamfer_progress_desc,
                show_progress=show_chamfer_pair_progress,
            ),
            pair_ctx=pair_ctx,
        ),
        float("nan"),
    )


@torch.no_grad()
def evaluate_at_sigma_cached(
    model: torch.nn.Module | None,
    dataset: GTReadyDataset,
    pair_ctx: PairEvalContext,
    sample_cache: Dict[int, Dict[str, torch.Tensor]] | None,
    device: torch.device,
    metric: str,
    chamfer_batch_pairs: int,
    sigma: float,
    noise_modes: Sequence[str],
    params: PerturbationParams,
    seed: int,
    eval_mode: str,
    show_chamfer_pair_progress: bool = False,
    chamfer_progress_desc: str = "",
) -> Dict[str, float]:
    if metric not in ALLOWED_METRICS:
        raise ValueError(f"Unsupported metric={metric}. Expected one of {list(ALLOWED_METRICS)}")
    if metric == "latent" and model is None:
        raise ValueError("metric=latent requires a loaded model")
    if model is not None:
        model.eval()

    if pair_ctx.n_samples == 0 or pair_ctx.pair_count == 0:
        return {
            "sigma": float(sigma),
            "spearman": float("nan"),
            "pearson": float("nan"),
            "intra_mean": float("nan"),
            "gate_mean": float("nan"),
            "n_eval": pair_ctx.n_samples,
            "n_pairs": pair_ctx.pair_count,
            "n_mesh_pairs": pair_ctx.mesh_pair_count,
            "n_subject_pairs": pair_ctx.subject_pair_count,
            "n_subjects": pair_ctx.n_subjects,
        }

    t_start = time.perf_counter()
    latent_vectors, full_vertex_sets, gate_vals = _collect_metric_inputs(
        model=model,
        dataset=dataset,
        pair_ctx=pair_ctx,
        sample_cache=sample_cache,
        device=device,
        metric=metric,
        sigma=sigma,
        noise_modes=noise_modes,
        params=params,
        seed=seed,
        eval_mode=eval_mode,
    )
    dist_vals, gate_mean = _compute_posthoc_metric_values(
        metric=metric,
        pair_ctx=pair_ctx,
        latent_vectors=latent_vectors,
        full_vertex_sets=full_vertex_sets,
        chamfer_batch_pairs=chamfer_batch_pairs,
        show_chamfer_pair_progress=show_chamfer_pair_progress,
        chamfer_progress_desc=chamfer_progress_desc,
    )
    if metric == "latent":
        gate_mean = float(np.mean(gate_vals)) if gate_vals else float("nan")

    elapsed_sec = float(time.perf_counter() - t_start)
    if dist_vals.size == 0:
        spearman = float("nan")
        pearson = float("nan")
    else:
        spearman = float(spearman_corr(pair_ctx.gt_vals, dist_vals))
        pearson = float(pearson_corr(pair_ctx.gt_vals, dist_vals))
    return {
        "sigma": float(sigma),
        "spearman": spearman,
        "pearson": pearson,
        "intra_mean": float("nan"),
        "gate_mean": gate_mean,
        "elapsed_sec": elapsed_sec,
        "n_eval": pair_ctx.n_samples,
        "n_pairs": pair_ctx.pair_count,
        "n_mesh_pairs": pair_ctx.mesh_pair_count,
        "n_subject_pairs": pair_ctx.subject_pair_count,
        "n_subjects": pair_ctx.n_subjects,
    }


@torch.no_grad()
def evaluate_robustness_grid_tqdm(
    model: torch.nn.Module | None,
    dataset: GTReadyDataset,
    pair_ctx: PairEvalContext,
    sample_cache: Dict[int, Dict[str, torch.Tensor]] | None,
    device: torch.device,
    metric: str,
    chamfer_batch_pairs: int,
    sigma_grid: Sequence[float],
    noise_modes: Sequence[str],
    params: PerturbationParams,
    seed: int,
    eval_mode: str,
    progress_desc: str,
    show_chamfer_pair_progress: bool,
    precomputed_clean: Dict[str, float] | None = None,
) -> Dict[str, object]:
    pbar = tqdm(total=len(sigma_grid) + 1, desc=progress_desc, leave=False, dynamic_ncols=True)
    pbar.set_postfix(stage="clean")
    if precomputed_clean is None:
        clean = evaluate_at_sigma_cached(
            model=model,
            dataset=dataset,
            pair_ctx=pair_ctx,
            sample_cache=sample_cache,
            device=device,
            metric=metric,
            chamfer_batch_pairs=chamfer_batch_pairs,
            sigma=0.0,
            noise_modes=noise_modes,
            params=params,
            seed=seed,
            eval_mode="fixed",
            show_chamfer_pair_progress=show_chamfer_pair_progress,
            chamfer_progress_desc=f"{progress_desc} clean pairs",
        )
    else:
        clean = dict(precomputed_clean)
    pbar.update(1)

    clean_sp = float(clean["spearman"])
    noisy_rows: List[Dict[str, float]] = []
    ratios: List[float] = []

    for i, sigma in enumerate(sigma_grid):
        if eval_mode == "average":
            per_mode = []
            for j, mode in enumerate(noise_modes):
                per_mode.append(
                    evaluate_at_sigma_cached(
                        model=model,
                        dataset=dataset,
                        pair_ctx=pair_ctx,
                        sample_cache=sample_cache,
                        device=device,
                        metric=metric,
                        chamfer_batch_pairs=chamfer_batch_pairs,
                        sigma=float(sigma),
                        noise_modes=[mode],
                        params=params,
                        seed=seed + 10_000 * (i + 1) + j,
                        eval_mode="fixed",
                        show_chamfer_pair_progress=show_chamfer_pair_progress,
                        chamfer_progress_desc=f"{progress_desc} {mode} {float(sigma):.2e} pairs",
                    )
                )
            row = {
                "sigma": float(sigma),
                "spearman": finite_nanmean([float(r["spearman"]) for r in per_mode]),
                "pearson": finite_nanmean([float(r["pearson"]) for r in per_mode]),
                "intra_mean": float("nan"),
                "gate_mean": finite_nanmean([float(r["gate_mean"]) for r in per_mode]),
                "elapsed_sec": finite_nanmean([float(r.get("elapsed_sec", float("nan"))) for r in per_mode]),
                "n_eval": int(np.nanmin([float(r["n_eval"]) for r in per_mode])) if per_mode else 0,
                "n_pairs": int(np.nanmin([float(r["n_pairs"]) for r in per_mode])) if per_mode else 0,
                "n_mesh_pairs": int(np.nanmin([float(r["n_mesh_pairs"]) for r in per_mode])) if per_mode else 0,
                "n_subject_pairs": int(np.nanmin([float(r["n_subject_pairs"]) for r in per_mode])) if per_mode else 0,
                "n_subjects": int(np.nanmin([float(r["n_subjects"]) for r in per_mode])) if per_mode else 0,
            }
        else:
            row = evaluate_at_sigma_cached(
                model=model,
                dataset=dataset,
                pair_ctx=pair_ctx,
                sample_cache=sample_cache,
                device=device,
                metric=metric,
                chamfer_batch_pairs=chamfer_batch_pairs,
                sigma=float(sigma),
                noise_modes=noise_modes,
                params=params,
                seed=seed + 1000 + i,
                eval_mode=eval_mode,
                show_chamfer_pair_progress=show_chamfer_pair_progress,
                chamfer_progress_desc=f"{progress_desc} {float(sigma):.2e} pairs",
            )

        sp = float(row["spearman"])
        ratio = float(sp / clean_sp) if np.isfinite(clean_sp) and abs(clean_sp) > 1e-12 and np.isfinite(sp) else float("nan")
        row["ratio"] = ratio
        noisy_rows.append(row)
        ratios.append(ratio)
        pbar.update(1)
        pbar.set_postfix(
            stage="sigma",
            sigma=f"{float(sigma):.2e}",
            sp=f"{sp:.3f}" if math.isfinite(sp) else "nan",
            ratio=f"{ratio:.3f}" if math.isfinite(ratio) else "nan",
            sec=f"{float(row.get('elapsed_sec', float('nan'))):.1f}" if math.isfinite(float(row.get("elapsed_sec", float("nan")))) else "nan",
        )

    pbar.close()

    gate_mean_noisy_max = float(noisy_rows[-1]["gate_mean"]) if noisy_rows else float("nan")
    spearman_noisy_max = float(noisy_rows[-1]["spearman"]) if noisy_rows else float("nan")
    pearson_noisy_max = float(noisy_rows[-1]["pearson"]) if noisy_rows else float("nan")
    ratio_noisy_max = float(noisy_rows[-1]["ratio"]) if noisy_rows else float("nan")
    auc_r = ratio_noisy_max if len(sigma_grid) < 2 else ratio_auc(sigmas=sigma_grid, ratios=ratios)

    return {
        "clean": clean,
        "noisy": noisy_rows,
        "auc_r": float(auc_r),
        "metric": str(metric),
        "pair_mode": str(pair_ctx.pair_mode),
        "aggregation_level": str(pair_ctx.aggregation_level),
        "spearman_clean": float(clean_sp),
        "pearson_clean": float(clean["pearson"]),
        "gate_mean_clean": float(clean["gate_mean"]),
        "gate_mean_noisy_max": gate_mean_noisy_max,
        "spearman_noisy_max": spearman_noisy_max,
        "pearson_noisy_max": pearson_noisy_max,
        "ratio_noisy_max": ratio_noisy_max,
        "n_eval": int(clean["n_eval"]),
        "n_pairs": int(clean["n_pairs"]),
        "n_mesh_pairs": int(clean["n_mesh_pairs"]),
        "n_subject_pairs": int(clean["n_subject_pairs"]),
        "n_subjects": int(clean["n_subjects"]),
    }


def _l2_rms_from_axis_std(axis_std: float) -> float:
    return float(math.sqrt(3.0) * float(axis_std))


def _format_affine_sigma_formula(offset: float, slope: float) -> str:
    offset = float(offset)
    slope = float(slope)
    if abs(offset) <= 1e-12:
        return f"{slope:.6g} * sigma"
    sign = "+" if slope >= 0.0 else "-"
    return f"{offset:.6g} {sign} {abs(slope):.6g} * sigma"


def build_perturbation_descriptor(
    sigma: float,
    noise_modes: Sequence[str],
    eval_mode: str,
    params: PerturbationParams,
) -> Dict[str, object]:
    sigma = float(sigma)
    mode_list = [str(mode) for mode in noise_modes]
    desc: Dict[str, object] = {
        "noise_modes": ",".join(mode_list),
        "eval_mode": str(eval_mode),
        "coord_unit": "normalized_xyz",
        "jitter_axis_std": float("nan"),
        "jitter_l2_rms": float("nan"),
        "rigid_angle_max_deg": float("nan"),
        "rigid_angle_std_deg": float("nan"),
        "rigid_trans_axis_std": float("nan"),
        "rigid_trans_l2_rms": float("nan"),
        "outlier_frac": float("nan"),
        "outlier_disp_axis_std": float("nan"),
        "outlier_disp_l2_rms": float("nan"),
    }

    if "jitter" in mode_list:
        desc["jitter_axis_std"] = sigma
        desc["jitter_l2_rms"] = _l2_rms_from_axis_std(sigma)

    if "rigid" in mode_list:
        angle_max_deg = rigid_angle_max_deg_from_sigma(
            sigma=sigma,
            rigid_rot_deg=params.rigid_rot_deg,
            rigid_rot_deg_min=params.rigid_rot_deg_min,
        )
        trans_axis_std = rigid_trans_axis_std_from_sigma(
            sigma=sigma,
            rigid_trans_scale=params.rigid_trans_scale,
            rigid_trans_scale_min=params.rigid_trans_scale_min,
        )
        desc["rigid_angle_max_deg"] = angle_max_deg
        desc["rigid_angle_std_deg"] = float(angle_max_deg / math.sqrt(3.0))
        desc["rigid_trans_axis_std"] = trans_axis_std
        desc["rigid_trans_l2_rms"] = _l2_rms_from_axis_std(trans_axis_std)

    if "outliers" in mode_list:
        out_axis_std = float(params.outlier_scale) * sigma
        desc["outlier_frac"] = float(params.outlier_frac)
        desc["outlier_disp_axis_std"] = out_axis_std
        desc["outlier_disp_l2_rms"] = _l2_rms_from_axis_std(out_axis_std)

    return desc


def build_perturbation_reference(params: PerturbationParams) -> Dict[str, object]:
    rigid_angle_formula = _format_affine_sigma_formula(params.rigid_rot_deg_min, params.rigid_rot_deg)
    rigid_trans_formula = _format_affine_sigma_formula(params.rigid_trans_scale_min, params.rigid_trans_scale)
    return {
        "coord_unit": "normalized_xyz",
        "coord_note": "Mesh verts are centered and normalized per mesh before evaluation.",
        "jitter": {
            "axis_std_formula": "sigma",
            "l2_rms_formula": "sqrt(3) * sigma",
        },
        "rigid": {
            "angle_sampling": "uniform in [-angle_max, angle_max]",
            "angle_max_deg_formula": f"0 if sigma<=0 else {rigid_angle_formula}",
            "angle_std_deg_formula": f"(0 if sigma<=0 else {rigid_angle_formula}) / sqrt(3)",
            "translation_axis_std_formula": f"0 if sigma<=0 else {rigid_trans_formula}",
            "translation_l2_rms_formula": f"sqrt(3) * (0 if sigma<=0 else {rigid_trans_formula})",
        },
        "outliers": {
            "affected_vertex_frac": float(params.outlier_frac),
            "displacement_axis_std_formula": f"{float(params.outlier_scale):.6g} * sigma",
            "displacement_l2_rms_formula": f"sqrt(3) * {float(params.outlier_scale):.6g} * sigma",
        },
    }


def threshold_label(thr: float) -> str:
    s = f"{float(thr):.3f}".rstrip("0").rstrip(".")
    s = s.replace(".", "")
    return f"sigma_r{s}"


def summarize_pack(name: str, eval_pack: Dict[str, object], thresholds: Sequence[float]) -> Dict[str, object]:
    clean = eval_pack["clean"]
    noisy_rows = list(eval_pack["noisy"]) if isinstance(eval_pack["noisy"], list) else []

    clean_sp = float(eval_pack.get("spearman_clean", float("nan")))
    noisy_sps = [float(row.get("spearman", float("nan"))) for row in noisy_rows]
    noisy_ratios = [float(row.get("ratio", float("nan"))) for row in noisy_rows]
    noisy_sigmas = [float(row.get("sigma", float("nan"))) for row in noisy_rows]

    finite_pairs = [
        (sigma, spearman, ratio)
        for sigma, spearman, ratio in zip(noisy_sigmas, noisy_sps, noisy_ratios)
        if math.isfinite(sigma) and math.isfinite(spearman) and math.isfinite(ratio)
    ]

    if finite_pairs:
        sigma_at_min, worst_sp, worst_ratio = min(finite_pairs, key=lambda item: item[1])
        _, _, min_ratio = min(finite_pairs, key=lambda item: item[2])
    else:
        sigma_at_min = float("nan")
        worst_sp = float("nan")
        worst_ratio = float("nan")
        min_ratio = float("nan")

    summary = {
        "scenario": name,
        "metric": str(eval_pack.get("metric", "")),
        "pair_mode": str(eval_pack.get("pair_mode", "")),
        "aggregation_level": str(eval_pack.get("aggregation_level", "")),
        "spearman_clean": clean_sp,
        "pearson_clean": float(eval_pack.get("pearson_clean", float("nan"))),
        "auc_r": float(eval_pack.get("auc_r", float("nan"))),
        "spearman_noisy_max": float(eval_pack.get("spearman_noisy_max", float("nan"))),
        "pearson_noisy_max": float(eval_pack.get("pearson_noisy_max", float("nan"))),
        "ratio_noisy_max": float(eval_pack.get("ratio_noisy_max", float("nan"))),
        "worst_spearman": worst_sp,
        "worst_ratio": worst_ratio,
        "min_ratio": min_ratio,
        "sigma_at_worst_spearman": sigma_at_min,
        "abs_drop_to_worst": (clean_sp - worst_sp) if math.isfinite(clean_sp) and math.isfinite(worst_sp) else float("nan"),
        "n_eval_samples": int(eval_pack.get("n_eval", clean.get("n_eval", 0))),
        "n_pairs": int(eval_pack.get("n_pairs", clean.get("n_pairs", 0))),
        "n_mesh_pairs": int(eval_pack.get("n_mesh_pairs", clean.get("n_mesh_pairs", 0))),
        "n_subject_pairs": int(eval_pack.get("n_subject_pairs", clean.get("n_subject_pairs", 0))),
        "n_subjects": int(eval_pack.get("n_subjects", clean.get("n_subjects", 0))),
        "thresholds": {},
    }

    for thr in thresholds:
        summary["thresholds"][threshold_label(thr)] = first_sigma_below(noisy_rows, thr)
    return summary


def write_grid_csv(
    path: Path,
    scenario_rows: Dict[str, Dict[str, object]],
    max_meshes_per_subject_eval: int,
    pair_ctx: PairEvalContext,
    eval_plan_summary: Dict[str, float],
) -> None:
    total_eval_meshes = int(eval_plan_summary.get("total_meshes", 0))
    mean_eval_meshes = float(eval_plan_summary.get("mean_meshes_per_subject", 0.0))
    min_eval_meshes = int(eval_plan_summary.get("min_meshes_per_subject", 0))
    max_eval_meshes_actual = int(eval_plan_summary.get("max_meshes_per_subject", 0))

    header = [
        "scenario",
        "metric",
        "pair_mode",
        "aggregation_level",
        "noise_modes",
        "eval_mode",
        "coord_unit",
        "sigma",
        "is_clean",
        "max_meshes_per_subject_eval",
        "n_subjects_kept",
        "n_topology_labels",
        "n_eval_samples",
        "n_pairs",
        "n_mesh_pairs",
        "n_subject_pairs",
        "chamfer_vertex_mode",
        "total_eval_meshes",
        "mean_eval_meshes_per_subject",
        "min_eval_meshes_per_subject",
        "max_eval_meshes_per_subject",
        "jitter_axis_std",
        "jitter_l2_rms",
        "rigid_angle_max_deg",
        "rigid_angle_std_deg",
        "rigid_trans_axis_std",
        "rigid_trans_l2_rms",
        "outlier_frac",
        "outlier_disp_axis_std",
        "outlier_disp_l2_rms",
        "spearman",
        "pearson",
        "ratio",
        "intra_mean",
        "gate_mean",
        "n_eval",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for scenario, pack in scenario_rows.items():
            clean = pack["clean"]
            metric = str(pack.get("metric", ""))
            chamfer_vertex_mode = "full_vertices" if metric == "chamfer" else ""
            noise_desc = build_perturbation_descriptor(
                sigma=0.0,
                noise_modes=pack.get("noise_modes", []),
                eval_mode=str(pack.get("eval_mode", "fixed")),
                params=PerturbationParams(
                    outlier_frac=float(pack.get("outlier_frac", float("nan"))),
                    outlier_scale=float(pack.get("outlier_scale", float("nan"))),
                    rigid_rot_deg=float(pack.get("rigid_rot_deg", float("nan"))),
                    rigid_trans_scale=float(pack.get("rigid_trans_scale", float("nan"))),
                    rigid_rot_deg_min=float(pack.get("rigid_rot_deg_min", 0.0)),
                    rigid_trans_scale_min=float(pack.get("rigid_trans_scale_min", 0.0)),
                ),
            )
            writer.writerow(
                [
                    scenario,
                    metric,
                    str(pack.get("pair_mode", "")),
                    str(pack.get("aggregation_level", "")),
                    noise_desc["noise_modes"],
                    noise_desc["eval_mode"],
                    noise_desc["coord_unit"],
                    0.0,
                    1,
                    int(max_meshes_per_subject_eval),
                    pair_ctx.n_subjects,
                    pair_ctx.n_topology_labels,
                    int(clean["n_eval"]),
                    int(clean.get("n_pairs", 0)),
                    int(clean.get("n_mesh_pairs", pair_ctx.mesh_pair_count)),
                    int(clean.get("n_subject_pairs", pair_ctx.subject_pair_count)),
                    chamfer_vertex_mode,
                    total_eval_meshes,
                    f"{mean_eval_meshes:.6f}",
                    min_eval_meshes,
                    max_eval_meshes_actual,
                    f"{float(noise_desc['jitter_axis_std']):.6f}",
                    f"{float(noise_desc['jitter_l2_rms']):.6f}",
                    f"{float(noise_desc['rigid_angle_max_deg']):.6f}",
                    f"{float(noise_desc['rigid_angle_std_deg']):.6f}",
                    f"{float(noise_desc['rigid_trans_axis_std']):.6f}",
                    f"{float(noise_desc['rigid_trans_l2_rms']):.6f}",
                    f"{float(noise_desc['outlier_frac']):.6f}",
                    f"{float(noise_desc['outlier_disp_axis_std']):.6f}",
                    f"{float(noise_desc['outlier_disp_l2_rms']):.6f}",
                    f"{float(clean['spearman']):.6f}",
                    f"{float(clean['pearson']):.6f}",
                    "1.000000",
                    f"{float(clean['intra_mean']):.6e}",
                    f"{float(clean['gate_mean']):.6f}",
                    int(clean["n_eval"]),
                ]
            )
            for row in pack["noisy"]:
                noise_desc = build_perturbation_descriptor(
                    sigma=float(row["sigma"]),
                    noise_modes=pack.get("noise_modes", []),
                    eval_mode=str(pack.get("eval_mode", "fixed")),
                    params=PerturbationParams(
                        outlier_frac=float(pack.get("outlier_frac", float("nan"))),
                        outlier_scale=float(pack.get("outlier_scale", float("nan"))),
                        rigid_rot_deg=float(pack.get("rigid_rot_deg", float("nan"))),
                        rigid_trans_scale=float(pack.get("rigid_trans_scale", float("nan"))),
                        rigid_rot_deg_min=float(pack.get("rigid_rot_deg_min", 0.0)),
                        rigid_trans_scale_min=float(pack.get("rigid_trans_scale_min", 0.0)),
                    ),
                )
                writer.writerow(
                    [
                        scenario,
                        metric,
                        str(pack.get("pair_mode", "")),
                        str(pack.get("aggregation_level", "")),
                        noise_desc["noise_modes"],
                        noise_desc["eval_mode"],
                        noise_desc["coord_unit"],
                        f"{float(row['sigma']):.8e}",
                        0,
                        int(max_meshes_per_subject_eval),
                        pair_ctx.n_subjects,
                        pair_ctx.n_topology_labels,
                        int(row["n_eval"]),
                        int(row.get("n_pairs", 0)),
                        int(row.get("n_mesh_pairs", pair_ctx.mesh_pair_count)),
                        int(row.get("n_subject_pairs", pair_ctx.subject_pair_count)),
                        chamfer_vertex_mode,
                        total_eval_meshes,
                        f"{mean_eval_meshes:.6f}",
                        min_eval_meshes,
                        max_eval_meshes_actual,
                        f"{float(noise_desc['jitter_axis_std']):.6f}",
                        f"{float(noise_desc['jitter_l2_rms']):.6f}",
                        f"{float(noise_desc['rigid_angle_max_deg']):.6f}",
                        f"{float(noise_desc['rigid_angle_std_deg']):.6f}",
                        f"{float(noise_desc['rigid_trans_axis_std']):.6f}",
                        f"{float(noise_desc['rigid_trans_l2_rms']):.6f}",
                        f"{float(noise_desc['outlier_frac']):.6f}",
                        f"{float(noise_desc['outlier_disp_axis_std']):.6f}",
                        f"{float(noise_desc['outlier_disp_l2_rms']):.6f}",
                        f"{float(row['spearman']):.6f}",
                        f"{float(row['pearson']):.6f}",
                        f"{float(row['ratio']):.6f}",
                        f"{float(row['intra_mean']):.6e}",
                        f"{float(row['gate_mean']):.6f}",
                        int(row["n_eval"]),
                    ]
                )


def write_summary_md(path: Path, summaries: Sequence[Dict[str, object]], thresholds: Sequence[float]) -> None:
    thr_cols = [threshold_label(t) for t in thresholds]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Robustness Breakdown Summary\n\n")
        headers = [
            "Scenario",
            "Metric",
            "Pair Mode",
            "Aggregation",
            "Obs",
            "MeshPairs",
            "SubjPairs",
            "Clean Sp",
            "Clean Pe",
            "AUC_R",
            "Noisy Sp @ max",
            "Worst Sp",
            "Worst Ratio",
            "Abs Drop",
        ] + thr_cols
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for summary in summaries:
            row = [
                str(summary["scenario"]),
                str(summary.get("metric", "")),
                str(summary.get("pair_mode", "")),
                str(summary.get("aggregation_level", "")),
                str(int(summary.get("n_pairs", 0))),
                str(int(summary.get("n_mesh_pairs", 0))),
                str(int(summary.get("n_subject_pairs", 0))),
                f"{float(summary['spearman_clean']):.6f}",
                f"{float(summary['pearson_clean']):.6f}",
                f"{float(summary['auc_r']):.6f}",
                f"{float(summary['spearman_noisy_max']):.6f}",
                f"{float(summary['worst_spearman']):.6f}",
                f"{float(summary['worst_ratio']):.6f}",
                f"{float(summary['abs_drop_to_worst']):.6f}",
            ]
            thr_map = summary["thresholds"]
            for col in thr_cols:
                val = float(thr_map.get(col, float("nan")))
                row.append("nan" if not math.isfinite(val) else f"{val:.8e}")
            f.write("| " + " | ".join(row) + " |\n")
