from __future__ import annotations

import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import open3d as o3d
import torch
from pycpd import DeformableRegistration
from tqdm import tqdm


THIS_FILE = Path(__file__).resolve()
INTRINSIC_DIR = THIS_FILE.parent.parent
if str(INTRINSIC_DIR) not in sys.path:
    sys.path.append(str(INTRINSIC_DIR))

from robustness.eval_utils import symmetric_chamfer_same_shape_batch  # noqa: E402


@dataclass(frozen=True)
class NonRigidWarp:
    centers: np.ndarray
    weights: np.ndarray
    beta: float


def build_sample_vertex_indices(
    vertex_count: int,
    n_points: int,
    seed: int,
) -> np.ndarray:
    n_verts = int(vertex_count)
    n_pick = int(max(0, n_points))
    if n_pick <= 0 or n_verts <= n_pick:
        return np.arange(n_verts, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    picked = rng.choice(n_verts, size=n_pick, replace=False)
    return np.asarray(np.sort(picked), dtype=np.int64)


def extract_point_subset(
    V: torch.Tensor,
    indices: np.ndarray,
) -> np.ndarray:
    if V.ndim != 2 or int(V.shape[1]) != 3:
        raise ValueError(f"Expected [N,3] vertices, got {tuple(V.shape)}")
    idx_np = np.asarray(indices, dtype=np.int64)
    if idx_np.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    idx_t = torch.as_tensor(idx_np, device=V.device, dtype=torch.long)
    subset = V.index_select(0, idx_t)
    return np.ascontiguousarray(subset.detach().cpu().numpy().astype(np.float64, copy=False))


def _estimate_rigid_icp_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
    max_correspondence_distance: float,
    max_iteration: int,
) -> np.ndarray:
    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(np.asarray(source_points, dtype=np.float64))
    pcd_tgt = o3d.geometry.PointCloud()
    pcd_tgt.points = o3d.utility.Vector3dVector(np.asarray(target_points, dtype=np.float64))

    reg = o3d.pipelines.registration.registration_icp(
        pcd_src,
        pcd_tgt,
        max_correspondence_distance=float(max_correspondence_distance),
        init=np.eye(4, dtype=np.float64),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=int(max_iteration),
        ),
    )
    return np.asarray(reg.transformation, dtype=np.float32)


def apply_rigid_transform(
    V: torch.Tensor,
    transform: np.ndarray,
) -> torch.Tensor:
    R = torch.as_tensor(transform[:3, :3], device=V.device, dtype=V.dtype)
    t = torch.as_tensor(transform[:3, 3], device=V.device, dtype=V.dtype)
    return V @ R.transpose(0, 1) + t


def _gaussian_kernel_cross(
    X: torch.Tensor,
    centers: torch.Tensor,
    beta: float,
    chunk_size: int,
) -> torch.Tensor:
    if int(X.shape[0]) == 0 or int(centers.shape[0]) == 0:
        return torch.empty((int(X.shape[0]), int(centers.shape[0])), device=X.device, dtype=X.dtype)

    beta_sq = float(beta) * float(beta)
    if beta_sq <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")

    parts: List[torch.Tensor] = []
    step = max(1, int(chunk_size))
    for start in range(0, int(X.shape[0]), step):
        stop = min(start + step, int(X.shape[0]))
        diff = X[start:stop, None, :] - centers[None, :, :]
        d2 = (diff * diff).sum(dim=2)
        parts.append(torch.exp(-d2 / (2.0 * beta_sq)))
    return torch.cat(parts, dim=0)


def apply_nonrigid_cpd_warp(
    V: torch.Tensor,
    warp: NonRigidWarp,
    chunk_size: int = 4096,
) -> torch.Tensor:
    if int(V.shape[0]) == 0 or int(warp.centers.shape[0]) == 0:
        return V
    centers_t = torch.as_tensor(warp.centers, device=V.device, dtype=V.dtype)
    weights_t = torch.as_tensor(warp.weights, device=V.device, dtype=V.dtype)
    kernel = _gaussian_kernel_cross(V, centers=centers_t, beta=float(warp.beta), chunk_size=int(chunk_size))
    return V + kernel @ weights_t


def _estimate_nonrigid_cpd_warp(
    source_points: np.ndarray,
    target_points: np.ndarray,
    alpha: float,
    beta: float,
    max_iterations: int,
    tolerance: float,
    outlier_weight: float,
) -> NonRigidWarp:
    reg = DeformableRegistration(
        X=np.asarray(target_points, dtype=np.float64),
        Y=np.asarray(source_points, dtype=np.float64),
        alpha=float(alpha),
        beta=float(beta),
        max_iterations=int(max_iterations),
        tolerance=float(tolerance),
        w=float(outlier_weight),
    )
    reg.register()
    _, weights = reg.get_registration_parameters()
    return NonRigidWarp(
        centers=np.asarray(reg.Y, dtype=np.float32),
        weights=np.asarray(weights, dtype=np.float32),
        beta=float(reg.beta),
    )


@dataclass(frozen=True)
class PairRegistrationResult:
    rigid_transform: np.ndarray
    nonrigid_warp: NonRigidWarp | None
    rigid_seconds: float
    nonrigid_seconds: float


def _register_single_pair(
    source_icp_points: np.ndarray,
    target_icp_points: np.ndarray,
    source_cpd_points: np.ndarray | None,
    target_cpd_points: np.ndarray | None,
    use_nonrigid_cpd: bool,
    icp_max_correspondence_distance: float,
    icp_max_iteration: int,
    cpd_alpha: float,
    cpd_beta: float,
    cpd_max_iteration: int,
    cpd_tolerance: float,
    cpd_outlier_weight: float,
) -> PairRegistrationResult:
    rigid_start = time.time()
    transform = _estimate_rigid_icp_transform(
        source_points=source_icp_points,
        target_points=target_icp_points,
        max_correspondence_distance=float(icp_max_correspondence_distance),
        max_iteration=int(icp_max_iteration),
    )
    rigid_seconds = float(time.time() - rigid_start)

    if not use_nonrigid_cpd:
        return PairRegistrationResult(
            rigid_transform=transform,
            nonrigid_warp=None,
            rigid_seconds=rigid_seconds,
            nonrigid_seconds=0.0,
        )

    if source_cpd_points is None or target_cpd_points is None:
        raise RuntimeError("Non-rigid CPD requested but CPD control points were not provided")

    aligned_source = source_cpd_points @ transform[:3, :3].T + transform[:3, 3]
    nonrigid_start = time.time()
    warp = _estimate_nonrigid_cpd_warp(
        source_points=aligned_source,
        target_points=target_cpd_points,
        alpha=float(cpd_alpha),
        beta=float(cpd_beta),
        max_iterations=int(cpd_max_iteration),
        tolerance=float(cpd_tolerance),
        outlier_weight=float(cpd_outlier_weight),
    )
    nonrigid_seconds = float(time.time() - nonrigid_start)
    return PairRegistrationResult(
        rigid_transform=transform,
        nonrigid_warp=warp,
        rigid_seconds=rigid_seconds,
        nonrigid_seconds=nonrigid_seconds,
    )


def _resolve_registration_workers(workers: int) -> int:
    if int(workers) > 0:
        return int(workers)
    return max(1, min(8, os.cpu_count() or 1))


def compute_pairwise_registered_chamfer_values(
    *,
    vertex_sets: Sequence[torch.Tensor],
    icp_point_sets: Sequence[np.ndarray],
    cpd_point_sets: Sequence[np.ndarray] | None,
    pair_i: Sequence[int] | np.ndarray,
    pair_j: Sequence[int] | np.ndarray,
    batch_pairs: int,
    registration_workers: int,
    icp_max_correspondence_distance: float,
    icp_max_iteration: int,
    use_nonrigid_cpd: bool,
    cpd_alpha: float,
    cpd_beta: float,
    cpd_max_iteration: int,
    cpd_tolerance: float,
    cpd_outlier_weight: float,
    warp_chunk_size: int = 4096,
    progress_desc: str = "",
    show_progress: bool = False,
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    pair_i_np = np.asarray(pair_i, dtype=np.int64)
    pair_j_np = np.asarray(pair_j, dtype=np.int64)
    n_pairs = int(pair_i_np.size)
    if n_pairs == 0:
        empty = np.empty((0,), dtype=np.float64)
        timings = {
            "rigid_icp_seconds": empty.copy(),
            "nonrigid_cpd_seconds": empty.copy(),
            "warp_seconds": empty.copy(),
            "registration_seconds": empty.copy(),
            "chamfer_seconds": empty.copy(),
            "total_pair_seconds": empty.copy(),
        }
        return empty, timings

    values = np.empty((n_pairs,), dtype=np.float64)
    rigid_seconds = np.empty((n_pairs,), dtype=np.float64)
    nonrigid_seconds = np.empty((n_pairs,), dtype=np.float64)
    warp_seconds = np.empty((n_pairs,), dtype=np.float64)
    chamfer_seconds = np.empty((n_pairs,), dtype=np.float64)
    total_pair_seconds = np.empty((n_pairs,), dtype=np.float64)

    pair_batch = max(1, int(batch_pairs))
    shape_groups: Dict[tuple[int, int], List[int]] = {}
    for pair_idx in range(n_pairs):
        src = vertex_sets[int(pair_i_np[pair_idx])]
        tgt = vertex_sets[int(pair_j_np[pair_idx])]
        shape_groups.setdefault((int(src.shape[0]), int(tgt.shape[0])), []).append(int(pair_idx))

    total_batches = int(sum(math.ceil(len(indices) / pair_batch) for indices in shape_groups.values()))
    progress_label = str(progress_desc).strip() or "registered chamfer pairs"
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
            desc=progress_label,
            dynamic_ncols=True,
            leave=False,
            position=2,
        )
        if use_tqdm
        else None
    )

    executor: ThreadPoolExecutor | None = None
    n_workers = _resolve_registration_workers(registration_workers)
    if n_workers > 1:
        executor = ThreadPoolExecutor(max_workers=n_workers)

    try:
        for shape_key, group_pair_indices in shape_groups.items():
            group_total = len(group_pair_indices)
            for start in range(0, group_total, pair_batch):
                stop = min(start + pair_batch, group_total)
                batch_pair_indices = group_pair_indices[start:stop]
                jobs = []
                for pair_idx in batch_pair_indices:
                    src_idx = int(pair_i_np[pair_idx])
                    tgt_idx = int(pair_j_np[pair_idx])
                    jobs.append(
                        (
                            icp_point_sets[src_idx],
                            icp_point_sets[tgt_idx],
                            None if cpd_point_sets is None else cpd_point_sets[src_idx],
                            None if cpd_point_sets is None else cpd_point_sets[tgt_idx],
                            bool(use_nonrigid_cpd),
                            float(icp_max_correspondence_distance),
                            int(icp_max_iteration),
                            float(cpd_alpha),
                            float(cpd_beta),
                            int(cpd_max_iteration),
                            float(cpd_tolerance),
                            float(cpd_outlier_weight),
                        )
                    )

                if executor is None:
                    registrations = [_register_single_pair(*job) for job in jobs]
                else:
                    registrations = list(executor.map(lambda job: _register_single_pair(*job), jobs))

                X_batch: List[torch.Tensor] = []
                Y_batch: List[torch.Tensor] = []
                warp_start = time.time()
                for local_idx, pair_idx in enumerate(batch_pair_indices):
                    src_idx = int(pair_i_np[pair_idx])
                    tgt_idx = int(pair_j_np[pair_idx])
                    reg = registrations[local_idx]
                    V_src = apply_rigid_transform(vertex_sets[src_idx], reg.rigid_transform)
                    if reg.nonrigid_warp is not None:
                        V_src = apply_nonrigid_cpd_warp(
                            V_src,
                            warp=reg.nonrigid_warp,
                            chunk_size=int(warp_chunk_size),
                        )
                    X_batch.append(V_src)
                    Y_batch.append(vertex_sets[tgt_idx])
                    rigid_seconds[int(pair_idx)] = float(reg.rigid_seconds)
                    nonrigid_seconds[int(pair_idx)] = float(reg.nonrigid_seconds)
                warp_elapsed_per_pair = float(time.time() - warp_start) / float(len(batch_pair_indices))
                for pair_idx in batch_pair_indices:
                    warp_seconds[int(pair_idx)] = warp_elapsed_per_pair

                chamfer_start = time.time()
                batch_vals = symmetric_chamfer_same_shape_batch(
                    X=torch.stack(X_batch, dim=0),
                    Y=torch.stack(Y_batch, dim=0),
                )
                chamfer_elapsed_per_pair = float(time.time() - chamfer_start) / float(len(batch_pair_indices))
                values[np.asarray(batch_pair_indices, dtype=np.int64)] = batch_vals.detach().cpu().numpy()
                for pair_idx in batch_pair_indices:
                    chamfer_seconds[int(pair_idx)] = chamfer_elapsed_per_pair
                    total_pair_seconds[int(pair_idx)] = (
                        float(rigid_seconds[int(pair_idx)])
                        + float(nonrigid_seconds[int(pair_idx)])
                        + float(warp_seconds[int(pair_idx)])
                        + float(chamfer_seconds[int(pair_idx)])
                    )

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
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
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

    timing_arrays = {
        "rigid_icp_seconds": rigid_seconds,
        "nonrigid_cpd_seconds": nonrigid_seconds,
        "warp_seconds": warp_seconds,
        "registration_seconds": rigid_seconds + nonrigid_seconds,
        "chamfer_seconds": chamfer_seconds,
        "total_pair_seconds": total_pair_seconds,
    }
    return values, timing_arrays
