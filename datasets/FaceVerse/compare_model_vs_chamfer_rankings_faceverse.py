#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import multiprocessing as mp
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
PERTURBATED_DIR = (
    REPO_ROOT
    / "face_embedding"
    / "gt_encdec"
    / "remeshing"
    / "intrinsic"
    / "perturbated"
)
if str(PERTURBATED_DIR) not in sys.path:
    sys.path.append(str(PERTURBATED_DIR))

import compare_model_vs_chamfer_rankings as base  # noqa: E402


DEFAULT_DATA_DIR = THIS_FILE.parent / "downsampled_with_ops"
DEFAULT_GT_MESH_DIR = THIS_FILE.parent / "extracted" / "detail"
DEFAULT_GT_DIST_NPZ = (
    THIS_FILE.parent
    / "gt_distance_matrix"
    / "faceverse_detail_pose01_vertex_mean_l2_normalized.npz"
)
FACEVERSE_STEM_RE = re.compile(r"^(?P<subject>\d+)_(?P<pose>\d+)$")

_GT_VERT_LIST: list[np.ndarray] | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare model latent ranking vs Chamfer ranking on FaceVerse using a GT "
            "distance matrix built from the original clean meshes."
        )
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Run directory or explicit checkpoint path for the model to evaluate",
    )
    parser.add_argument(
        "--checkpoint_selector",
        type=str,
        default="best_by_auc",
        choices=("best_by_auc", "best_by_clean", "latest"),
        help="Checkpoint selection when --model_path points to a run directory",
    )
    parser.add_argument("--config_json", type=str, default="", help="Optional explicit config.json path")
    parser.add_argument("--out_dir", type=str, default="", help="Optional output directory")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing FaceVerse operator .npz files used for model/chamfer inference",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_01.npz",
        help="Basename glob used to select inference samples inside --data_dir",
    )
    parser.add_argument(
        "--gt_mesh_dir",
        type=str,
        default=str(DEFAULT_GT_MESH_DIR),
        help="Directory containing original clean FaceVerse meshes used to build GT distances",
    )
    parser.add_argument(
        "--gt_pattern",
        type=str,
        default="*_01.ply",
        help="Basename glob used to select clean GT meshes inside --gt_mesh_dir",
    )
    parser.add_argument(
        "--dist_npz",
        type=str,
        default=str(DEFAULT_GT_DIST_NPZ),
        help="Cached GT distance matrix path; computed automatically if missing",
    )
    parser.add_argument(
        "--recompute_gt_dist",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force rebuilding the GT distance matrix from the original meshes",
    )
    parser.add_argument(
        "--pose_ids",
        nargs="*",
        default=None,
        help="Optional explicit pose ids such as 01 02 03. Applied to both data and GT filters.",
    )
    parser.add_argument(
        "--subject_ids",
        nargs="*",
        default=None,
        help="Optional explicit subject ids such as 001 017 110. Applied to both data and GT filters.",
    )
    parser.add_argument("--subject_split", type=str, default="all", choices=("eval", "train", "all"))
    parser.add_argument("--eval_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=-1, help="Negative => use checkpoint/config seed")
    parser.add_argument("--max_subjects", type=int, default=0, help="0 = all filtered subjects")
    parser.add_argument(
        "--max_meshes_per_subject_eval",
        type=int,
        default=1,
        help="Max inference meshes per subject. 0 = all filtered meshes per subject.",
    )
    parser.add_argument(
        "--preload_eval_samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preload selected inference samples before evaluation",
    )
    parser.add_argument("--preload_workers", type=int, default=4, help="Workers for sample preloading")

    parser.add_argument(
        "--pair_mode",
        type=str,
        default="within_topology",
        choices=base.ALLOWED_PAIR_MODES,
        help="Topology label corresponds to pose id, so within_topology keeps comparisons within the same pose",
    )
    parser.add_argument(
        "--aggregation_level",
        type=str,
        default="subject_pair_mean",
        choices=base.ALLOWED_AGGREGATION_LEVELS,
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="clean,jitter,translation,rotation,mixed",
        help="Comma-separated scenarios from: clean,jitter,translation,rotation,mixed",
    )

    parser.add_argument(
        "--base_sigma",
        type=float,
        default=-1.0,
        help="Default sigma for perturbation scenarios. Negative => use config sigma_max, else 0.05 fallback",
    )
    parser.add_argument("--jitter_sigma", type=float, default=-1.0, help="Override jitter sigma")
    parser.add_argument("--translation_sigma", type=float, default=-1.0, help="Override translation sigma")
    parser.add_argument("--rotation_sigma", type=float, default=-1.0, help="Override rotation sigma")
    parser.add_argument("--mixed_jitter_sigma", type=float, default=-1.0, help="Override mixed jitter sigma")
    parser.add_argument("--mixed_translation_sigma", type=float, default=-1.0, help="Override mixed translation sigma")
    parser.add_argument("--mixed_rotation_sigma", type=float, default=-1.0, help="Override mixed rotation sigma")

    parser.add_argument("--rigid_rot_deg", type=float, default=-1.0, help="Override rigid rotation slope in degrees")
    parser.add_argument("--rigid_trans_scale", type=float, default=-1.0, help="Override rigid translation slope")
    parser.add_argument("--rigid_rot_deg_min", type=float, default=-1.0, help="Override rigid minimum angle")
    parser.add_argument("--rigid_trans_scale_min", type=float, default=-1.0, help="Override rigid minimum translation")

    parser.add_argument(
        "--chamfer_use_icp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align source and target with rigid ICP before computing Chamfer",
    )
    parser.add_argument(
        "--icp_points",
        type=int,
        default=2048,
        help="Number of vertices sampled from each perturbed mesh to estimate the ICP rigid transform",
    )
    parser.add_argument(
        "--icp_max_correspondence_distance",
        type=float,
        default=0.05,
        help="Open3D ICP max_correspondence_distance",
    )
    parser.add_argument("--icp_max_iteration", type=int, default=20, help="Open3D ICP max iterations")
    parser.add_argument(
        "--icp_workers",
        type=int,
        default=4,
        help="CPU worker threads for pairwise ICP inside each Chamfer batch",
    )
    parser.add_argument("--chamfer_batch_pairs", type=int, default=64, help="Pair batch size for Chamfer evaluation")
    parser.add_argument(
        "--chamfer_pair_progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show pair-batch progress during Chamfer evaluation",
    )
    parser.add_argument(
        "--chamfer_cache_verts",
        type=str,
        default="auto",
        choices=("off", "auto", "force"),
        help="Cache preloaded vertices on eval device when beneficial",
    )
    parser.add_argument(
        "--chamfer_cache_verts_max_mb",
        type=float,
        default=256.0,
        help="Device-cache limit in auto mode",
    )

    parser.add_argument(
        "--gt_workers",
        type=int,
        default=0,
        help="Processes used to build the GT matrix. 0 = auto.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def _normalize_subject_id(text: str) -> str:
    text = str(text).strip()
    if text.isdigit():
        return f"{int(text):03d}"
    return text


def _normalize_pose_id(text: str) -> str:
    text = str(text).strip()
    if text.isdigit():
        return f"{int(text):02d}"
    return text


def _parse_faceverse_name(name: str) -> tuple[str, str] | None:
    stem = Path(str(name)).stem
    match = FACEVERSE_STEM_RE.fullmatch(stem)
    if match is None:
        return None
    return _normalize_subject_id(match.group("subject")), _normalize_pose_id(match.group("pose"))


def _normalize_vertices_like_dataset(V: np.ndarray) -> np.ndarray:
    arr = np.asarray(V, dtype=np.float64)
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = float(np.max(np.abs(arr)))
    if scale > 1e-6:
        arr = arr / scale
    else:
        arr = arr * 0.0
    return arr.astype(np.float32, copy=False)


def _filter_subject_entries(
    entries: Iterable[tuple[str, Path]],
    pattern: str,
    subject_ids: Sequence[str] | None,
    pose_ids: Sequence[str] | None,
) -> tuple[Dict[str, List[Path]], Dict[str, object]]:
    allowed_subjects = {_normalize_subject_id(sid) for sid in subject_ids} if subject_ids else None
    allowed_poses = {_normalize_pose_id(pid) for pid in pose_ids} if pose_ids else None

    subject_map: Dict[str, List[Path]] = {}
    pose_counter: Counter[str] = Counter()
    skipped_pattern = 0
    skipped_parse = 0
    skipped_subject = 0
    skipped_pose = 0

    for basename, path in entries:
        if not fnmatch.fnmatch(basename, pattern):
            skipped_pattern += 1
            continue
        parsed = _parse_faceverse_name(basename)
        if parsed is None:
            skipped_parse += 1
            continue
        subject_id, pose_id = parsed
        if allowed_subjects is not None and subject_id not in allowed_subjects:
            skipped_subject += 1
            continue
        if allowed_poses is not None and pose_id not in allowed_poses:
            skipped_pose += 1
            continue
        subject_map.setdefault(subject_id, []).append(path)
        pose_counter[pose_id] += 1

    for subject_id in list(subject_map.keys()):
        subject_map[subject_id] = sorted(subject_map[subject_id], key=lambda item: item.name)

    summary = {
        "pattern": str(pattern),
        "requested_subject_ids": sorted(list(allowed_subjects)) if allowed_subjects is not None else [],
        "requested_pose_ids": sorted(list(allowed_poses)) if allowed_poses is not None else [],
        "n_subjects": int(len(subject_map)),
        "n_samples": int(sum(len(v) for v in subject_map.values())),
        "pose_counts": dict(sorted(pose_counter.items())),
        "skipped_pattern": int(skipped_pattern),
        "skipped_parse": int(skipped_parse),
        "skipped_subject": int(skipped_subject),
        "skipped_pose": int(skipped_pose),
        "sample_preview": [str(path.name) for paths in subject_map.values() for path in paths][:12],
    }
    return subject_map, summary


def _resolve_gt_subject_paths(
    *,
    gt_mesh_dir: str,
    gt_pattern: str,
    subject_ids: Sequence[str] | None,
    pose_ids: Sequence[str] | None,
    dist_npz: str,
    recompute_gt_dist: bool,
) -> tuple[Dict[str, List[Path]], Dict[str, object]]:
    gt_root = Path(gt_mesh_dir).expanduser().resolve()
    gt_paths = sorted(gt_root.glob("*.ply")) if gt_root.exists() else []
    gt_subject_paths, gt_filter_summary = _filter_subject_entries(
        entries=[(path.name, path) for path in gt_paths],
        pattern=str(gt_pattern),
        subject_ids=subject_ids,
        pose_ids=pose_ids,
    )
    if gt_subject_paths:
        gt_filter_summary = dict(gt_filter_summary)
        gt_filter_summary["source"] = "gt_mesh_dir"
        gt_filter_summary["gt_mesh_dir_exists"] = bool(gt_root.exists())
        gt_filter_summary["gt_mesh_glob_count"] = int(len(gt_paths))
        return gt_subject_paths, gt_filter_summary

    dist_path = Path(dist_npz).expanduser().resolve()
    if dist_path.exists() and not recompute_gt_dist:
        _, cached_name_to_idx = _load_faceverse_gt_distance_matrix(dist_path)
        cached_subjects = sorted(str(name) for name in cached_name_to_idx.keys())

        allowed_subjects = {_normalize_subject_id(sid) for sid in subject_ids} if subject_ids else None
        allowed_poses = {_normalize_pose_id(pid) for pid in pose_ids} if pose_ids else None
        if allowed_poses is not None and "01" not in allowed_poses:
            cached_subjects = []
        if allowed_subjects is not None:
            cached_subjects = [sid for sid in cached_subjects if sid in allowed_subjects]

        gt_subject_paths = {sid: [] for sid in cached_subjects}
        gt_filter_summary = {
            "pattern": str(gt_pattern),
            "requested_subject_ids": sorted(list(allowed_subjects)) if allowed_subjects is not None else [],
            "requested_pose_ids": sorted(list(allowed_poses)) if allowed_poses is not None else [],
            "n_subjects": int(len(cached_subjects)),
            "n_samples": int(len(cached_subjects)),
            "pose_counts": {"01": int(len(cached_subjects))} if cached_subjects else {},
            "skipped_pattern": int(0),
            "skipped_parse": int(0),
            "skipped_subject": int(0),
            "skipped_pose": int(0),
            "sample_preview": [f"{sid}_01.ply" for sid in cached_subjects[:12]],
            "source": "distance_cache",
            "cache_path": str(dist_path),
            "gt_mesh_dir_exists": bool(gt_root.exists()),
            "gt_mesh_glob_count": int(len(gt_paths)),
        }
        return gt_subject_paths, gt_filter_summary

    gt_filter_summary = dict(gt_filter_summary)
    gt_filter_summary["source"] = "missing_gt_mesh_dir"
    gt_filter_summary["cache_path"] = str(dist_path)
    gt_filter_summary["gt_mesh_dir_exists"] = bool(gt_root.exists())
    gt_filter_summary["gt_mesh_glob_count"] = int(len(gt_paths))
    return gt_subject_paths, gt_filter_summary


def _build_faceverse_inference_subject_map(
    files: Sequence[str],
    pattern: str,
    subject_ids: Sequence[str] | None,
    pose_ids: Sequence[str] | None,
) -> tuple[Dict[str, List[int]], Dict[str, object]]:
    allowed_subjects = {_normalize_subject_id(sid) for sid in subject_ids} if subject_ids else None
    allowed_poses = {_normalize_pose_id(pid) for pid in pose_ids} if pose_ids else None

    subject_map: Dict[str, List[int]] = {}
    pose_counter: Counter[str] = Counter()
    skipped_pattern = 0
    skipped_parse = 0
    skipped_subject = 0
    skipped_pose = 0

    for idx, fname in enumerate(files):
        basename = Path(str(fname)).name
        if not fnmatch.fnmatch(basename, pattern):
            skipped_pattern += 1
            continue
        parsed = _parse_faceverse_name(basename)
        if parsed is None:
            skipped_parse += 1
            continue
        subject_id, pose_id = parsed
        if allowed_subjects is not None and subject_id not in allowed_subjects:
            skipped_subject += 1
            continue
        if allowed_poses is not None and pose_id not in allowed_poses:
            skipped_pose += 1
            continue
        subject_map.setdefault(subject_id, []).append(int(idx))
        pose_counter[pose_id] += 1

    for subject_id, idxs in subject_map.items():
        idxs.sort(key=lambda item: files[int(item)])

    summary = {
        "pattern": str(pattern),
        "requested_subject_ids": sorted(list(allowed_subjects)) if allowed_subjects is not None else [],
        "requested_pose_ids": sorted(list(allowed_poses)) if allowed_poses is not None else [],
        "n_subjects": int(len(subject_map)),
        "n_samples": int(sum(len(v) for v in subject_map.values())),
        "pose_counts": dict(sorted(pose_counter.items())),
        "skipped_pattern": int(skipped_pattern),
        "skipped_parse": int(skipped_parse),
        "skipped_subject": int(skipped_subject),
        "skipped_pose": int(skipped_pose),
        "sample_preview": [files[idx] for idxs in subject_map.values() for idx in idxs][:12],
    }
    return subject_map, summary


def _resolve_gt_workers(workers: int) -> int:
    if workers > 0:
        return workers
    return max(1, min(26, (os.cpu_count() or 1) - 2))


def _load_gt_vertices(path: Path) -> np.ndarray:
    import igl  # noqa: WPS433

    V, _ = igl.read_triangle_mesh(str(path))
    if V.ndim != 2 or V.shape[1] != 3:
        raise RuntimeError(f"Invalid vertex array in {path}: shape={V.shape}")
    return _normalize_vertices_like_dataset(V)


def _init_gt_worker(v_list: list[np.ndarray]) -> None:
    global _GT_VERT_LIST
    _GT_VERT_LIST = v_list


def _gt_pair_distance(pair: tuple[int, int]) -> tuple[int, int, float]:
    if _GT_VERT_LIST is None:
        raise RuntimeError("GT worker not initialized")
    i, j = pair
    Vi = _GT_VERT_LIST[i]
    Vj = _GT_VERT_LIST[j]
    diff = Vi - Vj
    d = float(np.sqrt((diff * diff).sum(axis=1)).mean())
    return int(i), int(j), d


def _pair_generator(n_items: int) -> Iterable[tuple[int, int]]:
    for i in range(n_items):
        for j in range(i + 1, n_items):
            yield i, j


def _compute_faceverse_gt_distance_matrix(
    gt_subject_paths: Dict[str, List[Path]],
    out_path: Path,
    workers: int,
) -> tuple[np.ndarray, Dict[str, int], Dict[str, object]]:
    subject_ids = sorted(gt_subject_paths.keys())
    if not subject_ids:
        raise RuntimeError("No GT FaceVerse meshes available to build the distance matrix")

    load_paths = []
    for subject_id in subject_ids:
        first_path = gt_subject_paths[subject_id][0]
        load_paths.append((subject_id, first_path))

    print(f"Loading GT meshes from original FaceVerse: {len(load_paths)} subjects")
    verts_list: list[np.ndarray] = []
    for subject_id, path in tqdm(load_paths, desc="GT meshes", dynamic_ncols=True):
        V = _load_gt_vertices(path)
        verts_list.append(V)

    vertex_counts = {int(V.shape[0]) for V in verts_list}
    if len(vertex_counts) != 1:
        raise RuntimeError(f"GT meshes do not share a common vertex count: {sorted(vertex_counts)}")

    n_items = len(subject_ids)
    D = np.zeros((n_items, n_items), dtype=np.float32)
    total_pairs = n_items * (n_items - 1) // 2
    resolved_workers = _resolve_gt_workers(workers)

    if total_pairs > 0:
        with mp.get_context("fork").Pool(
            processes=resolved_workers,
            initializer=_init_gt_worker,
            initargs=(verts_list,),
        ) as pool:
            for i, j, d in tqdm(
                pool.imap_unordered(_gt_pair_distance, _pair_generator(n_items), chunksize=32),
                total=total_pairs,
                desc="GT pair distances",
                dynamic_ncols=True,
            ):
                D[i, j] = np.float32(d)
                D[j, i] = np.float32(d)

    mask = D > 0
    if mask.any():
        D = D / float(D[mask].max())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        D_orig=D.astype(np.float32),
        names=np.asarray(subject_ids, dtype=str),
    )
    summary = {
        "path": str(out_path),
        "metric": "vertex_mean_l2",
        "source_mesh_dir": str(DEFAULT_GT_MESH_DIR),
        "n_subjects": int(n_items),
        "n_pairs": int(total_pairs),
        "vertex_count": int(next(iter(vertex_counts))),
        "workers": int(resolved_workers),
    }
    return D, {sid: idx for idx, sid in enumerate(subject_ids)}, summary


def _load_faceverse_gt_distance_matrix(path: Path) -> tuple[np.ndarray, Dict[str, int]]:
    pack = np.load(path, allow_pickle=False)
    if "D_orig" not in pack or "names" not in pack:
        raise KeyError(f"{path} must contain D_orig and names")
    D = np.asarray(pack["D_orig"], dtype=np.float64)
    names = [str(x) for x in np.asarray(pack["names"]).tolist()]
    pack.close()
    mask = D > 0
    if mask.any():
        D = D / float(D[mask].max())
    return D, {str(name): idx for idx, name in enumerate(names)}


def _resolve_faceverse_gt_distance_matrix(
    gt_subject_paths: Dict[str, List[Path]],
    out_path: Path,
    recompute: bool,
    workers: int,
) -> tuple[np.ndarray, Dict[str, int], Dict[str, object]]:
    selected_subjects = sorted(gt_subject_paths.keys())
    if not selected_subjects:
        raise RuntimeError("No GT subject paths available")

    if out_path.exists() and not recompute:
        D, name_to_idx = _load_faceverse_gt_distance_matrix(out_path)
        if all(subject_id in name_to_idx for subject_id in selected_subjects):
            return D, name_to_idx, {
                "path": str(out_path),
                "metric": "vertex_mean_l2",
                "status": "loaded_cache",
                "n_subjects_cached": int(len(name_to_idx)),
            }

    D, name_to_idx, summary = _compute_faceverse_gt_distance_matrix(
        gt_subject_paths=gt_subject_paths,
        out_path=out_path,
        workers=workers,
    )
    summary = dict(summary)
    summary["status"] = "recomputed"
    return D, name_to_idx, summary


def _sample_vertices_for_icp(V: torch.Tensor, n_points: int, seed: int) -> np.ndarray:
    arr = V.detach().cpu().numpy().astype(np.float64, copy=False)
    n_verts = int(arr.shape[0])
    if n_points <= 0 or n_verts <= n_points:
        return np.ascontiguousarray(arr)
    rng = np.random.default_rng(seed)
    picked = rng.choice(n_verts, size=int(n_points), replace=False)
    return np.ascontiguousarray(arr[np.asarray(picked, dtype=np.int64)])


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


def _apply_rigid_transform(V: torch.Tensor, transform: np.ndarray) -> torch.Tensor:
    R = torch.as_tensor(transform[:3, :3], device=V.device, dtype=V.dtype)
    t = torch.as_tensor(transform[:3, 3], device=V.device, dtype=V.dtype)
    return V @ R.transpose(0, 1) + t


def _symmetric_chamfer_same_shape_batch(
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
            parts.append(_symmetric_chamfer_same_shape_batch(X[start:stop], Y[start:stop], max_cdist_mb=max_cdist_mb))
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


def _resolve_icp_workers(workers: int) -> int:
    if workers > 0:
        return workers
    return max(1, min(8, os.cpu_count() or 1))


def _precompute_pairwise_icp_transforms(
    icp_point_sets: Sequence[np.ndarray],
    pair_i: Sequence[int] | np.ndarray,
    pair_j: Sequence[int] | np.ndarray,
    batch_pairs: int,
    icp_workers: int,
    icp_max_correspondence_distance: float,
    icp_max_iteration: int,
    progress_desc: str,
    show_progress: bool,
) -> np.ndarray:
    pair_i_np = np.asarray(pair_i, dtype=np.int64)
    pair_j_np = np.asarray(pair_j, dtype=np.int64)
    n_pairs = int(pair_i_np.size)
    transforms = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], n_pairs, axis=0)
    if n_pairs == 0:
        return transforms

    pair_batch = max(1, int(batch_pairs))
    total_batches = int(math.ceil(n_pairs / pair_batch))
    pbar = (
        tqdm(total=total_batches, desc=progress_desc, dynamic_ncols=True, leave=False)
        if show_progress and total_batches > 1
        else None
    )

    executor: ThreadPoolExecutor | None = None
    if _resolve_icp_workers(icp_workers) > 1:
        executor = ThreadPoolExecutor(max_workers=_resolve_icp_workers(icp_workers))

    try:
        for start in range(0, n_pairs, pair_batch):
            stop = min(start + pair_batch, n_pairs)
            jobs = [
                (
                    icp_point_sets[int(pair_i_np[pair_idx])],
                    icp_point_sets[int(pair_j_np[pair_idx])],
                    float(icp_max_correspondence_distance),
                    int(icp_max_iteration),
                )
                for pair_idx in range(start, stop)
            ]
            if executor is None:
                chunk_transforms = [_estimate_rigid_icp_transform(*job) for job in jobs]
            else:
                chunk_transforms = list(executor.map(lambda job: _estimate_rigid_icp_transform(*job), jobs))
            transforms[start:stop] = np.stack(chunk_transforms, axis=0).astype(np.float32, copy=False)
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(done=f"{stop}/{n_pairs}")
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        if pbar is not None:
            pbar.close()

    return transforms


def _compute_pairwise_icp_chamfer_values(
    vertex_sets: Sequence[torch.Tensor],
    icp_point_sets: Sequence[np.ndarray] | None,
    pair_i: Sequence[int] | np.ndarray,
    pair_j: Sequence[int] | np.ndarray,
    batch_pairs: int,
    use_icp: bool,
    pair_transforms: np.ndarray | None,
    icp_workers: int,
    icp_max_correspondence_distance: float,
    icp_max_iteration: int,
    progress_desc: str,
    show_progress: bool,
) -> np.ndarray:
    pair_i_np = np.asarray(pair_i, dtype=np.int64)
    pair_j_np = np.asarray(pair_j, dtype=np.int64)
    n_pairs = int(pair_i_np.size)
    if n_pairs == 0:
        return np.empty((0,), dtype=np.float64)

    pair_batch = max(1, int(batch_pairs))
    values = np.empty((n_pairs,), dtype=np.float64)
    shape_groups: Dict[tuple[int, int], List[int]] = {}
    for pair_idx in range(n_pairs):
        Xi = vertex_sets[int(pair_i_np[pair_idx])]
        Yj = vertex_sets[int(pair_j_np[pair_idx])]
        shape_groups.setdefault((int(Xi.shape[0]), int(Yj.shape[0])), []).append(int(pair_idx))

    total_batches = int(sum(math.ceil(len(indices) / pair_batch) for indices in shape_groups.values()))
    pbar = (
        tqdm(total=total_batches, desc=progress_desc, dynamic_ncols=True, leave=False)
        if show_progress and total_batches > 1
        else None
    )

    executor: ThreadPoolExecutor | None = None
    if use_icp and _resolve_icp_workers(icp_workers) > 1:
        executor = ThreadPoolExecutor(max_workers=_resolve_icp_workers(icp_workers))

    try:
        for shape_key, group_pair_indices in shape_groups.items():
            for start in range(0, len(group_pair_indices), pair_batch):
                stop = min(start + pair_batch, len(group_pair_indices))
                chunk_pair_indices = group_pair_indices[start:stop]

                transforms: list[np.ndarray]
                if pair_transforms is not None:
                    transforms = [
                        np.asarray(pair_transforms[int(pair_idx)], dtype=np.float32)
                        for pair_idx in chunk_pair_indices
                    ]
                elif use_icp:
                    if icp_point_sets is None:
                        raise RuntimeError("use_icp=True requires icp_point_sets when pair_transforms are not precomputed")
                    jobs = [
                        (
                            icp_point_sets[int(pair_i_np[pair_idx])],
                            icp_point_sets[int(pair_j_np[pair_idx])],
                            float(icp_max_correspondence_distance),
                            int(icp_max_iteration),
                        )
                        for pair_idx in chunk_pair_indices
                    ]

                    if executor is None:
                        transforms = [
                            _estimate_rigid_icp_transform(*job)
                            for job in jobs
                        ]
                    else:
                        transforms = list(executor.map(lambda job: _estimate_rigid_icp_transform(*job), jobs))
                else:
                    transforms = [np.eye(4, dtype=np.float32) for _ in chunk_pair_indices]

                X_batch = []
                Y_batch = []
                for local_idx, pair_idx in enumerate(chunk_pair_indices):
                    src_idx = int(pair_i_np[pair_idx])
                    tgt_idx = int(pair_j_np[pair_idx])
                    X_batch.append(_apply_rigid_transform(vertex_sets[src_idx], transforms[local_idx]))
                    Y_batch.append(vertex_sets[tgt_idx])

                batch_vals = _symmetric_chamfer_same_shape_batch(
                    X=torch.stack(X_batch, dim=0),
                    Y=torch.stack(Y_batch, dim=0),
                )
                values[np.asarray(chunk_pair_indices, dtype=np.int64)] = batch_vals.detach().cpu().numpy()

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(shape=f"{shape_key[0]}x{shape_key[1]}", done=f"{stop}/{len(group_pair_indices)}")
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        if pbar is not None:
            pbar.close()

    return values


@torch.no_grad()
def _evaluate_scenario(
    model: torch.nn.Module,
    dataset: base.GTReadyDataset,
    pair_ctx,
    sample_cache: Dict[int, Dict[str, torch.Tensor]] | None,
    device: torch.device,
    params: base.PerturbationParams,
    scenario: base.ScenarioSpec,
    scenario_index: int,
    base_seed: int,
    chamfer_batch_pairs: int,
    chamfer_use_icp: bool,
    pairwise_icp_transforms: np.ndarray | None,
    icp_workers: int,
    icp_max_correspondence_distance: float,
    icp_max_iteration: int,
    show_chamfer_pair_progress: bool,
) -> Dict[str, object]:
    latent_vectors: List[torch.Tensor] = []
    full_vertex_sets: List[torch.Tensor] = []

    for record in pair_ctx.sample_records:
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
        full_vertex_sets.append(V_in.contiguous())

    Z = torch.stack(latent_vectors, dim=0)
    latent_values = torch.linalg.vector_norm(
        Z.index_select(0, pair_ctx.pair_i) - Z.index_select(0, pair_ctx.pair_j),
        dim=1,
    )
    latent_values_np = base.aggregate_pair_observations(
        latent_values.detach().cpu().numpy().astype(np.float64, copy=False),
        pair_ctx=pair_ctx,
    )

    chamfer_mesh_values = _compute_pairwise_icp_chamfer_values(
        vertex_sets=full_vertex_sets,
        icp_point_sets=None,
        pair_i=pair_ctx.pair_i_cpu,
        pair_j=pair_ctx.pair_j_cpu,
        batch_pairs=int(chamfer_batch_pairs),
        use_icp=bool(chamfer_use_icp),
        pair_transforms=pairwise_icp_transforms,
        icp_workers=int(icp_workers),
        icp_max_correspondence_distance=float(icp_max_correspondence_distance),
        icp_max_iteration=int(icp_max_iteration),
        progress_desc=f"{scenario.name} aligned-chamfer pairs",
        show_progress=bool(show_chamfer_pair_progress),
    )
    chamfer_values_np = base.aggregate_pair_observations(chamfer_mesh_values, pair_ctx=pair_ctx)

    gt_vals = np.asarray(pair_ctx.gt_vals, dtype=np.float64)
    return {
        "scenario": scenario.name,
        "latent_spearman": float(base.spearman_corr(gt_vals, latent_values_np)),
        "latent_pearson": float(base.pearson_corr(gt_vals, latent_values_np)),
        "chamfer_spearman": float(base.spearman_corr(gt_vals, chamfer_values_np)),
        "chamfer_pearson": float(base.pearson_corr(gt_vals, chamfer_values_np)),
        "n_pairs": int(pair_ctx.pair_count),
        "n_mesh_pairs": int(pair_ctx.mesh_pair_count),
        "n_subject_pairs": int(pair_ctx.subject_pair_count),
        "n_subjects": int(pair_ctx.n_subjects),
        "n_samples": int(pair_ctx.n_samples),
    }


def _write_summary_csv(path: Path, rows: Sequence[dict], params: base.PerturbationParams, scenarios: Sequence[base.ScenarioSpec]) -> None:
    scenario_lookup = {spec.name: spec for spec in scenarios}
    header = [
        "scenario",
        "reference_metric",
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
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            desc = base._describe_scenario(scenario_lookup[row["scenario"]], params=params)
            writer.writerow(
                {
                    "scenario": row["scenario"],
                    "reference_metric": "gt_vertex_mean_l2",
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


def _write_summary_md(path: Path, rows: Sequence[dict], params: base.PerturbationParams, scenarios: Sequence[base.ScenarioSpec]) -> None:
    scenario_lookup = {spec.name: spec for spec in scenarios}
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# FaceVerse Model vs Chamfer Ranking Summary\n\n")
        handle.write("Reference metric: GT distance matrix from original clean FaceVerse meshes.\n\n")
        handle.write(
            "| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | "
            "Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            desc = base._describe_scenario(scenario_lookup[row["scenario"]], params=params)
            handle.write(
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

    run_dir, checkpoint_path = base._resolve_run_dir_and_checkpoint(
        cli_args.model_path,
        selector=str(cli_args.checkpoint_selector),
    )
    base_args = base._resolve_runtime_args(cli_args, base.merge_run_args(checkpoint_path, explicit_config_json=cli_args.config_json))
    params = base.PerturbationParams.from_namespace(base_args)

    base_sigma = float(cli_args.base_sigma)
    if base_sigma < 0.0:
        base_sigma = float(getattr(base_args, "sigma_max", 0.05))
    scenarios = base._parse_scenarios(cli_args.scenarios, base_sigma=base_sigma, cli_args=cli_args)

    base.seed_everything(int(base_args.seed))
    device = base._resolve_device(cli_args.device)

    dataset = base.GTReadyDataset(base_args.data_dir)
    inference_subject_map, inference_filter_summary = _build_faceverse_inference_subject_map(
        files=dataset.files,
        pattern=str(cli_args.pattern),
        subject_ids=cli_args.subject_ids,
        pose_ids=cli_args.pose_ids,
    )

    gt_subject_paths, gt_filter_summary = _resolve_gt_subject_paths(
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

    _, _, target_subjects = base._select_subject_subset(
        subjects=overlapping_subjects,
        subject_split=str(cli_args.subject_split),
        eval_fraction=float(cli_args.eval_fraction),
        seed=int(base_args.seed),
        max_subjects=int(cli_args.max_subjects),
    )
    if not target_subjects:
        raise RuntimeError("No FaceVerse subjects remained after subject selection")

    gt_matrix, gt_name_to_idx, gt_dist_summary = _resolve_faceverse_gt_distance_matrix(
        gt_subject_paths=gt_subject_paths,
        out_path=Path(cli_args.dist_npz).expanduser().resolve(),
        recompute=bool(cli_args.recompute_gt_dist),
        workers=int(cli_args.gt_workers),
    )

    eval_plan = base.build_eval_plan(
        subj_map=inference_subject_map,
        eval_subjects=target_subjects,
        max_meshes_per_subject_eval=int(base_args.max_meshes_per_subject_eval),
        seed=int(base_args.seed),
    )
    sample_cache = (
        base.preload_eval_samples(dataset=dataset, eval_plan=eval_plan, workers=int(cli_args.preload_workers))
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
                _sample_vertices_for_icp(
                    sample["verts"],
                    n_points=int(cli_args.icp_points),
                    seed=int(base_args.seed) * 1_000_003 + int(record.dataset_idx),
                )
            )
        pairwise_icp_transforms = _precompute_pairwise_icp_transforms(
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

    model = base.build_model(args=base_args, device=device)
    checkpoint_bundle = base.load_checkpoint_bundle(checkpoint_path)
    model.load_state_dict(checkpoint_bundle["state_dict"], strict=True)
    model.eval()

    rows: List[dict] = []
    for scenario_index, scenario in enumerate(scenarios):
        result = _evaluate_scenario(
            model=model,
            dataset=dataset,
            pair_ctx=pair_ctx,
            sample_cache=sample_cache,
            device=device,
            params=params,
            scenario=scenario,
            scenario_index=scenario_index,
            base_seed=int(base_args.seed),
            chamfer_batch_pairs=int(cli_args.chamfer_batch_pairs),
            chamfer_use_icp=bool(cli_args.chamfer_use_icp),
            pairwise_icp_transforms=pairwise_icp_transforms,
            icp_workers=int(cli_args.icp_workers),
            icp_max_correspondence_distance=float(cli_args.icp_max_correspondence_distance),
            icp_max_iteration=int(cli_args.icp_max_iteration),
            show_chamfer_pair_progress=bool(cli_args.chamfer_pair_progress),
        )
        result["delta_spearman"] = float(result["latent_spearman"] - result["chamfer_spearman"])
        result["delta_pearson"] = float(result["latent_pearson"] - result["chamfer_pearson"])
        result["model_beats_chamfer"] = bool(result["latent_spearman"] > result["chamfer_spearman"])
        rows.append(result)

    out_dir = (
        Path(cli_args.out_dir).expanduser().resolve()
        if cli_args.out_dir
        else run_dir
        / "faceverse_ranking_vs_chamfer"
        / base.slugify_token(
            f"{checkpoint_path.stem}_split-{cli_args.subject_split}_pairs-{cli_args.pair_mode}_"
            f"agglvl-{cli_args.aggregation_level}_subjects-{len(target_subjects)}_"
            f"meshes-{int(base_args.max_meshes_per_subject_eval)}_scenarios-{'-'.join(spec.name for spec in scenarios)}"
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "checkpoint": str(checkpoint_path),
        "run_dir": str(run_dir),
        "checkpoint_selector": str(cli_args.checkpoint_selector),
        "device": str(device),
        "data_dir": str(base_args.data_dir),
        "gt_mesh_dir": str(Path(cli_args.gt_mesh_dir).expanduser().resolve()),
        "dist_npz": str(Path(cli_args.dist_npz).expanduser().resolve()),
        "reference_metric": "gt_vertex_mean_l2",
        "subject_split": str(cli_args.subject_split),
        "eval_fraction": float(cli_args.eval_fraction),
        "seed": int(base_args.seed),
        "max_subjects_requested": int(cli_args.max_subjects),
        "max_meshes_per_subject_eval": int(base_args.max_meshes_per_subject_eval),
        "pair_mode": str(cli_args.pair_mode),
        "aggregation_level": str(cli_args.aggregation_level),
        "selected_subjects": list(target_subjects),
        "inference_filter_summary": inference_filter_summary,
        "gt_filter_summary": gt_filter_summary,
        "gt_distance_matrix_summary": gt_dist_summary,
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
        "scenarios": [spec.__dict__ for spec in scenarios],
        "rows": rows,
    }

    with open(out_dir / "ranking_summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)
        handle.write("\n")
    _write_summary_csv(out_dir / "ranking_summary.csv", rows=rows, params=params, scenarios=scenarios)
    _write_summary_md(out_dir / "ranking_summary.md", rows=rows, params=params, scenarios=scenarios)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {out_dir}")
    print(f"GT dist matrix: {payload['dist_npz']}")
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
