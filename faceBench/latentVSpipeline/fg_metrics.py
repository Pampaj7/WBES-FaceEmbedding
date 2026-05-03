from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class PairMetricResult:
    raw_chamfer: float = math.nan
    rigid_chamfer: float = math.nan
    fg_p2p: float = math.nan
    fg_p2tri: float = math.nan
    fg_corrected_p2p: float = math.nan
    raw_seconds: float = math.nan
    rigid_seconds: float = math.nan
    fg_seconds: float = math.nan
    status: str = "ok"
    error: str = ""


def load_vertices(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        for key in ("V", "verts", "vertices", "points"):
            if key in data.files:
                return _as_vertices(data[key], path=path, key=key)
        raise KeyError(f"No vertex key found in {path}; available keys={data.files}")
    return _as_vertices(np.loadtxt(path), path=path, key="text")


def load_mm_json(path: str | Path | None) -> Optional[dict]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_mesh_path(row: dict, side: str, npz_root: str | Path | None) -> str:
    explicit_keys = [f"mesh_{side}", f"npz_path_{side}", f"path_{side}", f"{side}_path"]
    for key in explicit_keys:
        value = str(row.get(key, "")).strip()
        if value:
            return value

    sample_key = f"sample_name_{side}"
    sample_name = str(row.get(sample_key, "")).strip()
    if sample_name and npz_root:
        return str(Path(npz_root) / f"{sample_name}.npz")

    raise ValueError(
        f"Could not infer mesh path for side={side}. Expected one of "
        f"{explicit_keys}, or {sample_key} together with --npz_root."
    )


def symmetric_chamfer(X: np.ndarray, Y: np.ndarray) -> float:
    if X.size == 0 or Y.size == 0:
        return math.nan
    tree_y = cKDTree(Y)
    d_xy, _ = tree_y.query(X, k=1)
    tree_x = cKDTree(X)
    d_yx, _ = tree_x.query(Y, k=1)
    return float(0.5 * (np.mean(d_xy) + np.mean(d_yx)))


def one_way_correspondence(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    _, idx = cKDTree(Y).query(X, k=1)
    return np.asarray(idx, dtype=np.int64)


def p2p_distance(X: np.ndarray, Y: np.ndarray, pidx: np.ndarray) -> np.ndarray:
    return np.linalg.norm(X - Y[pidx], axis=1)


def p2tri_distance(X: np.ndarray, Y: np.ndarray, pidx: np.ndarray) -> np.ndarray:
    selected = Y[pidx]
    if len(selected) < 3:
        return p2p_distance(X, Y, pidx)
    _, neighbor_idx = cKDTree(selected).query(X, k=3)
    return np.asarray(
        [_point_triangle_distance(X[i], selected[neighbor_idx[i]]) for i in range(len(X))],
        dtype=np.float64,
    )


def sample_vertices(V: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    V = np.asarray(V, dtype=np.float64)
    if max_points <= 0 or len(V) <= max_points:
        return V
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(V), size=max_points, replace=False)
    return V[np.sort(idx)]


def rigid_icp_align(
    source: np.ndarray,
    target: np.ndarray,
    *,
    max_points: int = 4096,
    max_iter: int = 30,
    tolerance: float = 1.0e-7,
    seed: int = 1234,
) -> np.ndarray:
    src_full = np.asarray(source, dtype=np.float64)
    tgt_full = np.asarray(target, dtype=np.float64)
    src_work = sample_vertices(src_full, max_points=max_points, seed=seed)
    tgt_work = sample_vertices(tgt_full, max_points=max_points, seed=seed + 17)

    src_aligned = _bbox_prealign(src_work, tgt_work)
    full_aligned = _apply_bbox_prealign(src_full, source_ref=src_work, target_ref=tgt_work)
    tree = cKDTree(tgt_work)
    prev_error = math.inf

    for _ in range(max_iter):
        dists, nn_idx = tree.query(src_aligned, k=1)
        scale, rot, trans = procrustes_transform(src_aligned, tgt_work[nn_idx], scaling=True)
        src_aligned = scale * (src_aligned @ rot) + trans
        full_aligned = scale * (full_aligned @ rot) + trans
        mean_error = float(np.mean(dists))
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    return full_aligned


def compute_pair_metrics(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    stages: Iterable[str],
    mm: Optional[dict] = None,
    icp_points: int = 4096,
    icp_iter: int = 30,
    p2tri_points: int = 0,
    seed: int = 1234,
) -> PairMetricResult:
    stages = {stage.strip().lower() for stage in stages if stage.strip()}
    result = PairMetricResult()

    try:
        if "raw" in stages:
            start = time.time()
            result.raw_chamfer = symmetric_chamfer(X, Y)
            result.raw_seconds = float(time.time() - start)

        X_rigid = X
        if "rigid" in stages or "fg" in stages:
            start = time.time()
            X_rigid = rigid_icp_align(
                X,
                Y,
                max_points=icp_points,
                max_iter=icp_iter,
                seed=seed,
            )
            result.rigid_chamfer = symmetric_chamfer(X_rigid, Y)
            result.rigid_seconds = float(time.time() - start)

        if "fg" in stages:
            start = time.time()
            pidx = one_way_correspondence(X_rigid, Y)
            result.fg_p2p = float(np.mean(p2p_distance(X_rigid, Y, pidx)))
            X_tri = sample_vertices(X_rigid, p2tri_points, seed=seed + 99) if p2tri_points > 0 else X_rigid
            pidx_tri = one_way_correspondence(X_tri, Y)
            result.fg_p2tri = float(np.mean(p2tri_distance(X_tri, Y, pidx_tri)))
            if mm is not None and _mm_compatible(mm, len(Y)):
                Y_corr = topology_consistency_corrector(X_rigid, Y, pidx, mm)
                result.fg_corrected_p2p = float(np.mean(p2p_distance(X_rigid, Y_corr, pidx)))
            result.fg_seconds = float(time.time() - start)
    except Exception as exc:
        result.status = "failed"
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def procrustes_transform(source: np.ndarray, target: np.ndarray, scaling: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    if source.shape != target.shape:
        raise ValueError(f"Shape mismatch: source={source.shape}, target={target.shape}")
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    S = source - mu_s
    T = target - mu_t
    norm_s = max(float(np.linalg.norm(S)), 1.0e-12)
    norm_t = max(float(np.linalg.norm(T)), 1.0e-12)
    S0 = S / norm_s
    T0 = T / norm_t
    U, singular_values, Vt = np.linalg.svd(S0.T @ T0)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    scale = float(singular_values.sum() * norm_t / norm_s) if scaling else 1.0
    trans = mu_t - scale * (mu_s @ R)
    return scale, R, trans


def topology_consistency_corrector(
    X: np.ndarray,
    G: np.ndarray,
    corr: np.ndarray,
    mm: dict,
    correction_strategy: str = "pair",
    weight_power: str = "sqrt",
    weight_strategy: str = "mixed",
) -> np.ndarray:
    from scipy.sparse import spdiags
    from scipy.sparse.linalg import spsolve

    weights = compute_landmark_base_vertex_weights(mm, weight_power, weight_strategy)
    N = X.shape[0]
    Y = G[corr]
    corrected_dims: List[np.ndarray] = []

    for dim in range(3):
        r = X[:, dim]
        g = Y[:, dim]
        if correction_strategy != "pair":
            raise ValueError("Only pair correction is implemented in this standalone runner.")

        rix = np.argsort(r)
        r_sorted = r[rix]
        g_sorted = g[rix]
        c = np.empty(N)
        c[1:-1] = -(
            2 * g_sorted[1:-1]
            - g_sorted[:-2]
            - g_sorted[2:]
            - 2 * r_sorted[1:-1]
            + r_sorted[:-2]
            + r_sorted[2:]
        )
        c[0] = -(g_sorted[0] - g_sorted[1] - (r_sorted[0] - r_sorted[1]))
        c[-1] = -(g_sorted[-1] - g_sorted[-2] - (r_sorted[-1] - r_sorted[-2]))

        d0 = 2 * np.ones(N) + 2 * weights[rix]
        d1 = -np.ones(N)
        A = 0.5 * spdiags([d1, d0, d1], [-1, 0, 1], N, N, format="csc")
        correction = spsolve(A, c)

        irix = np.empty_like(rix)
        irix[rix] = np.arange(N)
        corrected_dims.append(correction[irix])

    dG = np.stack(corrected_dims, axis=1)
    G_corrected = G.copy()
    G_corrected[corr] -= dG
    return G_corrected


def compute_landmark_base_vertex_weights(mm: dict, weight_power: str, weight_strategy: str) -> np.ndarray:
    Xmean = np.asarray(mm["mean_face_shape"], dtype=np.float64).copy()
    lmk_indices = list(mm["lmk_indices"])
    lix = int(mm.get("leye_oc_rel_index", 36))
    rix = int(mm.get("reye_oc_rel_index", 45))
    iod = max(float(np.linalg.norm(Xmean[lmk_indices[lix]] - Xmean[lmk_indices[rix]])), 1.0e-12)
    Xmean /= iod
    Xmean_lmks = Xmean[lmk_indices]
    adists = np.linalg.norm(Xmean_lmks[:, None, :] - Xmean[None, :, :], axis=2)
    adists = np.maximum(adists, 0.01)
    weights_min = 1.0 / adists.min(axis=0)

    ref_lis = np.array(
        [0, 2, 4, 5, 7, 9, 20, 21, 23, 24, 26, 27, 29, 30, 19, 22, 25, 28, 13, 14, 18, 31, 33, 34, 35, 37, 44, 45, 46, 39, 40, 41, 49, 48, 50]
    )
    indices = np.array(lmk_indices)[ref_lis] if len(lmk_indices) == 51 else np.array(lmk_indices)
    adists2 = np.linalg.norm(Xmean[indices, None, :] - Xmean[None, :, :], axis=2)
    weights_mean = 1.0 / np.maximum(np.abs(np.mean(adists2, axis=0) - 0.48), 1.0e-6)

    if weight_strategy == "mixed":
        weights = (weights_min + weights_mean) / 2.0
    elif weight_strategy == "min":
        weights = weights_min
    elif weight_strategy == "mean":
        weights = weights_mean
    else:
        raise ValueError(f"Invalid weight_strategy: {weight_strategy}")
    weights = np.maximum(weights, 1.0)
    if weight_power == "square":
        weights = (weights**2) / (np.mean(weights**2) / np.mean(weights))
    elif weight_power == "sqrt":
        weights = np.sqrt(weights)
    else:
        raise ValueError(f"Invalid weight_power: {weight_power}")
    return weights


def _as_vertices(arr: np.ndarray, *, path: Path, key: str) -> np.ndarray:
    V = np.asarray(arr, dtype=np.float64)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"Expected {path}:{key} to have shape (N, 3), got {V.shape}")
    return V


def _bbox_prealign(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    return _apply_bbox_prealign(source, source_ref=source, target_ref=target)


def _apply_bbox_prealign(source: np.ndarray, *, source_ref: np.ndarray, target_ref: np.ndarray) -> np.ndarray:
    source_center = source_ref.mean(axis=0)
    target_center = target_ref.mean(axis=0)
    source_centered_ref = source_ref - source_center
    target_centered_ref = target_ref - target_center
    source_scale = max(float(np.linalg.norm(source_centered_ref, axis=1).max()), 1.0e-12)
    target_scale = max(float(np.linalg.norm(target_centered_ref, axis=1).max()), 1.0e-12)
    return (source - source_center) / source_scale * target_scale + target_center


def _point_triangle_distance(P: np.ndarray, TRI: np.ndarray) -> float:
    A, B, C = TRI[0], TRI[1], TRI[2]
    AB, AC, AP = B - A, C - A, P - A
    N = np.cross(AB, AC)
    N_norm = np.linalg.norm(N)

    def segment_distance(Q: np.ndarray, U: np.ndarray, V: np.ndarray) -> float:
        UV = V - U
        denom = float(np.dot(UV, UV))
        if denom <= 1.0e-18:
            return float(np.linalg.norm(Q - U))
        t = np.clip(np.dot(Q - U, UV) / denom, 0.0, 1.0)
        return float(np.linalg.norm(Q - (U + t * UV)))

    if N_norm <= 1.0e-18:
        return min(segment_distance(P, A, B), segment_distance(P, A, C), segment_distance(P, B, C))

    N_unit = N / N_norm
    dist_to_plane = np.dot(AP, N_unit)
    P_proj = P - dist_to_plane * N_unit

    v0, v1, v2 = C - A, B - A, P_proj - A
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    denom = dot00 * dot11 - dot01 * dot01
    if abs(float(denom)) <= 1.0e-18:
        return min(segment_distance(P, A, B), segment_distance(P, A, C), segment_distance(P, B, C))
    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    if u >= 0 and v >= 0 and u + v <= 1:
        return float(abs(dist_to_plane))
    return min(segment_distance(P, A, B), segment_distance(P, A, C), segment_distance(P, B, C))


def _mm_compatible(mm: dict, n_vertices: int) -> bool:
    shape = np.asarray(mm.get("mean_face_shape", []))
    return shape.ndim == 2 and shape.shape[0] == n_vertices
