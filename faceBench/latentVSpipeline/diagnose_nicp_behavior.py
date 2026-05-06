#!/usr/bin/env python3
"""Quick diagnostics for FaceBench NICP behavior on REMESH pairs.

This is intentionally small and investigative: it checks whether NICP improves
surface fitting, whether rigid initialization matters, and whether the resulting
correspondences separate same-subject from different-subject pairs.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy
from scipy.spatial import Delaunay, KDTree
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu, spsolve
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
FACBENCH_DIR = REPO_ROOT / "faceBench"
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FACBENCH_DIR))
sys.path.insert(0, str(THIS_DIR))

import facebench as fb
from mesh_npz_utils import load_normalized_vertices_npz


def sample_pts(V: np.ndarray, max_pts: int, seed: int) -> np.ndarray:
    if max_pts <= 0 or len(V) <= max_pts:
        return V
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(V), size=max_pts, replace=False)
    return V[np.sort(idx)]


def nn_mean(X: np.ndarray, Y: np.ndarray) -> float:
    tree = KDTree(Y)
    dists, _ = tree.query(X, k=1, p=2)
    return float(np.mean(dists))


def triangles_to_edge_vertex_adjacent_matrix(triangles: np.ndarray) -> csc_matrix:
    pairs = []
    for v1, v2, v3 in triangles.T:
        pairs += [tuple(sorted((v1, v2))), tuple(sorted((v1, v3))), tuple(sorted((v2, v3)))]
    pair_array = np.array(list(set(pairs)))
    edge_indices = np.arange(len(pair_array))
    i = np.concatenate([edge_indices, edge_indices])
    j = np.concatenate([pair_array[:, 0], pair_array[:, 1]])
    data = np.concatenate([np.ones(len(edge_indices)), -np.ones(len(edge_indices))])
    return csc_matrix((data, (i, j)), shape=(len(pair_array), np.max(triangles) + 1))


def sparse_matrix_from_vertices(src: np.ndarray) -> csc_matrix:
    ones = np.ones((1, src.shape[1]), dtype=np.float32)
    src_hom = np.vstack([src, ones])
    Di = np.arange(src.shape[1])
    Dj = np.stack([Di * 4 + i for i in range(4)], axis=0).reshape(-1)
    Di = np.tile(Di, 4)
    values = src_hom.reshape(-1)
    return csc_matrix((values, (Di, Dj)), shape=(src.shape[1], src.shape[1] * 4), dtype=np.float32)


def spsolve_system(A: csc_matrix, B: csc_matrix) -> np.ndarray:
    ATA = (A.T @ A).tocsc()
    LU = splu(ATA, diag_pivot_thresh=0)
    P1 = csc_matrix((np.ones(LU.L.shape[0]), (LU.perm_r, np.arange(LU.L.shape[0])))).T
    P2 = csc_matrix((np.ones(LU.L.shape[0]), (np.arange(LU.L.shape[0]), LU.perm_c))).T
    b_tilde = P1.T @ (A.T @ B)
    z = spsolve(LU.L, b_tilde)
    y = spsolve(LU.U, z)
    return (P2.T @ y).toarray()


def instrumented_nicp(
    source_points: np.ndarray,
    target_points: np.ndarray,
    gamma: float = 1.0,
    alpha: float = 50.0,
    epsilon: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    source_tri = Delaunay(source_points[:, :2]).simplices.T
    source = source_points.T
    target = target_points.T

    G = np.diag([1.0, 1.0, 1.0, gamma]).astype(np.float32)
    M = triangles_to_edge_vertex_adjacent_matrix(source_tri)
    A1 = scipy.sparse.kron(M, G)
    B1 = csc_matrix((A1.shape[0], 3), dtype=np.float32)

    cur_src = source
    cur_X = np.zeros((source.shape[1] * 4, 3))
    X = np.ones_like(cur_X)
    stats: Dict[str, float] = {
        "initial_nn": nn_mean(source_points, target_points),
        "iterations": 0.0,
        "last_weight_frac": math.nan,
        "last_delta": math.nan,
    }

    for decay in range(3):
        cur_alpha = alpha * np.exp(-0.5 * decay)
        eps = epsilon * (0.5 ** decay) * 6
        while np.linalg.norm(X - cur_X) > eps:
            cur_X = X.copy()
            tree = KDTree(target.T)
            nn_distances, nn_indices = tree.query(cur_src.T, k=1, p=2)
            cur_nn_dst = target[:, nn_indices.astype(np.int32)]

            D = sparse_matrix_from_vertices(cur_src)
            weights = (nn_distances < max(np.mean(nn_distances) * 2, 1)).astype(np.float32)
            A2 = D.multiply(weights[:, np.newaxis])
            B2 = csc_matrix(np.multiply(cur_nn_dst.T, weights[:, np.newaxis]))
            A = scipy.sparse.vstack([A1.multiply(cur_alpha), A2])
            B = scipy.sparse.vstack([B1, B2])
            X = spsolve_system(A, B)
            cur_src = (D @ X).T

            stats["iterations"] += 1.0
            stats["last_weight_frac"] = float(np.mean(weights))
            stats["last_delta"] = float(np.linalg.norm(X - cur_X))

    aligned = cur_src.T
    stats["final_nn"] = nn_mean(aligned, target_points)
    stats["mean_displacement"] = float(np.mean(np.linalg.norm(aligned - source_points, axis=1)))
    stats["max_displacement"] = float(np.max(np.linalg.norm(aligned - source_points, axis=1)))
    return aligned, stats


def eval_pair(npz_root: Path, sa: str, sb: str, topo_a: str, topo_b: str, max_pts: int, seed: int) -> Dict[str, float]:
    X = load_normalized_vertices_npz(npz_root / f"{sa}_GTready_{topo_a}.npz")
    Y = load_normalized_vertices_npz(npz_root / f"{sb}_GTready_{topo_b}.npz")
    Xs = sample_pts(X, max_pts, seed)
    Ys = sample_pts(Y, max_pts, seed + 1)

    direct, direct_stats = instrumented_nicp(Xs, Ys)
    corr_direct = fb.chamfer_correspondence(direct, Ys)
    direct_orig_p2p = float(np.mean(fb.p2p_distance(Xs, Ys, corr_direct)))
    direct_fit_p2p = float(np.mean(fb.p2p_distance(direct, Ys, corr_direct)))

    X_rigid, _ = fb.icp_align(Xs, Ys, prealign="bbox")
    rigid_nn = nn_mean(X_rigid, Ys)
    rigid_nicp, rigid_stats = instrumented_nicp(X_rigid, Ys)
    corr_rigid_nicp = fb.chamfer_correspondence(rigid_nicp, Ys)
    rigid_nicp_orig_p2p = float(np.mean(fb.p2p_distance(X_rigid, Ys, corr_rigid_nicp)))
    rigid_nicp_fit_p2p = float(np.mean(fb.p2p_distance(rigid_nicp, Ys, corr_rigid_nicp)))

    return {
        "direct_initial_nn": direct_stats["initial_nn"],
        "direct_final_nn": direct_stats["final_nn"],
        "direct_iters": direct_stats["iterations"],
        "direct_disp": direct_stats["mean_displacement"],
        "direct_orig_p2p": direct_orig_p2p,
        "direct_fit_p2p": direct_fit_p2p,
        "rigid_nn": rigid_nn,
        "rigid_nicp_final_nn": rigid_stats["final_nn"],
        "rigid_nicp_iters": rigid_stats["iterations"],
        "rigid_nicp_disp": rigid_stats["mean_displacement"],
        "rigid_nicp_orig_p2p": rigid_nicp_orig_p2p,
        "rigid_nicp_fit_p2p": rigid_nicp_fit_p2p,
    }


def fmt(v: float) -> str:
    return f"{v:.4f}" if math.isfinite(v) else "nan"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npz_root", default="datasets/REMESH/npz_data_topo_500")
    p.add_argument("--topo_a", default="crop")
    p.add_argument("--topo_b", default="down8k")
    p.add_argument("--max_subjects", type=int, default=10)
    p.add_argument("--max_sample_points", type=int, default=1024)
    args = p.parse_args()

    npz_root = Path(args.npz_root)
    subjects = sorted({p.stem.split("_GTready_")[0] for p in npz_root.glob("*_GTready_original.npz")})[: args.max_subjects]

    rows: List[Tuple[str, str, str, Dict[str, float]]] = []
    for i, sa in enumerate(subjects[: min(5, len(subjects))]):
        rows.append(("same", sa, sa, eval_pair(npz_root, sa, sa, args.topo_a, args.topo_b, args.max_sample_points, i)))
    for i, sa in enumerate(subjects[: min(5, len(subjects))]):
        sb = subjects[-(i + 1)]
        if sa != sb:
            rows.append(("diff", sa, sb, eval_pair(npz_root, sa, sb, args.topo_a, args.topo_b, args.max_sample_points, 100 + i)))

    header = (
        "kind pair direct_nn direct_fit direct_orig direct_iter direct_disp "
        "rigid_nn rigid_nicp_fit rigid_nicp_orig rigid_nicp_iter rigid_nicp_disp"
    )
    print(header)
    for kind, sa, sb, r in rows:
        print(
            kind,
            f"{sa}->{sb}",
            fmt(r["direct_initial_nn"]) + "->" + fmt(r["direct_final_nn"]),
            fmt(r["direct_fit_p2p"]),
            fmt(r["direct_orig_p2p"]),
            fmt(r["direct_iters"]),
            fmt(r["direct_disp"]),
            fmt(r["rigid_nn"]),
            fmt(r["rigid_nicp_fit_p2p"]),
            fmt(r["rigid_nicp_orig_p2p"]),
            fmt(r["rigid_nicp_iters"]),
            fmt(r["rigid_nicp_disp"]),
        )

    same_direct = [r["direct_orig_p2p"] for kind, _, _, r in rows if kind == "same"]
    diff_direct = [r["direct_orig_p2p"] for kind, _, _, r in rows if kind == "diff"]
    same_rigid = [r["rigid_nicp_orig_p2p"] for kind, _, _, r in rows if kind == "same"]
    diff_rigid = [r["rigid_nicp_orig_p2p"] for kind, _, _, r in rows if kind == "diff"]
    print()
    print(f"same direct mean: {fmt(float(np.mean(same_direct)))} | diff direct mean: {fmt(float(np.mean(diff_direct)))}")
    print(f"same rigid+nicp mean: {fmt(float(np.mean(same_rigid)))} | diff rigid+nicp mean: {fmt(float(np.mean(diff_rigid)))}")


if __name__ == "__main__":
    main()
