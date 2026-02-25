#!/usr/bin/env python3
"""
One-mesh diagnostic for operator consistency.

It performs:
1) quick geometry/scale check between:
   - datasets/REMESH/npz_data_topo_500
   - datasets/REMESH/npz_data_topo_500_withops
2) one-mesh operator recomputation and evals comparison:
   - evals_old (stored in withops)
   - evals_new (recomputed from current verts/faces)

Goal: detect if operators are coherent or affected by scale mismatch.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np
import scipy  # must be imported before torch in some envs
import torch


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[4]

DIFFNET_SRC = Path("/equilibrium/lpampaloni/diffusion-net/src")
if str(DIFFNET_SRC) not in sys.path:
    sys.path.append(str(DIFFNET_SRC))

from diffusion_net.geometry import compute_operators  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check spectral-operator consistency on one mesh.")
    parser.add_argument(
        "--plain_dir",
        type=str,
        default=str(REPO_ROOT / "datasets" / "REMESH" / "npz_data_topo_500"),
    )
    parser.add_argument(
        "--withops_dir",
        type=str,
        default=str(REPO_ROOT / "datasets" / "REMESH" / "npz_data_topo_500_withops"),
    )
    parser.add_argument(
        "--sample_file",
        type=str,
        default="",
        help="Optional basename (e.g. id0000_GTready_original.npz). If empty, pick from --variant filter.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="original",
        help="Used only if --sample_file is empty.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--pick", type=str, default="first", choices=("first", "random"))
    parser.add_argument("--k_eig", type=int, default=128, help="requested eig count for recomputation")
    parser.add_argument(
        "--compare_k",
        type=int,
        default=30,
        help="number of lowest eigenvalues (from index 0) used in detailed report",
    )
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    parser.add_argument(
        "--out_json",
        type=str,
        default=str(THIS_FILE.parent / "operator_consistency_one_mesh.json"),
    )
    return parser.parse_args()


def list_common_files(plain_dir: Path, withops_dir: Path) -> List[str]:
    plain = {p.name for p in plain_dir.glob("*.npz")}
    ops = {p.name for p in withops_dir.glob("*.npz")}
    return sorted(list(plain & ops))


def pick_file(common_files: List[str], sample_file: str, variant: str, pick: str, seed: int) -> str:
    if sample_file:
        if sample_file not in common_files:
            raise FileNotFoundError(f"{sample_file} not found in common set")
        return sample_file

    cand = common_files
    if variant:
        suffix = f"_{variant.lower()}.npz"
        cand = [f for f in common_files if f.lower().endswith(suffix)]
        if not cand:
            raise RuntimeError(f"No files found with variant suffix '{suffix}'")

    if pick == "random":
        rng = np.random.default_rng(seed)
        return str(rng.choice(np.array(cand, dtype=object)))
    return cand[0]


def load_plain_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return data["V"].astype(np.float64), data["F"].astype(np.int64)


def load_withops_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return (
        data["verts"].astype(np.float64),
        data["faces"].astype(np.int64),
        data["evals"].astype(np.float64),
    )


def mesh_stats(V: np.ndarray) -> Dict[str, float]:
    bbox_min = V.min(axis=0)
    bbox_max = V.max(axis=0)
    bbox_size = bbox_max - bbox_min
    norms = np.linalg.norm(V, axis=1)
    return {
        "bbox_min_x": float(bbox_min[0]),
        "bbox_min_y": float(bbox_min[1]),
        "bbox_min_z": float(bbox_min[2]),
        "bbox_max_x": float(bbox_max[0]),
        "bbox_max_y": float(bbox_max[1]),
        "bbox_max_z": float(bbox_max[2]),
        "bbox_diag": float(np.linalg.norm(bbox_size)),
        "bbox_size_x": float(bbox_size[0]),
        "bbox_size_y": float(bbox_size[1]),
        "bbox_size_z": float(bbox_size[2]),
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std()),
        "max_norm": float(norms.max()),
        "mean_xyz_norm": float(np.linalg.norm(V.mean(axis=0))),
    }


def estimate_scale(V_src: np.ndarray, V_tgt: np.ndarray, eps: float) -> Dict[str, float]:
    src_c = V_src - V_src.mean(axis=0, keepdims=True)
    tgt_c = V_tgt - V_tgt.mean(axis=0, keepdims=True)

    num = float((src_c * tgt_c).sum())
    den = float((src_c * src_c).sum()) + eps
    s_ls = num / den

    src_rms = float(np.sqrt((src_c * src_c).sum(axis=1).mean()))
    tgt_rms = float(np.sqrt((tgt_c * tgt_c).sum(axis=1).mean()))
    s_rms = tgt_rms / (src_rms + eps)

    aligned = s_ls * src_c
    err = tgt_c - aligned
    err_rms = float(np.sqrt((err * err).sum(axis=1).mean()))

    return {
        "scale_ls_centered": float(s_ls),
        "scale_rms_centered": float(s_rms),
        "centered_alignment_rmse": err_rms,
    }


def compare_evals(evals_old: np.ndarray, evals_new: np.ndarray, compare_k: int, eps: float) -> Dict[str, float | list]:
    k = min(compare_k, evals_old.shape[0], evals_new.shape[0])
    if k < 3:
        raise RuntimeError(f"Not enough eigenvalues to compare: k={k}")

    old_k = evals_old[:k]
    new_k = evals_new[:k]

    # Skip lambda_0 for ratio diagnostics.
    old_nz = old_k[1:]
    new_nz = new_k[1:]
    mask = old_nz > eps

    if mask.sum() < 3:
        raise RuntimeError("Too few non-zero old eigenvalues for stable ratio analysis")

    old_v = old_nz[mask]
    new_v = new_nz[mask]
    ratio = new_v / (old_v + eps)

    slope_origin = float((old_v * new_v).sum() / ((old_v * old_v).sum() + eps))
    fit = slope_origin * old_v
    rel_rmse = float(np.sqrt(np.mean((new_v - fit) ** 2)) / (np.mean(np.abs(new_v)) + eps))

    ratio_mean = float(np.mean(ratio))
    ratio_std = float(np.std(ratio))
    ratio_cv = ratio_std / (abs(ratio_mean) + eps)
    ratio_median = float(np.median(ratio))
    ratio_min = float(np.min(ratio))
    ratio_max = float(np.max(ratio))

    # If lambda_new/lambda_old = c, then scale s ~= 1/sqrt(c)
    scale_from_ratio = float(1.0 / math.sqrt(ratio_median)) if ratio_median > 0 else float("nan")

    rel_abs = np.abs(new_v - old_v) / (np.abs(old_v) + eps)

    return {
        "k_compared": int(k),
        "ratio_mean": ratio_mean,
        "ratio_std": ratio_std,
        "ratio_cv": float(ratio_cv),
        "ratio_median": ratio_median,
        "ratio_min": ratio_min,
        "ratio_max": ratio_max,
        "scale_implied_from_ratio_median": scale_from_ratio,
        "slope_origin_new_vs_old": slope_origin,
        "fit_rel_rmse_new_vs_old": rel_rmse,
        "rel_abs_err_mean": float(np.mean(rel_abs)),
        "rel_abs_err_median": float(np.median(rel_abs)),
        "old_first10": old_k[:10].tolist(),
        "new_first10": new_k[:10].tolist(),
        "ratio_first10_from_lambda1": ratio[:10].tolist(),
    }


def main() -> None:
    args = parse_args()
    plain_dir = Path(args.plain_dir)
    withops_dir = Path(args.withops_dir)

    common_files = list_common_files(plain_dir, withops_dir)
    if not common_files:
        raise RuntimeError("No common .npz files found between plain_dir and withops_dir")

    sample_file = pick_file(
        common_files=common_files,
        sample_file=args.sample_file,
        variant=args.variant,
        pick=args.pick,
        seed=args.seed,
    )

    plain_path = plain_dir / sample_file
    withops_path = withops_dir / sample_file

    V_plain, F_plain = load_plain_npz(plain_path)
    V_ops, F_ops, evals_old = load_withops_npz(withops_path)

    if V_plain.shape != V_ops.shape:
        raise RuntimeError(f"Vertex shape mismatch: plain={V_plain.shape}, withops={V_ops.shape}")
    if F_plain.shape != F_ops.shape:
        raise RuntimeError(f"Face shape mismatch: plain={F_plain.shape}, withops={F_ops.shape}")

    face_equal = bool(np.array_equal(F_plain, F_ops))
    vert_abs_diff = np.abs(V_plain - V_ops)

    quick = {
        "file": sample_file,
        "face_equal": face_equal,
        "vert_abs_diff_max": float(vert_abs_diff.max()),
        "vert_abs_diff_mean": float(vert_abs_diff.mean()),
        "plain_stats": mesh_stats(V_plain),
        "withops_stats": mesh_stats(V_ops),
        "scale_estimate_plain_to_withops": estimate_scale(V_plain, V_ops, args.eps),
    }

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    V_t = torch.tensor(V_ops, dtype=torch.float32, device=device)
    F_t = torch.tensor(F_ops, dtype=torch.long, device=device)

    k_eig_use = min(args.k_eig, evals_old.shape[0], V_t.shape[0] - 2)
    if k_eig_use < 3:
        raise RuntimeError(f"k_eig_use too small: {k_eig_use}")

    print(f"[info] file: {sample_file}")
    print(f"[info] device: {device}")
    print(f"[info] recomputing operators with k_eig={k_eig_use} ...")
    ops = compute_operators(V_t, F_t, k_eig=int(k_eig_use))
    evals_new = ops[3].detach().cpu().numpy().astype(np.float64)

    spectral = compare_evals(
        evals_old=evals_old[:k_eig_use],
        evals_new=evals_new[:k_eig_use],
        compare_k=args.compare_k,
        eps=args.eps,
    )

    out = {
        "input": {
            "plain_dir": str(plain_dir),
            "withops_dir": str(withops_dir),
            "sample_file": sample_file,
            "device": str(device),
            "k_eig_use": int(k_eig_use),
        },
        "quick_geometry_check": quick,
        "spectral_check": spectral,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("")
    print("=== Quick Geometry Check ===")
    print(f"face_equal: {quick['face_equal']}")
    print(f"vert_abs_diff_max: {quick['vert_abs_diff_max']:.6e}")
    print(f"plain bbox_diag: {quick['plain_stats']['bbox_diag']:.6f}")
    print(f"withops bbox_diag: {quick['withops_stats']['bbox_diag']:.6f}")
    print(
        f"scale (rms centered plain->withops): "
        f"{quick['scale_estimate_plain_to_withops']['scale_rms_centered']:.6f}"
    )

    print("")
    print("=== Spectral Check ===")
    print("old first10:", np.array(spectral["old_first10"]))
    print("new first10:", np.array(spectral["new_first10"]))
    print(f"ratio mean:   {spectral['ratio_mean']:.6f}")
    print(f"ratio median: {spectral['ratio_median']:.6f}")
    print(f"ratio cv:     {spectral['ratio_cv']:.6f}")
    print(f"scale implied from ratio median: {spectral['scale_implied_from_ratio_median']:.6f}")
    print(f"slope (new~a*old): {spectral['slope_origin_new_vs_old']:.6f}")
    print(f"fit rel rmse: {spectral['fit_rel_rmse_new_vs_old']:.6f}")
    print("")
    print(f"saved json: {out_path}")


if __name__ == "__main__":
    main()
