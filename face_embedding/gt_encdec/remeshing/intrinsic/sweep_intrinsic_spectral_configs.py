#!/usr/bin/env python3
"""
Cross-configuration sweep for intrinsic spectral baselines.

This script DOES NOT modify the existing HKS baseline file.
It evaluates multiple descriptor/projection/embedding/distance configurations and
writes a CSV report with one row per configuration.

Main pipeline per mesh:
  descriptor (HKS/GPS/XYZ/combos) -> spectral projection (Phi^T M X) ->
  embedding reduction (flatten / row_l2 / svdvals / ...) -> subject mean

Main metrics per configuration:
  - Spearman(D_GT, D_emb)
  - Pearson(D_GT, D_emb)
  - NN match rate
  - intra-subject latent variance
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm


# ------------------------------------------------------------
# Paths and imports
# ------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[4]
AUTOENCODER_DIR = REPO_ROOT / "face_embedding" / "gt_encdec" / "autoencoder"
if str(AUTOENCODER_DIR) not in sys.path:
    sys.path.append(str(AUTOENCODER_DIR))

from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset  # noqa: E402


SUBJECT_RE = re.compile(r"(id\d{4})", re.IGNORECASE)


@dataclass(frozen=True)
class SweepConfig:
    descriptor: str
    n_hks: int
    n_gps: int
    hks_times_mode: str
    proj_k: int
    embed_type: str
    distance_mode: str
    abs_coeffs: bool
    fix_evec_sign: bool
    standardize_features: bool
    l2_normalize_embedding: bool
    svd_k: int


def parse_args() -> argparse.Namespace:
    default_data_dir = REPO_ROOT / "datasets" / "REMESH" / "npz_data_topo_500_withops"
    default_dist = (
        REPO_ROOT
        / "face_embedding"
        / "gt_encdec"
        / "autoencoder"
        / "latent_analysis"
        / "gt_distance_matrix"
        / "normalized_matrix_distances.npz"
    )
    default_out = THIS_FILE.parent / "runs_intrinsic_sweep"

    parser = argparse.ArgumentParser(description="Sweep intrinsic spectral baseline configurations.")
    parser.add_argument("--data_dir", type=str, default=str(default_data_dir))
    parser.add_argument("--dist_npz", type=str, default=str(default_dist))
    parser.add_argument("--out_dir", type=str, default=str(default_out))

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_subjects", type=int, default=300, help="0 = all subjects")

    parser.add_argument("--sweep_mode", type=str, default="curated", choices=("curated", "grid"))
    parser.add_argument("--max_configs", type=int, default=0, help="0 = all generated configs")

    # Grid options (used in --sweep_mode grid)
    parser.add_argument("--descriptors", type=str, default="hks,gps,xyz,hks_gps")
    parser.add_argument("--n_hks_list", type=str, default="16")
    parser.add_argument("--n_gps_list", type=str, default="32")
    parser.add_argument("--hks_times_modes", type=str, default="autoscale,spectral")
    parser.add_argument("--proj_ks", type=str, default="16,32")
    parser.add_argument("--embed_types", type=str, default="flatten,row_l2,svdvals")
    parser.add_argument("--distance_modes", type=str, default="l2,cosine")
    parser.add_argument("--abs_coeffs_opts", type=str, default="0,1")
    parser.add_argument("--fix_evec_sign_opts", type=str, default="1")
    parser.add_argument("--standardize_features_opts", type=str, default="1")
    parser.add_argument("--l2_norm_opts", type=str, default="1")

    # Shared hyperparams
    parser.add_argument("--hks_k_high", type=int, default=50)
    parser.add_argument("--svd_k", type=int, default=16)
    parser.add_argument("--eps", type=float, default=1e-6)
    return parser.parse_args()


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_bool_list01(text: str) -> List[bool]:
    vals = []
    for tok in text.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        vals.append(tok in ("1", "true", "True", "yes", "y"))
    return vals


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_subject_id(name: str) -> Optional[str]:
    m = SUBJECT_RE.search(name)
    return m.group(1).lower() if m else None


def build_subject_map(files: Sequence[str]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for idx, fname in enumerate(files):
        sid = extract_subject_id(fname)
        if sid is None:
            continue
        out.setdefault(sid, []).append(idx)
    return out


def load_gt_distance_matrix(path: str) -> Tuple[np.ndarray, Dict[str, int]]:
    pack = np.load(path, allow_pickle=True)
    if "D_orig" not in pack or "names" not in pack:
        raise KeyError(f"{path} must contain D_orig and names. Found: {pack.files}")

    D = pack["D_orig"].astype(np.float32)
    mask = D > 0
    if mask.any():
        D = D / float(D[mask].max())

    name_to_idx: Dict[str, int] = {}
    for i, n in enumerate(pack["names"]):
        if isinstance(n, bytes):
            n = n.decode("utf-8", errors="ignore")
        sid = extract_subject_id(str(n))
        if sid is not None:
            name_to_idx[sid] = i

    if not name_to_idx:
        raise RuntimeError("Could not parse subject ids from GT matrix names")
    return D, name_to_idx


def canonicalize_evec_sign(evecs: torch.Tensor, mass: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    moments = (mass[:, None] * (evecs ** 3)).sum(dim=0)
    signs = torch.where(moments >= 0, torch.ones_like(moments), -torch.ones_like(moments))

    near_zero = moments.abs() < eps
    if near_zero.any():
        idx = evecs.abs().argmax(dim=0)
        alt_sign = torch.sign(evecs[idx, torch.arange(evecs.shape[1], device=evecs.device)])
        alt_sign = torch.where(alt_sign == 0, torch.ones_like(alt_sign), alt_sign)
        signs = torch.where(near_zero, alt_sign, signs)

    return evecs * signs[None, :]


def mass_weighted_standardize(x: torch.Tensor, mass: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    w = mass / (mass.sum() + eps)
    mu = (w[:, None] * x).sum(dim=0, keepdim=True)
    var = (w[:, None] * (x - mu) ** 2).sum(dim=0, keepdim=True)
    return (x - mu) / torch.sqrt(var + 1e-4)


def compute_hks(
    evals: torch.Tensor,
    evecs: torch.Tensor,
    n_hks: int,
    hks_k_high: int,
    eps: float,
    times_mode: str,
) -> torch.Tensor:
    n_verts = evecs.shape[0]
    if n_hks <= 0:
        return torch.zeros(n_verts, 0, device=evecs.device, dtype=evecs.dtype)

    k = min(evals.numel(), evecs.shape[1])
    if k < 3:
        return torch.zeros(n_verts, n_hks, device=evecs.device, dtype=evecs.dtype)

    evals_k = evals[:k].clamp_min(eps)
    evecs_k = evecs[:, :k]

    if times_mode == "autoscale":
        t = torch.logspace(-2.0, 0.0, steps=n_hks, device=evals.device, dtype=evals.dtype)
    else:
        lam_low = evals_k[1]
        lam_high = evals_k[min(k - 1, hks_k_high)]
        t_min = float((4.0 / lam_high).detach().cpu())
        t_max = float((4.0 / lam_low).detach().cpu())
        if not (np.isfinite(t_min) and np.isfinite(t_max) and t_min > 0 and t_max > t_min):
            t_min, t_max = 1e-2, 1e2
        t = torch.logspace(
            math.log10(t_min),
            math.log10(t_max),
            steps=n_hks,
            device=evals.device,
            dtype=evals.dtype,
        )

    hks = (evecs_k ** 2) @ torch.exp(-evals_k[:, None] * t[None, :])
    return torch.log(hks + eps)


def compute_gps(
    evals: torch.Tensor,
    evecs: torch.Tensor,
    n_gps: int,
    eps: float,
) -> torch.Tensor:
    n_verts = evecs.shape[0]
    if n_gps <= 0:
        return torch.zeros(n_verts, 0, device=evecs.device, dtype=evecs.dtype)

    k = min(n_gps, evecs.shape[1] - 1, evals.numel() - 1)
    if k <= 0:
        return torch.zeros(n_verts, n_gps, device=evecs.device, dtype=evecs.dtype)

    phi = evecs[:, 1 : 1 + k]
    lam = evals[1 : 1 + k].clamp_min(eps)
    gps = phi / torch.sqrt(lam)[None, :]

    if k < n_gps:
        pad = torch.zeros(n_verts, n_gps - k, device=evecs.device, dtype=evecs.dtype)
        gps = torch.cat([gps, pad], dim=1)
    return gps


def to_basis(values: torch.Tensor, basis: torch.Tensor, massvec: torch.Tensor) -> torch.Tensor:
    return basis.transpose(-2, -1) @ (values * massvec[:, None])


def project_features(
    features: torch.Tensor,
    evecs: torch.Tensor,
    mass: torch.Tensor,
    proj_k: int,
    abs_coeffs: bool,
) -> torch.Tensor:
    k = min(proj_k, evecs.shape[1])
    basis = evecs[:, :k]
    coeff = to_basis(features, basis, mass)  # [k, C]
    if abs_coeffs:
        coeff = coeff.abs()

    if k < proj_k:
        pad = torch.zeros(proj_k - k, coeff.shape[1], device=coeff.device, dtype=coeff.dtype)
        coeff = torch.cat([coeff, pad], dim=0)
    return coeff


def coeff_to_embedding(coeff: torch.Tensor, embed_type: str, svd_k: int) -> torch.Tensor:
    if embed_type == "flatten":
        return coeff.reshape(-1)
    if embed_type == "row_l2":
        return torch.linalg.vector_norm(coeff, dim=1)
    if embed_type == "col_l2":
        return torch.linalg.vector_norm(coeff, dim=0)
    if embed_type == "row_mean":
        return coeff.mean(dim=1)
    if embed_type == "col_mean":
        return coeff.mean(dim=0)
    if embed_type == "svdvals":
        s = torch.linalg.svdvals(coeff)
        k = min(svd_k, s.shape[0])
        out = s[:k]
        if k < svd_k:
            out = torch.cat([out, torch.zeros(svd_k - k, device=s.device, dtype=s.dtype)], dim=0)
        return out
    raise ValueError(f"Unknown embed_type: {embed_type}")


def pairwise_distance_matrix(Z: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "l2":
        return torch.cdist(Z, Z, p=2)
    if mode == "cosine":
        Zn = torch.nn.functional.normalize(Z, dim=1)
        sim = (Zn @ Zn.T).clamp(-1.0, 1.0)
        return (1.0 - sim).clamp_min(0.0)
    raise ValueError(f"Unknown distance mode: {mode}")


def rankdata_average_ties(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)

    i = 0
    while i < sorted_vals.shape[0]:
        j = i + 1
        while j < sorted_vals.shape[0] and sorted_vals[j] == sorted_vals[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_corr(rankdata_average_ties(x), rankdata_average_ties(y))


def upper_triangular_values(M: torch.Tensor) -> torch.Tensor:
    iu = torch.triu_indices(M.shape[0], M.shape[1], offset=1, device=M.device)
    return M[iu[0], iu[1]]


def nn_match_rate(D_gt: torch.Tensor, D_emb: torch.Tensor) -> float:
    n = D_gt.shape[0]
    if n < 2:
        return float("nan")
    eye = torch.eye(n, dtype=torch.bool, device=D_gt.device)
    nn_gt = D_gt.masked_fill(eye, float("inf")).argmin(dim=1)
    nn_em = D_emb.masked_fill(eye, float("inf")).argmin(dim=1)
    return float((nn_gt == nn_em).float().mean().item())


def descriptor_tokens(descriptor: str) -> Tuple[bool, bool, bool]:
    parts = set(descriptor.lower().split("_"))
    return ("xyz" in parts, "hks" in parts, "gps" in parts)


def build_mesh_embedding(
    cfg: SweepConfig,
    sample: dict,
    device: torch.device,
    hks_k_high: int,
    eps: float,
) -> torch.Tensor:
    V = sample["verts"].to(device)
    mass = sample["mass"].to(device)
    evals = sample["evals"].to(device)
    evecs = sample["evecs"].to(device)

    if cfg.fix_evec_sign:
        evecs = canonicalize_evec_sign(evecs, mass, eps=eps)

    use_xyz, use_hks, use_gps = descriptor_tokens(cfg.descriptor)
    feats = []
    if use_xyz:
        feats.append(V)
    if use_hks:
        feats.append(
            compute_hks(
                evals=evals,
                evecs=evecs,
                n_hks=cfg.n_hks,
                hks_k_high=hks_k_high,
                eps=eps,
                times_mode=cfg.hks_times_mode,
            )
        )
    if use_gps:
        feats.append(compute_gps(evals=evals, evecs=evecs, n_gps=cfg.n_gps, eps=eps))

    if not feats:
        raise RuntimeError(f"Descriptor produced zero channels: {cfg.descriptor}")

    X = torch.cat(feats, dim=1)
    if cfg.standardize_features:
        X = mass_weighted_standardize(X, mass, eps=eps)

    coeff = project_features(
        features=X,
        evecs=evecs,
        mass=mass,
        proj_k=cfg.proj_k,
        abs_coeffs=cfg.abs_coeffs,
    )
    z = coeff_to_embedding(coeff, embed_type=cfg.embed_type, svd_k=cfg.svd_k)

    if cfg.l2_normalize_embedding:
        z = torch.nn.functional.normalize(z[None, :], dim=1).squeeze(0)
    return z


def curated_configs(args: argparse.Namespace) -> List[SweepConfig]:
    # Targeted experiments from your current direction: HKS baseline, GPS, non-flatten, XYZ projection.
    base = dict(
        proj_k=32,
        fix_evec_sign=True,
        standardize_features=True,
        l2_normalize_embedding=True,
        svd_k=args.svd_k,
    )

    cfgs = [
        SweepConfig("hks", 16, 0, "autoscale", base["proj_k"], "flatten", "l2", True, True, True, True, base["svd_k"]),
        SweepConfig("hks", 16, 0, "autoscale", base["proj_k"], "row_l2", "l2", True, True, True, True, base["svd_k"]),
        SweepConfig("hks", 16, 0, "autoscale", base["proj_k"], "svdvals", "l2", True, True, True, True, base["svd_k"]),
        SweepConfig("hks", 16, 0, "spectral", base["proj_k"], "flatten", "l2", True, True, True, True, base["svd_k"]),
        SweepConfig("gps", 0, 32, "autoscale", base["proj_k"], "flatten", "l2", False, True, True, True, base["svd_k"]),
        SweepConfig("gps", 0, 32, "autoscale", base["proj_k"], "row_l2", "l2", False, True, True, True, base["svd_k"]),
        SweepConfig("gps", 0, 32, "autoscale", base["proj_k"], "svdvals", "l2", False, True, True, True, base["svd_k"]),
        SweepConfig("xyz", 0, 0, "autoscale", base["proj_k"], "flatten", "l2", False, True, True, True, base["svd_k"]),
        SweepConfig("xyz", 0, 0, "autoscale", base["proj_k"], "row_l2", "l2", False, True, True, True, base["svd_k"]),
        SweepConfig("hks_gps", 16, 32, "autoscale", base["proj_k"], "flatten", "l2", False, True, True, True, base["svd_k"]),
        SweepConfig("hks_gps", 16, 32, "autoscale", base["proj_k"], "row_l2", "l2", False, True, True, True, base["svd_k"]),
        SweepConfig("hks_gps", 16, 32, "autoscale", base["proj_k"], "flatten", "cosine", False, True, True, True, base["svd_k"]),
    ]
    return cfgs


def grid_configs(args: argparse.Namespace) -> List[SweepConfig]:
    descriptors = parse_str_list(args.descriptors)
    proj_ks = parse_int_list(args.proj_ks)
    embed_types = parse_str_list(args.embed_types)
    distance_modes = parse_str_list(args.distance_modes)

    hks_times_modes = parse_str_list(args.hks_times_modes)
    n_hks_list = parse_int_list(args.n_hks_list)
    n_gps_list = parse_int_list(args.n_gps_list)

    abs_opts = parse_bool_list01(args.abs_coeffs_opts)
    fix_opts = parse_bool_list01(args.fix_evec_sign_opts)
    std_opts = parse_bool_list01(args.standardize_features_opts)
    l2_opts = parse_bool_list01(args.l2_norm_opts)

    out: List[SweepConfig] = []
    for descriptor in descriptors:
        use_xyz, use_hks, use_gps = descriptor_tokens(descriptor)
        if not (use_xyz or use_hks or use_gps):
            continue

        eff_hks_modes = hks_times_modes if use_hks else ["autoscale"]
        eff_hks_list = n_hks_list if use_hks else [0]
        eff_gps_list = n_gps_list if use_gps else [0]

        for (
            n_hks,
            n_gps,
            hks_mode,
            proj_k,
            embed_type,
            dist_mode,
            abs_coeffs,
            fix_sign,
            std_feat,
            l2_norm,
        ) in product(
            eff_hks_list,
            eff_gps_list,
            eff_hks_modes,
            proj_ks,
            embed_types,
            distance_modes,
            abs_opts,
            fix_opts,
            std_opts,
            l2_opts,
        ):
            out.append(
                SweepConfig(
                    descriptor=descriptor,
                    n_hks=int(n_hks),
                    n_gps=int(n_gps),
                    hks_times_mode=hks_mode,
                    proj_k=int(proj_k),
                    embed_type=embed_type,
                    distance_mode=dist_mode,
                    abs_coeffs=bool(abs_coeffs),
                    fix_evec_sign=bool(fix_sign),
                    standardize_features=bool(std_feat),
                    l2_normalize_embedding=bool(l2_norm),
                    svd_k=int(args.svd_k),
                )
            )

    # Deduplicate exact duplicates while preserving order.
    seen = set()
    uniq: List[SweepConfig] = []
    for c in out:
        key = tuple(asdict(c).items())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def evaluate_config(
    cfg: SweepConfig,
    dataset: GTReadyDataset,
    subject_map: Dict[str, List[int]],
    subjects: Sequence[str],
    gt_matrix: np.ndarray,
    gt_name_to_idx: Dict[str, int],
    device: torch.device,
    hks_k_high: int,
    eps: float,
) -> Dict[str, float | int | str]:
    t0 = time.time()

    embeddings = []
    valid_subjects = []
    intra_vals = []
    n_mesh_used = 0

    for sid in subjects:
        mesh_embs = []
        for mesh_idx in subject_map[sid]:
            try:
                sample = dataset[int(mesh_idx)]
            except Exception:
                continue

            z = build_mesh_embedding(
                cfg=cfg,
                sample=sample,
                device=device,
                hks_k_high=hks_k_high,
                eps=eps,
            )
            mesh_embs.append(z)
            n_mesh_used += 1

        if not mesh_embs:
            continue

        z_stack = torch.stack(mesh_embs, dim=0)
        z_subj = z_stack.mean(dim=0)
        embeddings.append(z_subj)
        valid_subjects.append(sid)
        intra_vals.append(float(((z_stack - z_subj[None, :]) ** 2).mean().item()))

    elapsed = time.time() - t0

    row: Dict[str, float | int | str] = {
        "descriptor": cfg.descriptor,
        "n_hks": cfg.n_hks,
        "n_gps": cfg.n_gps,
        "hks_times_mode": cfg.hks_times_mode,
        "proj_k": cfg.proj_k,
        "embed_type": cfg.embed_type,
        "distance_mode": cfg.distance_mode,
        "abs_coeffs": int(cfg.abs_coeffs),
        "fix_evec_sign": int(cfg.fix_evec_sign),
        "standardize_features": int(cfg.standardize_features),
        "l2_normalize_embedding": int(cfg.l2_normalize_embedding),
        "svd_k": cfg.svd_k,
        "n_subjects_used": int(len(valid_subjects)),
        "n_mesh_used": int(n_mesh_used),
        "intra_mean": float(np.mean(intra_vals)) if intra_vals else float("nan"),
        "spearman": float("nan"),
        "pearson": float("nan"),
        "nn_match_rate": float("nan"),
        "elapsed_sec": float(elapsed),
    }

    if len(embeddings) < 3:
        return row

    Z = torch.stack(embeddings, dim=0)
    D_emb = pairwise_distance_matrix(Z, cfg.distance_mode)
    gt_idx = np.array([gt_name_to_idx[s] for s in valid_subjects], dtype=int)
    D_gt = torch.tensor(gt_matrix[np.ix_(gt_idx, gt_idx)], dtype=Z.dtype, device=Z.device)

    gt_vals = upper_triangular_values(D_gt).detach().cpu().numpy()
    em_vals = upper_triangular_values(D_emb).detach().cpu().numpy()

    row["spearman"] = spearman_corr(gt_vals, em_vals)
    row["pearson"] = pearson_corr(gt_vals, em_vals)
    row["nn_match_rate"] = nn_match_rate(D_gt, D_emb)
    return row


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")
    print(f"Sweep mode: {args.sweep_mode}")

    dataset = GTReadyDataset(args.data_dir)
    subject_map = build_subject_map(dataset.files)
    if not subject_map:
        raise RuntimeError("No subject IDs parsed from dataset files.")

    gt_matrix, gt_name_to_idx = load_gt_distance_matrix(args.dist_npz)
    subjects = sorted([s for s in subject_map.keys() if s in gt_name_to_idx])
    if len(subjects) < 3:
        raise RuntimeError(f"Not enough overlapping subjects with GT matrix: {len(subjects)}")

    if args.max_subjects > 0 and len(subjects) > args.max_subjects:
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(subjects, size=args.max_subjects, replace=False)
        subjects = sorted(pick.tolist())

    if args.sweep_mode == "curated":
        cfgs = curated_configs(args)
    else:
        cfgs = grid_configs(args)

    if args.max_configs > 0:
        cfgs = cfgs[: args.max_configs]
    if not cfgs:
        raise RuntimeError("No configurations generated.")

    print(f"Subjects used: {len(subjects)}")
    print(f"Configs to evaluate: {len(cfgs)}")

    out_dir = Path(args.out_dir)
    config_json = out_dir / "sweep_configs.json"
    with open(config_json, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in cfgs], f, indent=2)

    report_csv = out_dir / "sweep_report.csv"
    fields = [
        "config_id",
        "descriptor",
        "n_hks",
        "n_gps",
        "hks_times_mode",
        "proj_k",
        "embed_type",
        "distance_mode",
        "abs_coeffs",
        "fix_evec_sign",
        "standardize_features",
        "l2_normalize_embedding",
        "svd_k",
        "n_subjects_used",
        "n_mesh_used",
        "intra_mean",
        "spearman",
        "pearson",
        "nn_match_rate",
        "elapsed_sec",
    ]

    rows = []
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i, cfg in enumerate(cfgs, start=1):
            desc = f"{cfg.descriptor}|{cfg.embed_type}|{cfg.distance_mode}"
            print(f"\n[{i}/{len(cfgs)}] {desc}")
            row = evaluate_config(
                cfg=cfg,
                dataset=dataset,
                subject_map=subject_map,
                subjects=tqdm(subjects, desc=f"cfg{i}", leave=False, ncols=100),
                gt_matrix=gt_matrix,
                gt_name_to_idx=gt_name_to_idx,
                device=device,
                hks_k_high=args.hks_k_high,
                eps=args.eps,
            )
            row["config_id"] = i
            rows.append(row)
            writer.writerow(row)
            f.flush()

            sp = row["spearman"]
            pr = row["pearson"]
            nn = row["nn_match_rate"]
            print(f"  Spearman={sp:.4f} | Pearson={pr:.4f} | NN={nn:.4f} | t={row['elapsed_sec']:.1f}s")

    # Save sorted view by Spearman
    rows_sorted = sorted(rows, key=lambda r: (float("-inf") if np.isnan(r["spearman"]) else r["spearman"]), reverse=True)
    with open(out_dir / "sweep_report_sorted_by_spearman.json", "w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, indent=2)

    print("")
    print(f"CSV report: {report_csv}")
    print(f"Config list: {config_json}")
    print(f"Sorted JSON: {out_dir / 'sweep_report_sorted_by_spearman.json'}")


if __name__ == "__main__":
    main()
