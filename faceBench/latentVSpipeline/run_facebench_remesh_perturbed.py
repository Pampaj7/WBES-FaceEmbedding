#!/usr/bin/env python3
"""
run_facebench_remesh_perturbed.py

FaceBench geometry pipeline vs latent embedding on REMESH cross-topology pairs
UNDER PERTURBATIONS matching dn_mixed_topology_v1 training settings.

For each (sigma, noise_mode) scenario, perturbs BOTH meshes in each pair,
then computes latent distance AND FG pipeline metrics on the perturbed meshes.

Perturbation settings from dn_mixed_topology_v1:
  noise_modes: translation, rotation, jitter
  sigma eval grid: logspace(0.001, 0.1, 6)
  rigid_rot_deg=12, rigid_rot_deg_min=0.5
  rigid_trans_scale=0.03, rigid_trans_scale_min=0.001

Outputs:
  - per-scenario ranking_summary.csv
  - overall perturbation sweep summary
  - clean baseline for comparison

Usage (from repo root):
  python faceBench/latentVSpipeline/run_facebench_remesh_perturbed.py \
    --npz_root datasets/REMESH/npz_data_topo_500 \
    --withops_root datasets/REMESH/npz_data_topo_500_withops \
    --checkpoint face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1/mixed_xtopo_rank0p5_id0p25_bs5_best/checkpoints/best_by_xtopo_mesh_clean.pth \
    --model_config face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1/mixed_xtopo_rank0p5_id0p25_bs5_best/config.json \
    --gt_matrix face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz \
    --out_dir faceBench/latentVSpipeline/outputs/perturbed_sweep \
    --max_subjects 100 \
    --stages raw,rigid,nicp,nicp_direct \
    --max_sample_points 4096 \
    --workers 8
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, fields
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr, pearsonr

# ── repo root on path ──────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
FACBENCH_DIR = REPO_ROOT / "faceBench"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "diffusion-net" / "src"))
sys.path.insert(0, str(FACBENCH_DIR))
sys.path.insert(0, str(REPO_ROOT / "face_embedding" / "gt_encdec" / "autoencoder"))

import facebench as fb
from facebench.rigid_aligners.icp import prealign_by_bbox
from mesh_npz_utils import load_normalized_vertices_npz, load_withops_sample_npz

# ── import perturbation utilities from the robustness package ──────────────
from face_embedding.gt_encdec.remeshing.intrinsic.robustness.noise import (
    PerturbationParams,
    apply_xyz_perturbation,
    rigid_angle_max_deg_from_sigma,
    rigid_trans_axis_std_from_sigma,
)

TOPOLOGIES = ["crop", "down8k", "noisy", "original", "remesh", "up60k"]

ALL_TOPO_PAIRS: List[Tuple[str, str]] = [
    (a, b) for a in TOPOLOGIES for b in TOPOLOGIES if a != b
]

# ── Default perturbation settings from dn_mixed_topology_v1 ────────────────
DEFAULT_SIGMA_GRID = [0.001, 0.00251, 0.00631, 0.01585, 0.03981, 0.1]
DEFAULT_NOISE_MODES = ["translation", "rotation", "jitter"]
DEFAULT_PERTURB_PARAMS = {
    "rigid_rot_deg": 12.0,
    "rigid_rot_deg_min": 0.5,
    "rigid_trans_scale": 0.03,
    "rigid_trans_scale_min": 0.001,
    "outlier_frac": 0.02,
    "outlier_scale": 6.0,
}


# ── deterministic seeds ──────────────────────────────────────────────────────

def perturb_seed(sample_name: str, scenario_idx: int) -> int:
    """Stable 32-bit seed for perturbing a given mesh in a given scenario.
    Used identically in the latent re-embed loop and in FG workers so that
    the same mesh within a scenario gets the same perturbation everywhere."""
    key = f"{Path(sample_name).name}:{int(scenario_idx)}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


# ── data structures ──────────────────────────────────────────────────────────

@dataclass
class PairRecord:
    subject_a: str
    topology_a: str
    sample_name_a: str
    subject_b: str
    topology_b: str
    sample_name_b: str
    gt_distance: float
    latent_distance: float
    raw_chamfer: float = math.nan
    rigid_p2p: float = math.nan
    nicp_p2p: float = math.nan
    nicp_p2tri: float = math.nan
    nicp_direct_p2p: float = math.nan
    nicp_direct_p2tri: float = math.nan
    nicp_bbox_p2p: float = math.nan
    nicp_bbox_p2tri: float = math.nan
    status: str = "ok"
    error: str = ""


# ── NPZ loading ──────────────────────────────────────────────────────────────

def load_verts(path: Path, scale: float = 1.0) -> np.ndarray:
    """Load vertices from a plain-geometry NPZ (keys V or verts)."""
    return load_normalized_vertices_npz(path, scale=scale)


def sample_pts(V: np.ndarray, max_pts: int, seed: int = 0) -> np.ndarray:
    if max_pts <= 0 or len(V) <= max_pts:
        return V
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(V), size=max_pts, replace=False)
    return V[np.sort(idx)]


# ── perturbation helpers ─────────────────────────────────────────────────────

def perturb_verts_torch(
    V_np: np.ndarray,
    mode: str,
    sigma: float,
    params: PerturbationParams,
    seed: int,
    device: str = "cuda",
) -> np.ndarray:
    """Apply perturbation to vertices using torch, return numpy array."""
    import torch
    rng = np.random.default_rng(seed)

    V_t = torch.tensor(V_np, dtype=torch.float32, device=device)

    if sigma <= 0.0:
        return V_np

    V_pert = apply_xyz_perturbation(
        V=V_t,
        mode=mode,
        sigma=sigma,
        outlier_frac=params.outlier_frac,
        outlier_scale=params.outlier_scale,
        rigid_rot_deg=params.rigid_rot_deg,
        rigid_trans_scale=params.rigid_trans_scale,
        rigid_rot_deg_min=params.rigid_rot_deg_min,
        rigid_trans_scale_min=params.rigid_trans_scale_min,
    )
    return V_pert.detach().cpu().numpy()


def perturb_verts_np(
    V_np: np.ndarray,
    mode: str,
    sigma: float,
    params: PerturbationParams,
    seed: int,
) -> np.ndarray:
    """CPU-only perturbation using numpy (no torch dependency for multiprocessing workers)."""
    rng = np.random.default_rng(seed)

    if sigma <= 0.0:
        return V_np.copy()

    V = V_np.astype(np.float64, copy=True)

    if mode == "jitter":
        noise = rng.normal(0.0, sigma, size=V.shape)
        return V + noise

    if mode in ("rigid", "rotation", "translation"):
        center = V.mean(axis=0, keepdims=True)
        V_out = V.copy()

        if mode in ("rigid", "rotation"):
            axis = rng.normal(0.0, 1.0, size=(3,))
            axis = axis / (np.linalg.norm(axis) + 1e-12)

            angle_max_deg = rigid_angle_max_deg_from_sigma(
                sigma=sigma,
                rigid_rot_deg=params.rigid_rot_deg,
                rigid_rot_deg_min=params.rigid_rot_deg_min,
            )
            angle_max = np.radians(angle_max_deg)
            angle = rng.uniform(-angle_max, angle_max)

            ax, ay, az = axis
            K = np.array([
                [0, -az, ay],
                [az, 0, -ax],
                [-ay, ax, 0],
            ])
            I = np.eye(3)
            R = I + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
            V_out = (V - center) @ R.T + center

        if mode in ("rigid", "translation"):
            trans_std = rigid_trans_axis_std_from_sigma(
                sigma=sigma,
                rigid_trans_scale=params.rigid_trans_scale,
                rigid_trans_scale_min=params.rigid_trans_scale_min,
            )
            trans = rng.normal(0.0, trans_std, size=(1, 3))
            V_out = V_out + trans

        return V_out

    if mode == "outliers":
        n_verts = V.shape[0]
        mask = rng.random(n_verts) < params.outlier_frac
        if params.outlier_frac > 0.0 and not mask.any():
            idx = rng.integers(0, n_verts)
            mask[idx] = True
        displacement = params.outlier_scale * sigma * rng.normal(0.0, 1.0, size=V.shape)
        return V + displacement * mask[:, None]

    raise ValueError(f"Unknown perturbation mode: {mode}")


# ── geometry pipeline using facebench ────────────────────────────────────────

def symmetric_chamfer(X: np.ndarray, Y: np.ndarray) -> float:
    """Symmetric Chamfer distance using facebench correspondence + p2p."""
    corr_xy = fb.chamfer_correspondence(X, Y)
    corr_yx = fb.chamfer_correspondence(Y, X)
    d_xy = float(np.mean(fb.p2p_distance(X, Y, corr_xy)))
    d_yx = float(np.mean(fb.p2p_distance(Y, X, corr_yx)))
    return 0.5 * (d_xy + d_yx)


def run_geometry_pipeline(
    V_a: np.ndarray,
    V_b: np.ndarray,
    stages: List[str],
    max_sample_points: int,
    seed: int,
) -> Dict[str, float]:
    """Run facebench pipeline on one pair of vertex arrays (already perturbed)."""
    result: Dict[str, float] = {
        "raw_chamfer": math.nan,
        "rigid_p2p": math.nan,
        "nicp_p2p": math.nan,
        "nicp_p2tri": math.nan,
        "nicp_direct_p2p": math.nan,
        "nicp_direct_p2tri": math.nan,
        "nicp_bbox_p2p": math.nan,
        "nicp_bbox_p2tri": math.nan,
        "rigid_seconds": math.nan,
        "nicp_seconds": math.nan,
        "nicp_direct_seconds": math.nan,
        "nicp_bbox_seconds": math.nan,
    }
    try:
        X = V_a
        Y = V_b

        if "raw" in stages:
            Xs = sample_pts(X, max_sample_points, seed)
            Ys = sample_pts(Y, max_sample_points, seed + 1)
            result["raw_chamfer"] = symmetric_chamfer(Xs, Ys)

        if "rigid" in stages:
            t0 = time.time()
            Xs = sample_pts(X, max_sample_points, seed)
            Ys = sample_pts(Y, max_sample_points, seed + 1)
            X_rigid, _ = fb.icp_align(Xs, Ys, prealign="bbox")
            result["rigid_seconds"] = float(time.time() - t0)
            corr_r = fb.chamfer_correspondence(X_rigid, Ys)
            result["rigid_p2p"] = float(np.mean(
                fb.p2p_distance(X_rigid, Ys, corr_r)
            ))

        if "nicp" in stages:
            Xs = sample_pts(X, max_sample_points, seed)
            Ys = sample_pts(Y, max_sample_points, seed + 1)
            X_rigid, _ = fb.icp_align(Xs, Ys, prealign="bbox")
            t0 = time.time()
            X_nicp = fb.nonrigid_icp_align(X_rigid, Ys)
            result["nicp_seconds"] = float(time.time() - t0)
            corr_n = fb.chamfer_correspondence(X_nicp, Ys)
            result["nicp_p2p"] = float(np.mean(fb.p2p_distance(X_nicp, Ys, corr_n)))
            if "p2tri" in stages or "nicp" in stages:
                result["nicp_p2tri"] = float(np.mean(fb.p2tri_distance(X_nicp, Ys, corr_n)))

        if "nicp_direct" in stages:
            Xs = sample_pts(X, max_sample_points, seed)
            Ys = sample_pts(Y, max_sample_points, seed + 1)
            t0 = time.time()
            X_nicp_direct = fb.nonrigid_icp_align(Xs, Ys)
            result["nicp_direct_seconds"] = float(time.time() - t0)
            corr_nd = fb.chamfer_correspondence(X_nicp_direct, Ys)
            result["nicp_direct_p2p"] = float(np.mean(fb.p2p_distance(Xs, Ys, corr_nd)))
            if "p2tri" in stages or "nicp_direct" in stages:
                result["nicp_direct_p2tri"] = float(np.mean(fb.p2tri_distance(Xs, Ys, corr_nd)))

        if "nicp_bbox" in stages:
            Xs = sample_pts(X, max_sample_points, seed)
            Ys = sample_pts(Y, max_sample_points, seed + 1)
            X_bbox = prealign_by_bbox(Xs, Ys)
            t0 = time.time()
            X_nicp_bbox = fb.nonrigid_icp_align(X_bbox, Ys)
            result["nicp_bbox_seconds"] = float(time.time() - t0)
            corr_nb = fb.chamfer_correspondence(X_nicp_bbox, Ys)
            result["nicp_bbox_p2p"] = float(np.mean(fb.p2p_distance(X_bbox, Ys, corr_nb)))
            if "p2tri" in stages or "nicp_bbox" in stages:
                result["nicp_bbox_p2tri"] = float(np.mean(fb.p2tri_distance(X_bbox, Ys, corr_nb)))

    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


# ── latent embedding ─────────────────────────────────────────────────────────

def load_model(checkpoint: Path, model_config: Path, device: str):
    """Build the DiffusionNet encoder and load weights. Returns (model, torch_device).
    Args mirror run_facebench_remesh.py so the two scripts load the same network."""
    import torch
    from face_embedding.gt_encdec.remeshing.intrinsic.robustness.model_helpers import (
        build_model,
    )

    with open(model_config) as f:
        cfg = json.load(f)

    class _Args:
        pass
    args = _Args()
    args.model = cfg.get("model", "xyz_dn")
    args.latent_dim = cfg.get("latent_dim", 256)
    args.width = cfg.get("width", 128)
    args.n_blocks = cfg.get("n_blocks", 4)
    args.dropout = cfg.get("dropout", 0.1)
    args.pooling = cfg.get("pooling", "meanmax")
    args.k_eig = cfg.get("k_eig", 300)
    args.hks_k = cfg.get("hks_k", 16)
    args.wks_k = cfg.get("wks_k", 16)
    args.k_spec = cfg.get("k_spec", 100)
    args.log_spec = cfg.get("log_spec", False)
    args.eps = cfg.get("eps", 1e-8)
    args.pool_mode = cfg.get("pool_mode", "meanmax")
    args.xyz_feature_dropout = cfg.get("xyz_feature_dropout", 0.0)
    args.fusion = cfg.get("fusion", "linear")
    args.fusion_layers = cfg.get("fusion_layers", 1)

    dev = torch.device(device)
    model = build_model(args, dev)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    state = (
        ckpt.get("model_state_dict")
        or ckpt.get("state_dict")
        or ckpt
    )
    model.load_state_dict(state)
    model.eval()
    return model, dev


def _load_sample_dict(npz_path: Path, device) -> Optional[Tuple]:
    """Load (verts_tensor, sample_dict) from a withops NPZ, or None if missing/bad."""
    return load_withops_sample_npz(npz_path, device)


def embed_meshes(
    withops_root: Path,
    sample_names: List[str],
    model,
    device,
) -> Dict[str, np.ndarray]:
    """Embed CLEAN meshes. Returns {sample_name: latent_vector}."""
    import torch
    from face_embedding.gt_encdec.remeshing.intrinsic.robustness.model_helpers import (
        forward_model,
    )

    latents: Dict[str, np.ndarray] = {}
    missing = 0
    for name in sample_names:
        loaded = _load_sample_dict(withops_root / f"{name}.npz", device)
        if loaded is None:
            missing += 1
            continue
        verts, sample_dict = loaded
        try:
            with torch.no_grad():
                z, _ = forward_model(
                    model, sample_dict, verts,
                    return_gate_info=False, add_noise=False,
                )
            latents[name] = z.cpu().numpy().squeeze()
        except Exception as exc:
            print(f"  [warn] embed failed for {name}: {exc}", flush=True)

    if missing:
        print(f"  [warn] {missing} withops NPZ not found; skipped.", flush=True)
    print(f"  Embedded {len(latents)} meshes (clean).", flush=True)
    return latents


def embed_meshes_perturbed(
    withops_root: Path,
    sample_names: List[str],
    model,
    device,
    mode: str,
    sigma: float,
    params: "PerturbationParams",
    scenario_idx: int,
) -> Dict[str, np.ndarray]:
    """Embed each mesh AFTER applying the same XYZ perturbation used at training
    (torch-based, operators left untouched — matches robustness.eval_utils recipe).
    Seed per (mesh, scenario_idx) so the same mesh always sees the same noise
    within a scenario, and so FG workers can reproduce it via `perturb_seed`."""
    import torch
    from face_embedding.gt_encdec.remeshing.intrinsic.robustness.model_helpers import (
        forward_model,
    )
    from face_embedding.gt_encdec.remeshing.intrinsic.robustness.noise import (
        apply_xyz_perturbation,
    )

    latents: Dict[str, np.ndarray] = {}
    for name in sample_names:
        loaded = _load_sample_dict(withops_root / f"{name}.npz", device)
        if loaded is None:
            continue
        verts, sample_dict = loaded
        try:
            seed = perturb_seed(name, scenario_idx)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            V_in = apply_xyz_perturbation(
                V=verts,
                mode=mode,
                sigma=float(sigma),
                outlier_frac=params.outlier_frac,
                outlier_scale=params.outlier_scale,
                rigid_rot_deg=params.rigid_rot_deg,
                rigid_trans_scale=params.rigid_trans_scale,
                rigid_rot_deg_min=params.rigid_rot_deg_min,
                rigid_trans_scale_min=params.rigid_trans_scale_min,
            )
            with torch.no_grad():
                z, _ = forward_model(
                    model, sample_dict, V_in,
                    return_gate_info=False, add_noise=False,
                )
            latents[name] = z.cpu().numpy().squeeze()
        except Exception as exc:
            print(f"  [warn] re-embed failed for {name}: {exc}", flush=True)

    return latents


def _load_coo_from_npz(data, key: str, device):
    """Reconstruct a torch sparse COO tensor from _indices/_values/_shape triplet."""
    import torch
    indices = torch.tensor(data[f"{key}_indices"], dtype=torch.long)
    values = torch.tensor(data[f"{key}_values"], dtype=torch.float32)
    shape = tuple(int(x) for x in data[f"{key}_shape"])
    return torch.sparse_coo_tensor(indices, values, torch.Size(shape)).to(device)


# ── GT distance matrix ───────────────────────────────────────────────────────

def load_gt_matrix(gt_matrix_path: Path) -> Tuple[Dict[str, int], np.ndarray]:
    data = np.load(gt_matrix_path, allow_pickle=True)
    matrix_key = "D_orig" if "D_orig" in data.files else data.files[0]
    matrix = data[matrix_key]
    names = [str(n) for n in data["names"]] if "names" in data.files else []
    name_to_idx: Dict[str, int] = {}
    for i, n in enumerate(names):
        name_to_idx[n] = i
        name_to_idx[n.split("_GTready")[0]] = i
    return name_to_idx, matrix


def gt_dist(subject_a: str, subject_b: str, name_to_idx: Dict[str, int], matrix: np.ndarray) -> float:
    i = name_to_idx.get(subject_a)
    j = name_to_idx.get(subject_b)
    if i is None or j is None:
        return math.nan
    return float(matrix[i, j])


# ── pair generation ──────────────────────────────────────────────────────────

def build_pairs(
    subject_ids: List[str],
    topo_a: str,
    topo_b: str,
    npz_root: Path,
    latents: Dict[str, np.ndarray],
    gt_name_to_idx: Dict[str, int],
    gt_matrix: np.ndarray,
) -> List[PairRecord]:
    records: List[PairRecord] = []
    valid_subjects = [
        s for s in subject_ids
        if f"{s}_GTready_{topo_a}" in latents
        and f"{s}_GTready_{topo_b}" in latents
        and (npz_root / f"{s}_GTready_{topo_a}.npz").exists()
        and (npz_root / f"{s}_GTready_{topo_b}.npz").exists()
    ]
    for i, sa in enumerate(valid_subjects):
        for sb in valid_subjects[i + 1:]:
            name_a = f"{sa}_GTready_{topo_a}"
            name_b = f"{sb}_GTready_{topo_b}"
            z_a = latents[name_a]
            z_b = latents[name_b]
            lat_dist = float(np.linalg.norm(z_a - z_b))
            gt_d = gt_dist(sa, sb, gt_name_to_idx, gt_matrix)
            records.append(PairRecord(
                subject_a=sa, topology_a=topo_a,
                sample_name_a=str(npz_root / f"{name_a}.npz"),
                subject_b=sb, topology_b=topo_b,
                sample_name_b=str(npz_root / f"{name_b}.npz"),
                gt_distance=gt_d,
                latent_distance=lat_dist,
            ))
    return records


# ── statistics ───────────────────────────────────────────────────────────────

def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return math.nan
    r, _ = spearmanr(x[mask], y[mask])
    return float(r)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return math.nan
    r, _ = pearsonr(x[mask], y[mask])
    return float(r)


def summarize(records: List[PairRecord], metric_cols: List[str]) -> Dict[str, object]:
    gt = np.array([r.gt_distance for r in records], dtype=np.float64)
    row: Dict[str, object] = {"n_pairs": len(records)}
    for col in metric_cols:
        vals = np.array([getattr(r, col, math.nan) for r in records], dtype=np.float64)
        row[f"spearman_{col}"] = safe_spearman(gt, vals)
        row[f"pearson_{col}"] = safe_pearson(gt, vals)
    return row


# ── CSV I/O ──────────────────────────────────────────────────────────────────

def write_pair_csv(path: Path, records: List[PairRecord]) -> None:
    fld = [f.name for f in fields(PairRecord)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fld)
        w.writeheader()
        for r in records:
            row = {f.name: getattr(r, f.name) for f in fields(PairRecord)}
            for side in ("sample_name_a", "sample_name_b"):
                row[side] = Path(row[side]).name.replace(".npz", "")
            w.writerow(row)


def write_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


# ── perturbed worker ─────────────────────────────────────────────────────────

def _perturbed_worker(args_tuple) -> List[PairRecord]:
    """
    Worker: for each PairRecord apply the (mode, sigma) perturbation to BOTH
    meshes — using a deterministic per-(mesh, scenario) seed so the same mesh
    gets the same noise realisation in every pair it appears in — then run the
    FG pipeline. Latent distance is set externally from re-embedded latents;
    this worker leaves it as passed in.
    """
    records, mode, sigma, npz_root, params_dict, stages, max_pts, scenario_idx = args_tuple

    params = PerturbationParams(
        outlier_frac=params_dict["outlier_frac"],
        outlier_scale=params_dict["outlier_scale"],
        rigid_rot_deg=params_dict["rigid_rot_deg"],
        rigid_trans_scale=params_dict["rigid_trans_scale"],
        rigid_rot_deg_min=params_dict["rigid_rot_deg_min"],
        rigid_trans_scale_min=params_dict["rigid_trans_scale_min"],
    )

    out_records = []
    for rec in records:
        new_rec = PairRecord(
            subject_a=rec.subject_a,
            topology_a=rec.topology_a,
            sample_name_a=rec.sample_name_a,
            subject_b=rec.subject_b,
            topology_b=rec.topology_b,
            sample_name_b=rec.sample_name_b,
            gt_distance=rec.gt_distance,
            latent_distance=rec.latent_distance,
            raw_chamfer=math.nan,
            rigid_p2p=math.nan,
            nicp_p2p=math.nan,
            nicp_p2tri=math.nan,
            nicp_direct_p2p=math.nan,
            nicp_direct_p2tri=math.nan,
            nicp_bbox_p2p=math.nan,
            nicp_bbox_p2tri=math.nan,
        )
        try:
            V_a = load_verts(Path(rec.sample_name_a))
            V_b = load_verts(Path(rec.sample_name_b))

            seed_a = perturb_seed(rec.sample_name_a, scenario_idx)
            seed_b = perturb_seed(rec.sample_name_b, scenario_idx)

            V_a_pert = perturb_verts_np(V_a, mode, sigma, params, seed_a)
            V_b_pert = perturb_verts_np(V_b, mode, sigma, params, seed_b)

            geo = run_geometry_pipeline(V_a_pert, V_b_pert, stages, max_pts, seed_a)
            new_rec.raw_chamfer = geo.get("raw_chamfer", math.nan)
            new_rec.rigid_p2p = geo.get("rigid_p2p", math.nan)
            new_rec.nicp_p2p = geo.get("nicp_p2p", math.nan)
            new_rec.nicp_p2tri = geo.get("nicp_p2tri", math.nan)
            new_rec.nicp_direct_p2p = geo.get("nicp_direct_p2p", math.nan)
            new_rec.nicp_direct_p2tri = geo.get("nicp_direct_p2tri", math.nan)
            new_rec.nicp_bbox_p2p = geo.get("nicp_bbox_p2p", math.nan)
            new_rec.nicp_bbox_p2tri = geo.get("nicp_bbox_p2tri", math.nan)

        except Exception as exc:
            new_rec.status = "failed"
            new_rec.error = f"{type(exc).__name__}: {exc}"

        out_records.append(new_rec)
    return out_records


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FaceBench pipeline vs latent UNDER PERTURBATIONS.")
    p.add_argument("--npz_root", required=True, help="Plain-geometry NPZ dir")
    p.add_argument("--withops_root", required=True, help="Operator-enriched NPZ dir")
    p.add_argument("--checkpoint", required=True, help="Model checkpoint .pth")
    p.add_argument("--model_config", required=True, help="Model config.json")
    p.add_argument("--gt_matrix", required=True, help="GT distance matrix .npz")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--max_subjects", type=int, default=100, help="Max subjects (0=all)")
    p.add_argument("--topo_pairs", default="all", help="'all' or 'original,remesh;crop,down8k'")
    p.add_argument("--stages", default="raw,rigid,nicp", help="FG stages: raw,rigid,nicp,nicp_direct,nicp_bbox")
    p.add_argument("--max_sample_points", type=int, default=4096, help="Max points per mesh for FG")
    p.add_argument("--workers", type=int, default=8, help="Parallel workers")
    p.add_argument("--device", default="cuda")
    # Perturbation args
    p.add_argument("--sigma_grid", default="", help="Comma-separated sigma values (default: dn_mixed_topology_v1 eval grid)")
    p.add_argument("--noise_modes", default="translation,rotation,jitter", help="Perturbation modes")
    p.add_argument("--rigid_rot_deg", type=float, default=12.0)
    p.add_argument("--rigid_rot_deg_min", type=float, default=0.5)
    p.add_argument("--rigid_trans_scale", type=float, default=0.03)
    p.add_argument("--rigid_trans_scale_min", type=float, default=0.001)
    p.add_argument("--outlier_frac", type=float, default=0.02)
    p.add_argument("--outlier_scale", type=float, default=6.0)
    return p.parse_args()


def _flush_summaries(out_dir: Path, all_summary_rows: List[Dict[str, object]], metric_cols: List[str]) -> None:
    """Write ranking_summary.csv + scenario_pivot.csv incrementally so partial
    runs still produce usable aggregates if the job is killed."""
    write_summary_csv(out_dir / "ranking_summary.csv", all_summary_rows)

    pivot_rows: List[Dict[str, object]] = []
    scenarios_seen: Dict[str, Dict[str, List[float]]] = {}
    for row in all_summary_rows:
        scenario = str(row.get("scenario", ""))
        bucket = scenarios_seen.setdefault(scenario, {"sigma": [row.get("sigma", 0.0)], "noise_mode": [row.get("noise_mode", "")], "n_pairs": []})
        bucket["n_pairs"].append(int(row.get("n_pairs", 0) or 0))
        for col in metric_cols:
            bucket.setdefault(f"sp_{col}", []).append(float(row.get(f"spearman_{col}", math.nan)))
            bucket.setdefault(f"pr_{col}", []).append(float(row.get(f"pearson_{col}", math.nan)))

    for scenario, bucket in scenarios_seen.items():
        pivot_row: Dict[str, object] = {
            "scenario": scenario,
            "sigma": bucket["sigma"][0],
            "noise_mode": bucket["noise_mode"][0],
        }
        for col in metric_cols:
            vals = np.array(bucket[f"sp_{col}"], dtype=np.float64)
            pivot_row[f"sp_{col}"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else math.nan
            vals_p = np.array(bucket[f"pr_{col}"], dtype=np.float64)
            pivot_row[f"pr_{col}"] = float(np.nanmean(vals_p)) if np.isfinite(vals_p).any() else math.nan
        pivot_row["n_pairs_total"] = int(np.sum(bucket["n_pairs"]))
        pivot_rows.append(pivot_row)

    write_summary_csv(out_dir / "scenario_pivot.csv", pivot_rows)


def _run_scenario(
    scenario_label: str,
    scenario_idx: int,
    mode: str,
    sigma: float,
    latents_scenario: Dict[str, np.ndarray],
    topo_pairs: List[Tuple[str, str]],
    all_subjects: List[str],
    npz_root: Path,
    gt_name_to_idx: Dict[str, int],
    gt_matrix: np.ndarray,
    params_dict: Dict[str, float],
    stages: List[str],
    max_sample_points: int,
    workers: int,
    pool,
    out_dir: Path,
    metric_cols: List[str],
) -> List[Dict[str, object]]:
    """Build pair records from scenario latents, run FG workers on perturbed
    geometry, write per-pair CSVs and return summary rows for the scenario.
    `pool` is a long-lived multiprocessing Pool reused across all scenarios
    to avoid paying the ~30s spawn tax on every topology pair."""
    scenario_rows: List[Dict[str, object]] = []
    print(f"  [_run_scenario] Starting {len(topo_pairs)} topology pairs...", flush=True)
    for topo_a, topo_b in topo_pairs:
        pair_label = f"{topo_a}__to__{topo_b}"
        print(f"  [_run_scenario] Building pairs for {pair_label}...", flush=True)
        records = build_pairs(
            subject_ids=all_subjects,
            topo_a=topo_a,
            topo_b=topo_b,
            npz_root=npz_root,
            latents=latents_scenario,
            gt_name_to_idx=gt_name_to_idx,
            gt_matrix=gt_matrix,
        )
        print(f"  [_run_scenario] Built {len(records)} pairs for {pair_label}", flush=True)
        if not records:
            continue

        n_workers = max(1, int(workers))
        chunk_size = max(1, math.ceil(len(records) / (n_workers * 4)))
        chunks: List[List[PairRecord]] = [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]
        worker_args = [
            (chunk, mode, float(sigma), npz_root, params_dict, stages, max_sample_points, scenario_idx)
            for chunk in chunks
        ]
        print(f"  [_run_scenario] Running {len(worker_args)} worker chunks for {pair_label}...", flush=True)

        if pool is not None:
            results = pool.map(_perturbed_worker, worker_args)
        else:
            results = [_perturbed_worker(wa) for wa in worker_args]

        pert_records = [r for sublist in results for r in sublist]

        pair_dir = out_dir / pair_label / scenario_label
        pair_dir.mkdir(parents=True, exist_ok=True)
        write_pair_csv(pair_dir / "pair_metrics.csv", pert_records)

        summary = summarize(pert_records, metric_cols)
        summary["scenario"] = scenario_label
        summary["sigma"] = float(sigma)
        summary["noise_mode"] = mode
        summary["topology_a"] = topo_a
        summary["topology_b"] = topo_b
        summary["pair_label"] = pair_label
        scenario_rows.append(summary)

    return scenario_rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_root = Path(args.npz_root)
    withops_root = Path(args.withops_root)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    metric_cols = [
        "latent_distance",
        "raw_chamfer",
        "rigid_p2p",
        "nicp_p2p",
        "nicp_p2tri",
        "nicp_direct_p2p",
        "nicp_direct_p2tri",
        "nicp_bbox_p2p",
        "nicp_bbox_p2tri",
    ]

    # ── sigma grid ────────────────────────────────────────────────────────────
    if args.sigma_grid.strip():
        sigma_grid = [float(x.strip()) for x in args.sigma_grid.split(",") if x.strip()]
    else:
        sigma_grid = list(np.logspace(np.log10(0.001), np.log10(0.1), 6))

    noise_modes = [m.strip() for m in args.noise_modes.split(",") if m.strip()]

    params = PerturbationParams(
        outlier_frac=args.outlier_frac,
        outlier_scale=args.outlier_scale,
        rigid_rot_deg=args.rigid_rot_deg,
        rigid_trans_scale=args.rigid_trans_scale,
        rigid_rot_deg_min=args.rigid_rot_deg_min,
        rigid_trans_scale_min=args.rigid_trans_scale_min,
    )
    params_dict = {
        "rigid_rot_deg": args.rigid_rot_deg,
        "rigid_rot_deg_min": args.rigid_rot_deg_min,
        "rigid_trans_scale": args.rigid_trans_scale,
        "rigid_trans_scale_min": args.rigid_trans_scale_min,
        "outlier_frac": args.outlier_frac,
        "outlier_scale": args.outlier_scale,
    }

    # ── topology pairs ────────────────────────────────────────────────────────
    if args.topo_pairs == "all":
        topo_pairs = ALL_TOPO_PAIRS
    else:
        topo_pairs = [
            tuple(s.strip().split(",")) for s in args.topo_pairs.split(";") if s.strip()
        ]

    # ── subject list ──────────────────────────────────────────────────────────
    all_subjects = sorted({
        p.stem.split("_GTready_")[0]
        for p in npz_root.glob("*_GTready_original.npz")
    })
    if args.max_subjects > 0:
        all_subjects = all_subjects[:args.max_subjects]
    print(f"Subjects: {len(all_subjects)}", flush=True)
    print(f"Topology pairs: {len(topo_pairs)}", flush=True)
    print(f"Sigma grid ({len(sigma_grid)} levels): {[f'{s:.4f}' for s in sigma_grid]}", flush=True)
    print(f"Noise modes: {noise_modes}", flush=True)
    print(f"Total scenarios: {len(sigma_grid) * len(noise_modes)} + clean", flush=True)

    all_sample_names = [
        f"{s}_GTready_{t}"
        for s in all_subjects
        for t in TOPOLOGIES
    ]

    # ── load model + GT once ──────────────────────────────────────────────────
    print("\nLoading model + GT matrix...", flush=True)
    model, dev = load_model(Path(args.checkpoint), Path(args.model_config), args.device)
    gt_name_to_idx, gt_matrix = load_gt_matrix(Path(args.gt_matrix))

    all_summary_rows: List[Dict[str, object]] = []

    # ── Long-lived worker pool ────────────────────────────────────────────────
    # Spawn takes ~2-3s per worker; reusing one Pool for the whole run saves
    # hours vs. creating a Pool per (scenario, topo_pair).
    pool = None
    if args.workers > 1:
        import multiprocessing as _mp
        ctx = _mp.get_context("spawn")
        pool = ctx.Pool(processes=args.workers)
        print(f"Spawned worker pool ({args.workers} procs)", flush=True)

    try:
        # ── CLEAN baseline ────────────────────────────────────────────────────
        print("\n=== CLEAN baseline ===", flush=True)
        print("Embedding meshes (clean)...", flush=True)
        latents_clean = embed_meshes(withops_root=withops_root, sample_names=all_sample_names, model=model, device=dev)

        clean_rows = _run_scenario(
            scenario_label="clean",
            scenario_idx=0,
            mode="jitter",
            sigma=0.0,
            latents_scenario=latents_clean,
            topo_pairs=topo_pairs,
            all_subjects=all_subjects,
            npz_root=npz_root,
            gt_name_to_idx=gt_name_to_idx,
            gt_matrix=gt_matrix,
            params_dict=params_dict,
            stages=stages,
            max_sample_points=args.max_sample_points,
            workers=args.workers,
            pool=pool,
            out_dir=out_dir,
            metric_cols=metric_cols,
        )
        for row in clean_rows:
            row["scenario"] = "clean"
            row["sigma"] = 0.0
            row["noise_mode"] = "none"
        all_summary_rows.extend(clean_rows)
        _flush_summaries(out_dir, all_summary_rows, metric_cols)

        # ── Perturbation sweep ────────────────────────────────────────────────
        total_scenarios = len(sigma_grid) * len(noise_modes)
        scenario_idx = 0
        t_start = time.time()
        for sigma in sigma_grid:
            for mode in noise_modes:
                scenario_idx += 1
                scenario_label = f"{mode}_sigma{sigma:.4f}"
                print(f"\n[{scenario_idx}/{total_scenarios}] {scenario_label}", flush=True)

                t_embed = time.time()
                latents_scenario = embed_meshes_perturbed(
                    withops_root=withops_root,
                    sample_names=all_sample_names,
                    model=model,
                    device=dev,
                    mode=mode,
                    sigma=float(sigma),
                    params=params,
                    scenario_idx=scenario_idx,
                )
                print(f"  re-embedded {len(latents_scenario)} meshes in {time.time()-t_embed:.1f}s", flush=True)

                t_fg = time.time()
                scenario_rows = _run_scenario(
                    scenario_label=scenario_label,
                    scenario_idx=scenario_idx,
                    mode=mode,
                    sigma=float(sigma),
                    latents_scenario=latents_scenario,
                    topo_pairs=topo_pairs,
                    all_subjects=all_subjects,
                    npz_root=npz_root,
                    gt_name_to_idx=gt_name_to_idx,
                    gt_matrix=gt_matrix,
                    params_dict=params_dict,
                    stages=stages,
                    max_sample_points=args.max_sample_points,
                    workers=args.workers,
                    pool=pool,
                    out_dir=out_dir,
                    metric_cols=metric_cols,
                )
                print(f"  FG pipeline in {time.time()-t_fg:.1f}s", flush=True)
                all_summary_rows.extend(scenario_rows)
                _flush_summaries(out_dir, all_summary_rows, metric_cols)

                elapsed = time.time() - t_start
                eta = elapsed * (total_scenarios - scenario_idx) / max(scenario_idx, 1)
                print(f"  elapsed {elapsed/60:.1f}min; eta {eta/60:.1f}min", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    print("\n" + "=" * 100)
    print(f"Done. Outputs under {out_dir}")
    print("=" * 100)


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "  —  "
    return f"{float(v):+.3f}"


if __name__ == "__main__":
    main()
