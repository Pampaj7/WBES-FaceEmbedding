#!/usr/bin/env python3
"""
run_facebench_remesh.py

Full FaceBench geometry pipeline vs latent embedding on REMESH cross-topology pairs.
Mirrors large_conf_test.py but for REMESH .npz files and our latent model checkpoint.

Pipeline stages (all use the real facebench library, not fg_metrics.py):
  raw     – symmetric Chamfer distance (no alignment)
  rigid   – ICP (bbox prealign) + P2P mean distance
  nicp    – rigid ICP → non-rigid ICP → Chamfer correspondence → P2P mean
  p2tri   – same as nicp but P2Tri distance

Outputs per topology pair and overall:
  - pair_metrics.csv       : one row per mesh pair with all metrics + latent + gt distance
  - ranking_summary.csv    : Spearman/Pearson vs GT per metric per topology pair
  - overall_summary.csv    : aggregated across all topology pairs

Usage (from repo root, wbes-twotower-robust env):
  python faceBench/latentVSpipeline/run_facebench_remesh.py \\
    --npz_root datasets/REMESH/npz_data_topo_500 \\
    --withops_root datasets/REMESH/npz_data_topo_500_withops \\
    --checkpoint face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1/mixed_xtopo_rank0p5_id0p25_bs5_best/checkpoints/best_by_xtopo_mesh_clean.pth \\
    --model_config face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1/mixed_xtopo_rank0p5_id0p25_bs5_best/config.json \\
    --gt_matrix face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz \\
    --out_dir faceBench/latentVSpipeline/outputs/facebench_remesh_full \\
    --max_subjects 100 \\
    --stages raw,rigid,nicp \\
    --max_sample_points 4096 \\
    --workers 8
"""

from __future__ import annotations

import argparse
import csv
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

# ── repo root on path so we can import project modules ──────────────────────
# File lives at <repo>/faceBench/latentVSpipeline/run_facebench_remesh.py
REPO_ROOT = Path(__file__).resolve().parents[2]   # → <repo>/
FACBENCH_DIR = REPO_ROOT / "faceBench"            # → <repo>/faceBench/
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "diffusion-net" / "src"))
sys.path.insert(0, str(FACBENCH_DIR))             # makes `import facebench` work
sys.path.insert(0, str(THIS_DIR))

import facebench as fb
from mesh_npz_utils import load_normalized_vertices_npz, load_withops_sample_npz

TOPOLOGIES = ["crop", "down8k", "noisy", "original", "remesh", "up60k"]

# All ordered cross-topology pairs (A != B)
ALL_TOPO_PAIRS: List[Tuple[str, str]] = [
    (a, b) for a in TOPOLOGIES for b in TOPOLOGIES if a != b
]


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
    rigid_seconds: float = math.nan
    nicp_seconds: float = math.nan
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


# ── geometry pipeline using facebench ────────────────────────────────────────

def symmetric_chamfer(X: np.ndarray, Y: np.ndarray) -> float:
    """Symmetric Chamfer distance using facebench correspondence + p2p."""
    corr_xy = fb.chamfer_correspondence(X, Y)
    corr_yx = fb.chamfer_correspondence(Y, X)
    d_xy = float(np.mean(fb.p2p_distance(X, Y, corr_xy)))
    d_yx = float(np.mean(fb.p2p_distance(Y, X, corr_yx)))
    return 0.5 * (d_xy + d_yx)


def run_geometry_pipeline(
    path_a: str,
    path_b: str,
    stages: List[str],
    max_sample_points: int,
    seed: int,
) -> Dict[str, float]:
    """Run facebench pipeline on one pair. Returns dict of metric values."""
    result: Dict[str, float] = {
        "raw_chamfer": math.nan,
        "rigid_p2p": math.nan,
        "nicp_p2p": math.nan,
        "nicp_p2tri": math.nan,
        "rigid_seconds": math.nan,
        "nicp_seconds": math.nan,
    }
    try:
        X = load_verts(Path(path_a))
        Y = load_verts(Path(path_b))

        # 1. Raw symmetric Chamfer (sampled for speed)
        if "raw" in stages:
            Xs = sample_pts(X, max_sample_points, seed)
            Ys = sample_pts(Y, max_sample_points, seed + 1)
            result["raw_chamfer"] = symmetric_chamfer(Xs, Ys)

        # 2. Rigid ICP (bbox prealign) → P2P
        if "rigid" in stages or "nicp" in stages:
            t0 = time.time()
            X_rigid, _ = fb.icp_align(
                sample_pts(X, max_sample_points, seed),
                sample_pts(Y, max_sample_points, seed + 1),
                prealign="bbox",
            )
            # Apply the same transform estimated on subsamples to full X
            # (facebench icp_align works on the passed arrays; for the full mesh
            #  we use the sampled result as an approximation — sufficient for ranking)
            result["rigid_seconds"] = float(time.time() - t0)
            if "rigid" in stages:
                corr_r = fb.chamfer_correspondence(X_rigid, sample_pts(Y, max_sample_points, seed + 1))
                result["rigid_p2p"] = float(np.mean(
                    fb.p2p_distance(X_rigid, sample_pts(Y, max_sample_points, seed + 1), corr_r)
                ))

        # 3. Non-rigid ICP → P2P and P2Tri
        if "nicp" in stages:
            Xs = sample_pts(X, max_sample_points, seed)
            Ys = sample_pts(Y, max_sample_points, seed + 1)
            X_rigid_s, _ = fb.icp_align(Xs, Ys, prealign="bbox")
            t0 = time.time()
            X_nicp = fb.nonrigid_icp_align(X_rigid_s, Ys)
            result["nicp_seconds"] = float(time.time() - t0)
            corr_n = fb.chamfer_correspondence(X_nicp, Ys)
            result["nicp_p2p"] = float(np.mean(fb.p2p_distance(X_nicp, Ys, corr_n)))
            if "p2tri" in stages or "nicp" in stages:
                result["nicp_p2tri"] = float(np.mean(fb.p2tri_distance(X_nicp, Ys, corr_n)))

    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def _worker(args: tuple) -> PairRecord:
    """Multiprocessing worker: run geometry pipeline for one pair."""
    rec, stages, max_pts, seed = args
    metrics = run_geometry_pipeline(
        rec.sample_name_a, rec.sample_name_b, stages, max_pts, seed
    )
    rec.raw_chamfer = metrics.get("raw_chamfer", math.nan)
    rec.rigid_p2p = metrics.get("rigid_p2p", math.nan)
    rec.nicp_p2p = metrics.get("nicp_p2p", math.nan)
    rec.nicp_p2tri = metrics.get("nicp_p2tri", math.nan)
    rec.rigid_seconds = metrics.get("rigid_seconds", math.nan)
    rec.nicp_seconds = metrics.get("nicp_seconds", math.nan)
    if "status" in metrics:
        rec.status = metrics["status"]
        rec.error = metrics.get("error", "")
    return rec


# ── latent embedding ─────────────────────────────────────────────────────────

def embed_meshes(
    withops_root: Path,
    sample_names: List[str],
    checkpoint: Path,
    model_config: Path,
    device: str,
) -> Dict[str, np.ndarray]:
    """
    Embed all meshes with the latent model.
    Returns {sample_name: latent_vector}.
    """
    import torch
    from face_embedding.gt_encdec.remeshing.intrinsic.robustness.model_helpers import (
        build_model, forward_model,
    )

    with open(model_config) as f:
        cfg = json.load(f)

    # Build args namespace from config
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
    # checkpoint may wrap state_dict under various keys
    state = (
        ckpt.get("model_state_dict")
        or ckpt.get("state_dict")
        or ckpt
    )
    model.load_state_dict(state)
    model.eval()

    latents: Dict[str, np.ndarray] = {}
    missing = []

    for name in sample_names:
        npz_path = withops_root / f"{name}.npz"
        if not npz_path.exists():
            missing.append(name)
            continue
        try:
            loaded = load_withops_sample_npz(npz_path, dev)
            if loaded is None:
                missing.append(name)
                continue
            verts, sample_dict = loaded
            with torch.no_grad():
                z, _ = forward_model(
                    model, sample_dict, verts,
                    return_gate_info=False, add_noise=False,
                )
            latents[name] = z.cpu().numpy().squeeze()
        except Exception as exc:
            print(f"  [warn] embed failed for {name}: {exc}", flush=True)

    if missing:
        print(f"  [warn] {len(missing)} withops NPZ not found; skipped.", flush=True)
    print(f"  Embedded {len(latents)} meshes.", flush=True)
    return latents


def _load_coo_from_npz(data, key: str, device):
    """
    Reconstruct a torch sparse COO tensor from the _indices/_values/_shape triplet
    format used by precompute_operators_npz.py.
    e.g. L → L_indices (2×nnz), L_values (nnz,), L_shape (2,)
    """
    import torch
    indices = torch.tensor(data[f"{key}_indices"], dtype=torch.long)
    values = torch.tensor(data[f"{key}_values"], dtype=torch.float32)
    shape = tuple(int(x) for x in data[f"{key}_shape"])
    return torch.sparse_coo_tensor(indices, values, torch.Size(shape)).to(device)


# ── GT distance matrix ───────────────────────────────────────────────────────

def load_gt_matrix(gt_matrix_path: Path) -> Tuple[Dict[str, int], np.ndarray]:
    """Load GT distance matrix. Returns (subject_id_to_idx, matrix).

    The NPZ uses key 'D_orig' and names like 'id0000_GTready'.
    We index by the raw subject ID prefix (e.g. 'id0000') for easy lookup.
    """
    data = np.load(gt_matrix_path, allow_pickle=True)
    # find the distance matrix key
    matrix_key = "D_orig" if "D_orig" in data.files else data.files[0]
    matrix = data[matrix_key]
    names = [str(n) for n in data["names"]] if "names" in data.files else []
    # Build index: strip '_GTready' suffix so we can look up by subject id
    name_to_idx: Dict[str, int] = {}
    for i, n in enumerate(names):
        name_to_idx[n] = i                          # full name e.g. 'id0000_GTready'
        name_to_idx[n.split("_GTready")[0]] = i     # short id e.g. 'id0000'
    return name_to_idx, matrix


def gt_dist(subject_a: str, subject_b: str, name_to_idx: Dict[str, int], matrix: np.ndarray) -> float:
    """Look up GT distance between two subjects."""
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
    """All subject pairs for one (topo_a, topo_b) cross-topology combination."""
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
            # shorten absolute paths to relative
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


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FaceBench full pipeline vs latent on REMESH.")
    p.add_argument("--npz_root", required=True, help="Plain-geometry NPZ dir (key V)")
    p.add_argument("--withops_root", required=True, help="Operator-enriched NPZ dir")
    p.add_argument("--checkpoint", required=True, help="Model checkpoint .pth")
    p.add_argument("--model_config", required=True, help="Model config.json")
    p.add_argument("--gt_matrix", required=True, help="GT distance matrix .npz")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--max_subjects", type=int, default=100,
                   help="Max subjects to use (0=all 500)")
    p.add_argument("--topo_pairs", default="all",
                   help="'all' or comma-separated e.g. 'original,remesh;crop,down8k'")
    p.add_argument("--stages", default="raw,rigid,nicp",
                   help="Comma-separated stages: raw, rigid, nicp")
    p.add_argument("--max_sample_points", type=int, default=4096,
                   help="Max points per mesh for geometry pipeline (0=all)")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel workers for geometry pipeline")
    p.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    return p.parse_args()


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_root = Path(args.npz_root)
    withops_root = Path(args.withops_root)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    metric_cols = ["latent_distance", "raw_chamfer", "rigid_p2p", "nicp_p2p", "nicp_p2tri"]

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

    # ── embed all meshes ──────────────────────────────────────────────────────
    print("Phase 1: embedding meshes with latent model...", flush=True)
    all_sample_names = [
        f"{s}_GTready_{t}"
        for s in all_subjects
        for t in TOPOLOGIES
    ]
    latents = embed_meshes(
        withops_root=withops_root,
        sample_names=all_sample_names,
        checkpoint=Path(args.checkpoint),
        model_config=Path(args.model_config),
        device=args.device,
    )

    # ── GT matrix ────────────────────────────────────────────────────────────
    print("Loading GT distance matrix...", flush=True)
    gt_name_to_idx, gt_matrix = load_gt_matrix(Path(args.gt_matrix))

    # ── build pairs and run geometry pipeline ─────────────────────────────────
    print(f"Phase 2: geometry pipeline on {len(topo_pairs)} topology pairs...", flush=True)
    all_records: List[PairRecord] = []
    summary_rows: List[Dict[str, object]] = []

    for topo_a, topo_b in topo_pairs:
        pair_label = f"{topo_a}__to__{topo_b}"
        print(f"  [{pair_label}]", flush=True)

        records = build_pairs(
            subject_ids=all_subjects,
            topo_a=topo_a,
            topo_b=topo_b,
            npz_root=npz_root,
            latents=latents,
            gt_name_to_idx=gt_name_to_idx,
            gt_matrix=gt_matrix,
        )
        if not records:
            print(f"    no valid pairs, skipping", flush=True)
            continue
        print(f"    {len(records)} pairs", flush=True)

        # Parallel geometry pipeline
        # open3d (used by icp_align) is not fork-safe → use spawn context
        worker_args = [
            (r, stages, args.max_sample_points, i)
            for i, r in enumerate(records)
        ]
        if args.workers > 1:
            import multiprocessing as _mp
            ctx = _mp.get_context("spawn")
            with ctx.Pool(processes=args.workers) as pool:
                records = pool.map(_worker, worker_args)
        else:
            records = [_worker(wa) for wa in worker_args]

        # Per topology-pair summary
        summary = summarize(records, metric_cols)
        summary["topology_a"] = topo_a
        summary["topology_b"] = topo_b
        summary["pair_label"] = pair_label
        summary_rows.append(summary)
        all_records.extend(records)

        # Save per-pair CSV for this topology pair
        pair_dir = out_dir / pair_label
        pair_dir.mkdir(exist_ok=True)
        write_pair_csv(pair_dir / "pair_metrics.csv", records)

    # ── overall summary ───────────────────────────────────────────────────────
    print("Phase 3: writing summaries...", flush=True)
    overall = summarize(all_records, metric_cols)
    overall["topology_a"] = "all"
    overall["topology_b"] = "all"
    overall["pair_label"] = "overall"
    summary_rows.insert(0, overall)

    write_summary_csv(out_dir / "ranking_summary.csv", summary_rows)
    write_pair_csv(out_dir / "all_pairs.csv", all_records)

    # ── print results table (like generate_results_table) ────────────────────
    print("\n" + "=" * 70)
    print("RANKING SUMMARY (Spearman vs GT distance matrix)")
    print("=" * 70)
    header = f"{'Pair':<25} {'lat_sp':>8} {'raw_sp':>8} {'rig_sp':>8} {'nicp_sp':>9} {'p2tri_sp':>9} {'n':>7}"
    print(header)
    print("-" * 70)
    for row in summary_rows:
        label = row["pair_label"]
        print(
            f"{label:<25}"
            f" {_fmt(row.get('spearman_latent_distance')):>8}"
            f" {_fmt(row.get('spearman_raw_chamfer')):>8}"
            f" {_fmt(row.get('spearman_rigid_p2p')):>8}"
            f" {_fmt(row.get('spearman_nicp_p2p')):>9}"
            f" {_fmt(row.get('spearman_nicp_p2tri')):>9}"
            f" {row.get('n_pairs', ''):>7}"
        )
    print("=" * 70)
    print(f"\nOutputs written to {out_dir}")


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "  —  "
    return f"{float(v):+.3f}"


if __name__ == "__main__":
    main()
