#!/usr/bin/env python3
"""
run_facebench_full_pipeline_perturbed.py

Full FaceBench pipeline (large_conf_test.py recipe) vs latent embedding,
under the same perturbation sweep used elsewhere in latentVSpipeline/.

Pipeline steps replicated per pair (via facebench.utils.pipeline.run_pipeline):
  1. rigid ICP with LANDMARK prealign
  2. non-rigid ICP (NICP)
  3. chamfer correspondence
  4. topology_consistency_corrector (default cfg: PAIR + SQRT + MIXED)
  5. P2P distance

The corrector needs the BFM template (lmk_indices, eye indices, mean_face_shape),
so pairs are restricted to topologies that share the BFM vertex count (23470):
that's `original` and `noisy` in REMESH. Other topologies (remesh/down8k/up60k/crop)
have different vertex counts and no per-mesh landmarks — they are skipped.

Outputs mirror run_facebench_remesh_perturbed.py:
  - ranking_summary.csv + scenario_pivot.csv (flushed each scenario)
  - <topo_a>__to__<topo_b>/<scenario>/pair_metrics.csv

Usage (from repo root):
  .venv_twotower_robust_312/bin/python faceBench/latentVSpipeline/run_facebench_full_pipeline_perturbed.py \
    --npz_root datasets/REMESH/npz_data_topo_500 \
    --withops_root datasets/REMESH/npz_data_topo_500_withops \
    --checkpoint face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1/mixed_xtopo_rank0p5_id0p25_bs5_best/checkpoints/best_by_xtopo_mesh_clean.pth \
    --model_config face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1/mixed_xtopo_rank0p5_id0p25_bs5_best/config.json \
    --gt_matrix face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz \
    --mm_json faceBench/info/BFM-p23470.json \
    --out_dir faceBench/latentVSpipeline/outputs/full_pipeline_subset \
    --max_subjects 30 \
    --sigma_grid 0.001,0.01,0.1 \
    --workers 12
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr, pearsonr

REPO_ROOT = Path(__file__).resolve().parents[2]
FACBENCH_DIR = REPO_ROOT / "faceBench"
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "diffusion-net" / "src"))
sys.path.insert(0, str(FACBENCH_DIR))
sys.path.insert(0, str(THIS_DIR))

import facebench as fb
from facebench.utils.pipeline import run_pipeline as fb_run_pipeline

from face_embedding.gt_encdec.remeshing.intrinsic.robustness.noise import (
    PerturbationParams,
)

# Reuse helpers from the sibling script — same perturbation semantics, same seeds.
from run_facebench_remesh_perturbed import (  # type: ignore
    perturb_seed,
    perturb_verts_np,
    load_verts,
    load_model,
    embed_meshes,
    embed_meshes_perturbed,
    load_gt_matrix,
    gt_dist,
    safe_spearman,
    safe_pearson,
    write_summary_csv,
)

BFM_VERT_COUNT = 23470
BFM_COMPATIBLE_TOPOS = {"original", "noisy"}  # both have 23470 verts


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
    full_pipeline_p2p: float = math.nan
    pipeline_seconds: float = math.nan
    status: str = "ok"
    error: str = ""


def _normalize_mm(mm: dict) -> dict:
    """Ensure arrays (some JSON keys arrive as lists)."""
    out = dict(mm)
    out["mean_face_shape"] = np.asarray(out["mean_face_shape"], dtype=np.float64)
    out["lmk_indices"] = np.asarray(out["lmk_indices"], dtype=np.int64)
    out["leye_oc_rel_index"] = int(out["leye_oc_rel_index"])
    out["reye_oc_rel_index"] = int(out["reye_oc_rel_index"])
    return out


def _build_pipeline_config(mm: dict) -> fb.PipelineConfig:
    """Match large_conf_test.py:
      rigid ICP + LANDMARK prealign
      non-rigid ELASTIC alignment (landmark-driven, stable and cheap)
      CHAMFER correspondence
      topology_consistency corrector (defaults)
      P2P distance
    We swap ELASTIC for NICP if --use_nicp is passed (slower, denser).
    """
    return fb.PipelineConfig(
        rigid_aligner=fb.RigidAlignerConfig(
            type=fb.RigidAlignerType.ICP,
            prealign=fb.PrealignMethod.LANDMARK,
        ),
        nonrigid_aligner=fb.NonRigidAlignerConfig(
            type=fb.NonRigidAlignerType.ELASTIC,
            ref_lmk_indices=list(mm["lmk_indices"]),
        ),
        corr_establisher=fb.CorrEstablisherConfig(type=fb.CorrEstablisherType.CHAMFER),
        corrector=fb.CorrectorConfig(),
        distance_computer=fb.DistanceComputerConfig(type=fb.DistanceComputerType.P2P),
    )


def _build_pipeline_config_nicp(mm: dict) -> fb.PipelineConfig:
    return fb.PipelineConfig(
        rigid_aligner=fb.RigidAlignerConfig(
            type=fb.RigidAlignerType.ICP,
            prealign=fb.PrealignMethod.LANDMARK,
        ),
        nonrigid_aligner=fb.NonRigidAlignerConfig(
            type=fb.NonRigidAlignerType.NICP,
            ref_lmk_indices=list(mm["lmk_indices"]),
        ),
        corr_establisher=fb.CorrEstablisherConfig(type=fb.CorrEstablisherType.CHAMFER),
        corrector=fb.CorrectorConfig(),
        distance_computer=fb.DistanceComputerConfig(type=fb.DistanceComputerType.P2P),
    )


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
    valid = [
        s for s in subject_ids
        if f"{s}_GTready_{topo_a}" in latents
        and f"{s}_GTready_{topo_b}" in latents
        and (npz_root / f"{s}_GTready_{topo_a}.npz").exists()
        and (npz_root / f"{s}_GTready_{topo_b}.npz").exists()
    ]
    for i, sa in enumerate(valid):
        for sb in valid[i + 1:]:
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


# ── worker ──────────────────────────────────────────────────────────────────

def _worker_pair(args_tuple) -> List[PairRecord]:
    """Apply (mode, sigma) perturbation to both meshes, run fb.run_pipeline,
    take mean P2P error. Landmarks = V[mm['lmk_indices']] (assumes BFM topology)."""
    records, mode, sigma, params_dict, scenario_idx, mm_json_path, use_nicp = args_tuple

    # Rebuild mm + pipeline config once per worker chunk
    with open(mm_json_path) as f:
        mm = _normalize_mm(json.load(f))
    cfg = _build_pipeline_config_nicp(mm) if use_nicp else _build_pipeline_config(mm)
    lmk_idx = mm["lmk_indices"]

    params = PerturbationParams(
        outlier_frac=params_dict["outlier_frac"],
        outlier_scale=params_dict["outlier_scale"],
        rigid_rot_deg=params_dict["rigid_rot_deg"],
        rigid_trans_scale=params_dict["rigid_trans_scale"],
        rigid_rot_deg_min=params_dict["rigid_rot_deg_min"],
        rigid_trans_scale_min=params_dict["rigid_trans_scale_min"],
    )

    out: List[PairRecord] = []
    for rec in records:
        new_rec = PairRecord(
            subject_a=rec.subject_a, topology_a=rec.topology_a,
            sample_name_a=rec.sample_name_a,
            subject_b=rec.subject_b, topology_b=rec.topology_b,
            sample_name_b=rec.sample_name_b,
            gt_distance=rec.gt_distance,
            latent_distance=rec.latent_distance,
        )
        try:
            V_a = load_verts(Path(rec.sample_name_a))
            V_b = load_verts(Path(rec.sample_name_b))
            if V_a.shape[0] != BFM_VERT_COUNT or V_b.shape[0] != BFM_VERT_COUNT:
                new_rec.status = "skipped"
                new_rec.error = f"non-BFM topo: {V_a.shape[0]},{V_b.shape[0]}"
                out.append(new_rec)
                continue

            seed_a = perturb_seed(rec.sample_name_a, scenario_idx)
            seed_b = perturb_seed(rec.sample_name_b, scenario_idx)
            V_a_p = perturb_verts_np(V_a, mode, sigma, params, seed_a)
            V_b_p = perturb_verts_np(V_b, mode, sigma, params, seed_b)

            Rlmks = V_a_p[lmk_idx]
            Glmks = V_b_p[lmk_idx]

            t0 = time.time()
            error, _Rref = fb_run_pipeline(V_a_p, V_b_p, Rlmks, Glmks, mm, cfg)
            new_rec.pipeline_seconds = float(time.time() - t0)
            new_rec.full_pipeline_p2p = float(np.mean(np.asarray(error)))

        except Exception as exc:
            new_rec.status = "failed"
            new_rec.error = f"{type(exc).__name__}: {exc}"

        out.append(new_rec)
    return out


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


def summarize(records: List[PairRecord]) -> Dict[str, object]:
    gt = np.array([r.gt_distance for r in records], dtype=np.float64)
    row: Dict[str, object] = {"n_pairs": len(records)}
    for col in ("latent_distance", "full_pipeline_p2p"):
        vals = np.array([getattr(r, col, math.nan) for r in records], dtype=np.float64)
        row[f"spearman_{col}"] = safe_spearman(gt, vals)
        row[f"pearson_{col}"] = safe_pearson(gt, vals)
    return row


def _flush_summaries(out_dir: Path, rows: List[Dict[str, object]]) -> None:
    write_summary_csv(out_dir / "ranking_summary.csv", rows)
    # scenario pivot: mean Spearman across topo pairs per scenario
    by_scenario: Dict[str, Dict[str, List[float]]] = {}
    for r in rows:
        sc = str(r.get("scenario", ""))
        b = by_scenario.setdefault(sc, {"sigma": [r.get("sigma", 0.0)], "noise_mode": [r.get("noise_mode", "")]})
        for col in ("latent_distance", "full_pipeline_p2p"):
            b.setdefault(f"sp_{col}", []).append(float(r.get(f"spearman_{col}", math.nan)))
            b.setdefault(f"pr_{col}", []).append(float(r.get(f"pearson_{col}", math.nan)))
        b.setdefault("n_pairs", []).append(int(r.get("n_pairs", 0) or 0))
    pivot = []
    for sc, b in by_scenario.items():
        p = {"scenario": sc, "sigma": b["sigma"][0], "noise_mode": b["noise_mode"][0]}
        for col in ("latent_distance", "full_pipeline_p2p"):
            vs = np.array(b[f"sp_{col}"], dtype=np.float64)
            p[f"sp_{col}"] = float(np.nanmean(vs)) if np.isfinite(vs).any() else math.nan
            vp = np.array(b[f"pr_{col}"], dtype=np.float64)
            p[f"pr_{col}"] = float(np.nanmean(vp)) if np.isfinite(vp).any() else math.nan
        p["n_pairs_total"] = int(np.sum(b["n_pairs"]))
        pivot.append(p)
    write_summary_csv(out_dir / "scenario_pivot.csv", pivot)


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FaceBench full pipeline + latent under perturbations.")
    p.add_argument("--npz_root", required=True)
    p.add_argument("--withops_root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model_config", required=True)
    p.add_argument("--gt_matrix", required=True)
    p.add_argument("--mm_json", required=True, help="BFM-p23470.json")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_subjects", type=int, default=30)
    p.add_argument("--topo_pairs", default="original,noisy;noisy,original",
                   help="BFM-compatible pairs only (23470 verts)")
    p.add_argument("--sigma_grid", default="0.001,0.01,0.1")
    p.add_argument("--noise_modes", default="translation,rotation,jitter")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--device", default="cuda")
    p.add_argument("--use_nicp", action="store_true",
                   help="Swap ELASTIC for NICP (slower, denser deformation)")
    p.add_argument("--rigid_rot_deg", type=float, default=12.0)
    p.add_argument("--rigid_rot_deg_min", type=float, default=0.5)
    p.add_argument("--rigid_trans_scale", type=float, default=0.03)
    p.add_argument("--rigid_trans_scale_min", type=float, default=0.001)
    p.add_argument("--outlier_frac", type=float, default=0.02)
    p.add_argument("--outlier_scale", type=float, default=6.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_root = Path(args.npz_root)
    withops_root = Path(args.withops_root)

    # Validate requested topo pairs — must be subset of BFM_COMPATIBLE_TOPOS
    topo_pairs = [
        tuple(s.strip().split(",")) for s in args.topo_pairs.split(";") if s.strip()
    ]
    bad = [t for t in topo_pairs if t[0] not in BFM_COMPATIBLE_TOPOS or t[1] not in BFM_COMPATIBLE_TOPOS]
    if bad:
        print(f"[ERROR] non-BFM-compatible topo pairs: {bad}", flush=True)
        print(f"Only {sorted(BFM_COMPATIBLE_TOPOS)} have 23470 verts (BFM template).", flush=True)
        sys.exit(1)

    sigma_grid = [float(x.strip()) for x in args.sigma_grid.split(",") if x.strip()]
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

    all_subjects = sorted({
        p.stem.split("_GTready_")[0]
        for p in npz_root.glob("*_GTready_original.npz")
    })
    if args.max_subjects > 0:
        all_subjects = all_subjects[:args.max_subjects]

    topos_needed = sorted({t for tp in topo_pairs for t in tp})
    all_sample_names = [f"{s}_GTready_{t}" for s in all_subjects for t in topos_needed]

    print(f"Subjects: {len(all_subjects)}", flush=True)
    print(f"Topo pairs: {topo_pairs}", flush=True)
    print(f"Sigma grid: {sigma_grid}", flush=True)
    print(f"Modes: {noise_modes}", flush=True)
    print(f"Nonrigid: {'NICP' if args.use_nicp else 'ELASTIC'}", flush=True)

    print("\nLoading model + GT + mm...", flush=True)
    model, dev = load_model(Path(args.checkpoint), Path(args.model_config), args.device)
    gt_name_to_idx, gt_matrix = load_gt_matrix(Path(args.gt_matrix))

    # ── worker pool (single spawn for the whole run) ─────────────────────────
    pool = None
    if args.workers > 1:
        import multiprocessing as _mp
        ctx = _mp.get_context("spawn")
        pool = ctx.Pool(processes=args.workers)
        print(f"Spawned pool ({args.workers} procs)", flush=True)

    def run_scenario(scenario_label, scenario_idx, mode, sigma, latents):
        rows: List[Dict[str, object]] = []
        for topo_a, topo_b in topo_pairs:
            pair_label = f"{topo_a}__to__{topo_b}"
            recs = build_pairs(all_subjects, topo_a, topo_b, npz_root, latents, gt_name_to_idx, gt_matrix)
            if not recs:
                continue
            n_workers = max(1, int(args.workers))
            chunk_size = max(1, math.ceil(len(recs) / (n_workers * 4)))
            chunks = [recs[i:i + chunk_size] for i in range(0, len(recs), chunk_size)]
            wargs = [
                (c, mode, float(sigma), params_dict, scenario_idx, str(args.mm_json), args.use_nicp)
                for c in chunks
            ]
            if pool is not None:
                results = pool.map(_worker_pair, wargs)
            else:
                results = [_worker_pair(w) for w in wargs]
            pert = [r for sl in results for r in sl]

            pair_dir = out_dir / pair_label / scenario_label
            pair_dir.mkdir(parents=True, exist_ok=True)
            write_pair_csv(pair_dir / "pair_metrics.csv", pert)

            summary = summarize(pert)
            summary["scenario"] = scenario_label
            summary["sigma"] = float(sigma)
            summary["noise_mode"] = mode
            summary["topology_a"] = topo_a
            summary["topology_b"] = topo_b
            summary["pair_label"] = pair_label
            rows.append(summary)
        return rows

    all_rows: List[Dict[str, object]] = []

    try:
        # Clean
        print("\n=== CLEAN ===", flush=True)
        latents_clean = embed_meshes(withops_root=withops_root, sample_names=all_sample_names, model=model, device=dev)
        clean_rows = run_scenario("clean", 0, "jitter", 0.0, latents_clean)
        for r in clean_rows:
            r["scenario"] = "clean"
            r["sigma"] = 0.0
            r["noise_mode"] = "none"
        all_rows.extend(clean_rows)
        _flush_summaries(out_dir, all_rows)

        # Perturbation sweep
        total = len(sigma_grid) * len(noise_modes)
        sc = 0
        t_start = time.time()
        for sigma in sigma_grid:
            for mode in noise_modes:
                sc += 1
                label = f"{mode}_sigma{sigma:.4f}"
                print(f"\n[{sc}/{total}] {label}", flush=True)

                t0 = time.time()
                lat = embed_meshes_perturbed(
                    withops_root=withops_root, sample_names=all_sample_names,
                    model=model, device=dev,
                    mode=mode, sigma=float(sigma), params=params, scenario_idx=sc,
                )
                print(f"  re-embedded {len(lat)} meshes in {time.time()-t0:.1f}s", flush=True)

                t0 = time.time()
                rows = run_scenario(label, sc, mode, float(sigma), lat)
                print(f"  full pipeline in {time.time()-t0:.1f}s", flush=True)
                all_rows.extend(rows)
                _flush_summaries(out_dir, all_rows)

                elapsed = time.time() - t_start
                eta = elapsed * (total - sc) / max(sc, 1)
                print(f"  elapsed {elapsed/60:.1f}min; eta {eta/60:.1f}min", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    print(f"\nDone. Outputs under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
