#!/usr/bin/env python3
"""
Two-path intrinsic experiment report.

Path A (linear, interpretable):
  - Build fixed spectral embeddings (HKS pipeline)
  - Learn diagonal Mahalanobis metric on train subjects
  - Evaluate ranking on val subjects
  - Test multiple subject-level aggregations (mean/median/trimmed_mean)

Path B (non-linear):
  - Run DiffusionNet triplet-ranking training (existing script)
  - Parse train log and report best Spearman/Pearson/NN

Outputs:
  - report CSV with both paths
  - learned diagonal weights per aggregation
  - JSON summary
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent
REPO_ROOT = THIS_FILE.parents[4]
AUTOENCODER_DIR = REPO_ROOT / "face_embedding" / "gt_encdec" / "autoencoder"

for p in (str(THIS_DIR), str(AUTOENCODER_DIR)):
    if p not in sys.path:
        sys.path.append(p)

from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset  # noqa: E402
from sweep_intrinsic_spectral_configs import SweepConfig, build_mesh_embedding  # noqa: E402
from intrinsic_utils import (  # noqa: E402
    build_subject_map,
    load_gt_distance_matrix,
    nn_match_rate,
    pairwise_distance_matrix,
    pearson_corr,
    seed_everything,
    spearman_corr,
    split_subjects,
    upper_triangular_values,
)


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
    default_out = THIS_DIR / "runs_two_path_report"
    default_triplet_script = THIS_DIR / "train_intrinsic_triplet.py"
    default_diff_out = THIS_DIR / "runs_triplet_two_path"

    parser = argparse.ArgumentParser(description="Run both paths: linear metric learning + DiffusionNet triplet.")

    parser.add_argument("--data_dir", type=str, default=str(default_data_dir))
    parser.add_argument("--dist_npz", type=str, default=str(default_dist))
    parser.add_argument("--out_dir", type=str, default=str(default_out))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_subjects", type=int, default=300, help="0 = all")
    parser.add_argument("--val_fraction", type=float, default=0.2)

    # Fixed spectral embedding configuration for path A
    parser.add_argument("--n_hks", type=int, default=16)
    parser.add_argument("--proj_k", type=int, default=32)
    parser.add_argument("--hks_times_mode", type=str, default="autoscale", choices=("autoscale", "spectral"))
    parser.add_argument("--hks_k_high", type=int, default=50)
    parser.add_argument("--abs_coeffs", action="store_true", default=True)
    parser.add_argument("--no_abs_coeffs", action="store_false", dest="abs_coeffs")
    parser.add_argument("--fix_evec_sign", action="store_true", default=True)
    parser.add_argument("--no_fix_evec_sign", action="store_false", dest="fix_evec_sign")
    parser.add_argument("--standardize_features", action="store_true", default=True)
    parser.add_argument("--no_standardize_features", action="store_false", dest="standardize_features")
    parser.add_argument("--l2_normalize_embedding", action="store_true", default=True)
    parser.add_argument("--no_l2_normalize_embedding", action="store_false", dest="l2_normalize_embedding")
    parser.add_argument("--eps", type=float, default=1e-6)

    # Aggregation experiments
    parser.add_argument("--aggregators", type=str, default="mean,median,trimmed_mean")
    parser.add_argument("--trim_frac", type=float, default=0.25)

    # Linear diagonal metric learning
    parser.add_argument("--lin_epochs", type=int, default=500)
    parser.add_argument("--lin_lr", type=float, default=5e-2)
    parser.add_argument("--lin_weight_decay", type=float, default=1e-4)
    parser.add_argument("--lin_reg_lambda", type=float, default=1e-2)
    parser.add_argument("--lin_verbose_every", type=int, default=100)

    # DiffusionNet (path B)
    parser.add_argument("--skip_diffusion", action="store_true", help="If set, run only path A.")
    parser.add_argument("--python_exec", type=str, default=sys.executable)
    parser.add_argument("--diffusion_script", type=str, default=str(default_triplet_script))
    parser.add_argument("--diffusion_out_dir", type=str, default=str(default_diff_out))
    parser.add_argument("--diffusion_epochs", type=int, default=50)
    parser.add_argument("--diffusion_extra_args", type=str, default="")
    parser.add_argument(
        "--diffusion_log_csv",
        type=str,
        default="",
        help="Optional existing train_log.csv to parse instead of launching training.",
    )
    return parser.parse_args()


def build_subject_mesh_embeddings(
    dataset: GTReadyDataset,
    subject_map: Dict[str, List[int]],
    subjects: Sequence[str],
    cfg: SweepConfig,
    device: torch.device,
    hks_k_high: int,
    eps: float,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for sid in tqdm(subjects, desc="mesh->embed", ncols=100):
        z_list: List[torch.Tensor] = []
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
            z_list.append(z.detach())
        if z_list:
            out[sid] = torch.stack(z_list, dim=0)
    return out


def aggregate_subject_embeddings(
    subject_mesh_emb: Dict[str, torch.Tensor],
    subjects: Sequence[str],
    method: str,
    trim_frac: float,
) -> Tuple[torch.Tensor, List[str]]:
    agg_list: List[torch.Tensor] = []
    kept: List[str] = []

    for sid in subjects:
        if sid not in subject_mesh_emb:
            continue
        x = subject_mesh_emb[sid]  # [M,D]
        if method == "mean":
            z = x.mean(dim=0)
        elif method == "median":
            z = x.median(dim=0).values
        elif method == "trimmed_mean":
            m = x.shape[0]
            t = int(math.floor(trim_frac * m))
            if t == 0 or (2 * t >= m):
                z = x.mean(dim=0)
            else:
                xs = torch.sort(x, dim=0).values
                z = xs[t : m - t].mean(dim=0)
        else:
            raise ValueError(f"Unknown aggregator: {method}")

        agg_list.append(z)
        kept.append(sid)

    if not agg_list:
        return torch.empty(0), []
    return torch.stack(agg_list, dim=0), kept


def pairwise_weighted_l2(Z: torch.Tensor, w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    diff = Z[:, None, :] - Z[None, :, :]
    d2 = (diff * diff * w[None, None, :]).sum(dim=-1)
    return torch.sqrt(d2 + eps)


def offdiag_normalize(D: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    n = D.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=D.device)
    off = D.masked_select(~eye)
    scale = off.mean().clamp_min(eps)
    return D / scale


def learn_diag_metric(
    Z_train: torch.Tensor,
    D_gt_train: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    reg_lambda: float,
    verbose_every: int,
) -> torch.Tensor:
    d = Z_train.shape[1]
    alpha = torch.nn.Parameter(torch.zeros(d, device=Z_train.device, dtype=Z_train.dtype))
    opt = torch.optim.Adam([alpha], lr=lr, weight_decay=weight_decay)

    n = Z_train.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=Z_train.device)
    Dgt_n = offdiag_normalize(D_gt_train)

    for ep in range(1, epochs + 1):
        w = F.softplus(alpha) + 1e-8
        Dp = pairwise_weighted_l2(Z_train, w)
        Dp_n = offdiag_normalize(Dp)

        main_loss = F.smooth_l1_loss(Dp_n.masked_select(~eye), Dgt_n.masked_select(~eye))
        reg_loss = reg_lambda * ((w - 1.0) ** 2).mean()
        loss = main_loss + reg_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if verbose_every > 0 and (ep % verbose_every == 0 or ep == 1 or ep == epochs):
            print(
                f"    [lin-metric] ep={ep:04d} loss={loss.item():.6f} "
                f"main={main_loss.item():.6f} reg={reg_loss.item():.6f}"
            )

    return (F.softplus(alpha).detach() + 1e-8)


def compute_gt_submatrix(
    subj_ids: Sequence[str],
    gt_matrix: np.ndarray,
    gt_name_to_idx: Dict[str, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    idx = np.array([gt_name_to_idx[s] for s in subj_ids], dtype=int)
    return torch.tensor(gt_matrix[np.ix_(idx, idx)], device=device, dtype=dtype)


def compute_intra_mean(
    subject_mesh_emb: Dict[str, torch.Tensor],
    subj_ids: Sequence[str],
    centers: Dict[str, torch.Tensor],
    w: Optional[torch.Tensor],
) -> float:
    vals: List[float] = []
    for sid in subj_ids:
        if sid not in subject_mesh_emb or sid not in centers:
            continue
        X = subject_mesh_emb[sid]
        c = centers[sid]
        if X.shape[0] <= 1:
            vals.append(0.0)
            continue
        diff = X - c[None, :]
        if w is None:
            d = torch.sqrt((diff * diff).sum(dim=1) + 1e-8)
        else:
            d = torch.sqrt((diff * diff * w[None, :]).sum(dim=1) + 1e-8)
        vals.append(float(d.mean().item()))
    return float(np.mean(vals)) if vals else float("nan")


def evaluate_distance_matrix(D_gt: torch.Tensor, D_emb: torch.Tensor) -> Tuple[float, float, float, float]:
    gt_vals = upper_triangular_values(D_gt).detach().cpu().numpy()
    em_vals = upper_triangular_values(D_emb).detach().cpu().numpy()
    spearman = spearman_corr(gt_vals, em_vals)
    pearson = pearson_corr(gt_vals, em_vals)
    nn = nn_match_rate(D_gt, D_emb)
    inter = float(upper_triangular_values(D_emb).mean().item()) if D_emb.shape[0] > 1 else float("nan")
    return spearman, pearson, nn, inter


def parse_diffusion_log(log_csv: Path) -> Optional[dict]:
    if not log_csv.exists():
        return None

    best = None
    with open(log_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sp = float(row.get("spearman", "nan"))
            except Exception:
                continue
            if np.isnan(sp):
                continue
            if (best is None) or (sp > best["spearman"]):
                best = {
                    "epoch": int(float(row.get("epoch", "0"))),
                    "spearman": sp,
                    "pearson": float(row.get("pearson", "nan")),
                    "nn_match_rate": float(row.get("nn_match", "nan")),
                    "intra_mean": float(row.get("intra_eval", "nan")),
                    "n_subjects_used": int(float(row.get("n_eval", "0"))),
                }
    return best


def maybe_run_diffusion(args: argparse.Namespace) -> Tuple[str, Optional[dict]]:
    if args.diffusion_log_csv:
        log_path = Path(args.diffusion_log_csv)
        parsed = parse_diffusion_log(log_path)
        if parsed is None:
            return f"log_not_found_or_invalid:{log_path}", None
        return "parsed_existing_log", parsed

    if args.skip_diffusion:
        return "skipped", None

    script = Path(args.diffusion_script)
    if not script.exists():
        return f"diffusion_script_not_found:{script}", None

    cmd = [
        args.python_exec,
        str(script),
        "--data_dir",
        args.data_dir,
        "--dist_npz",
        args.dist_npz,
        "--out_dir",
        args.diffusion_out_dir,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--max_subjects",
        str(args.max_subjects),
        "--epochs",
        str(args.diffusion_epochs),
        "--feature_preset",
        "hks_wks",
        "--pooling",
        "mean",
    ]
    if args.diffusion_extra_args.strip():
        cmd.extend(shlex.split(args.diffusion_extra_args.strip()))

    print("[path-B] launching diffusion triplet training:")
    print(" ", " ".join(shlex.quote(x) for x in cmd))
    try:
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    except subprocess.CalledProcessError as e:
        return f"launch_failed:{e.returncode}", None

    log_path = Path(args.diffusion_out_dir) / "train_log.csv"
    parsed = parse_diffusion_log(log_path)
    if parsed is None:
        return f"training_done_but_log_invalid:{log_path}", None
    return "trained_and_parsed", parsed


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")

    dataset = GTReadyDataset(args.data_dir)
    subject_map = build_subject_map(dataset.files)
    gt_matrix, gt_name_to_idx = load_gt_distance_matrix(args.dist_npz)

    subjects = sorted([s for s in subject_map.keys() if s in gt_name_to_idx])
    if args.max_subjects > 0 and len(subjects) > args.max_subjects:
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(np.array(subjects, dtype=object), size=args.max_subjects, replace=False)
        subjects = sorted(pick.tolist())

    train_subj, val_subj = split_subjects(subjects, args.val_fraction, args.seed)
    print(f"Subjects: total={len(subjects)} train={len(train_subj)} val={len(val_subj)}")

    # Fixed embedding config for path A
    cfg = SweepConfig(
        descriptor="hks",
        n_hks=args.n_hks,
        n_gps=0,
        hks_times_mode=args.hks_times_mode,
        proj_k=args.proj_k,
        embed_type="flatten",
        distance_mode="l2",
        abs_coeffs=args.abs_coeffs,
        fix_evec_sign=args.fix_evec_sign,
        standardize_features=args.standardize_features,
        l2_normalize_embedding=args.l2_normalize_embedding,
        svd_k=16,
    )

    mesh_emb = build_subject_mesh_embeddings(
        dataset=dataset,
        subject_map=subject_map,
        subjects=subjects,
        cfg=cfg,
        device=device,
        hks_k_high=args.hks_k_high,
        eps=args.eps,
    )

    aggregators = [a.strip() for a in args.aggregators.split(",") if a.strip()]
    rows: List[dict] = []

    for agg in aggregators:
        print(f"\n[path-A] aggregator={agg}")
        Z_train, kept_train = aggregate_subject_embeddings(mesh_emb, train_subj, agg, args.trim_frac)
        Z_val, kept_val = aggregate_subject_embeddings(mesh_emb, val_subj, agg, args.trim_frac)

        if Z_train.numel() == 0 or Z_val.numel() == 0 or len(kept_train) < 3 or len(kept_val) < 3:
            rows.append(
                {
                    "path": "A_linear",
                    "variant": "insufficient_data",
                    "aggregator": agg,
                    "spearman": float("nan"),
                    "pearson": float("nan"),
                    "nn_match_rate": float("nan"),
                    "inter_mean": float("nan"),
                    "intra_mean": float("nan"),
                    "inter_intra_ratio": float("nan"),
                    "n_train": int(len(kept_train)),
                    "n_val": int(len(kept_val)),
                }
            )
            continue

        Dgt_train = compute_gt_submatrix(kept_train, gt_matrix, gt_name_to_idx, device=Z_train.device, dtype=Z_train.dtype)
        Dgt_val = compute_gt_submatrix(kept_val, gt_matrix, gt_name_to_idx, device=Z_val.device, dtype=Z_val.dtype)

        centers_val = {sid: Z_val[i] for i, sid in enumerate(kept_val)}
        centers_train = {sid: Z_train[i] for i, sid in enumerate(kept_train)}

        # Baseline (unweighted L2)
        Demb_val_base = pairwise_distance_matrix(Z_val, mode="l2")
        sp_b, pr_b, nn_b, inter_b = evaluate_distance_matrix(Dgt_val, Demb_val_base)
        intra_b = compute_intra_mean(mesh_emb, kept_val, centers_val, w=None)
        rows.append(
            {
                "path": "A_linear",
                "variant": "baseline_l2",
                "aggregator": agg,
                "spearman": sp_b,
                "pearson": pr_b,
                "nn_match_rate": nn_b,
                "inter_mean": inter_b,
                "intra_mean": intra_b,
                "inter_intra_ratio": float(inter_b / (intra_b + 1e-12)),
                "n_train": int(len(kept_train)),
                "n_val": int(len(kept_val)),
            }
        )

        # Learn diagonal weights on train
        w = learn_diag_metric(
            Z_train=Z_train,
            D_gt_train=Dgt_train,
            epochs=args.lin_epochs,
            lr=args.lin_lr,
            weight_decay=args.lin_weight_decay,
            reg_lambda=args.lin_reg_lambda,
            verbose_every=args.lin_verbose_every,
        )
        np.save(Path(args.out_dir) / f"diag_weights_{agg}.npy", w.detach().cpu().numpy())

        Demb_val_w = pairwise_weighted_l2(Z_val, w)
        sp_w, pr_w, nn_w, inter_w = evaluate_distance_matrix(Dgt_val, Demb_val_w)
        intra_w = compute_intra_mean(mesh_emb, kept_val, centers_val, w=w)
        rows.append(
            {
                "path": "A_linear",
                "variant": "diag_metric",
                "aggregator": agg,
                "spearman": sp_w,
                "pearson": pr_w,
                "nn_match_rate": nn_w,
                "inter_mean": inter_w,
                "intra_mean": intra_w,
                "inter_intra_ratio": float(inter_w / (intra_w + 1e-12)),
                "n_train": int(len(kept_train)),
                "n_val": int(len(kept_val)),
            }
        )

        # Train-side diagnostics for learned metric
        Demb_train_w = pairwise_weighted_l2(Z_train, w)
        sp_tr, pr_tr, nn_tr, inter_tr = evaluate_distance_matrix(Dgt_train, Demb_train_w)
        intra_tr = compute_intra_mean(mesh_emb, kept_train, centers_train, w=w)
        rows.append(
            {
                "path": "A_linear",
                "variant": "diag_metric_train",
                "aggregator": agg,
                "spearman": sp_tr,
                "pearson": pr_tr,
                "nn_match_rate": nn_tr,
                "inter_mean": inter_tr,
                "intra_mean": intra_tr,
                "inter_intra_ratio": float(inter_tr / (intra_tr + 1e-12)),
                "n_train": int(len(kept_train)),
                "n_val": int(len(kept_val)),
            }
        )

    # Path B
    status_b, parsed_b = maybe_run_diffusion(args)
    if parsed_b is not None:
        rows.append(
            {
                "path": "B_diffusionnet",
                "variant": f"triplet_best_epoch_{parsed_b['epoch']}",
                "aggregator": "n/a",
                "spearman": float(parsed_b.get("spearman", float("nan"))),
                "pearson": float(parsed_b.get("pearson", float("nan"))),
                "nn_match_rate": float(parsed_b.get("nn_match_rate", float("nan"))),
                "inter_mean": float("nan"),
                "intra_mean": float(parsed_b.get("intra_mean", float("nan"))),
                "inter_intra_ratio": float("nan"),
                "n_train": int(len(train_subj)),
                "n_val": int(parsed_b.get("n_subjects_used", 0)),
            }
        )
    else:
        rows.append(
            {
                "path": "B_diffusionnet",
                "variant": f"no_result({status_b})",
                "aggregator": "n/a",
                "spearman": float("nan"),
                "pearson": float("nan"),
                "nn_match_rate": float("nan"),
                "inter_mean": float("nan"),
                "intra_mean": float("nan"),
                "inter_intra_ratio": float("nan"),
                "n_train": int(len(train_subj)),
                "n_val": int(len(val_subj)),
            }
        )

    # Save report
    out_dir = Path(args.out_dir)
    report_csv = out_dir / "two_path_report.csv"
    fields = [
        "path",
        "variant",
        "aggregator",
        "spearman",
        "pearson",
        "nn_match_rate",
        "inter_mean",
        "intra_mean",
        "inter_intra_ratio",
        "n_train",
        "n_val",
    ]
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # sort by spearman desc
    rows_sorted = sorted(
        rows,
        key=lambda r: (float("-inf") if np.isnan(r["spearman"]) else float(r["spearman"])),
        reverse=True,
    )
    with open(out_dir / "two_path_report_sorted.json", "w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, indent=2)

    with open(out_dir / "two_path_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2+

    print("")
    print(f"path-B status: {status_b}")
    print(f"report csv: {report_csv}")
    print(f"sorted json: {out_dir / 'two_path_report_sorted.json'}")


if __name__ == "__main__":
    t_start = time.time()
    main()
    print(f"done in {time.time() - t_start:.1f}s")
