#!/usr/bin/env python3
"""
Intrinsic diagnostics pipeline (4 steps):
1) HKS-only pre-network correlation vs GT distance matrix.
2) HKS distribution and inter/intra variance diagnostics.
3) PCA on latent embeddings from a trained DiffusionEncoderOnlyIntrinsec checkpoint.
4) Laplacian spectrum similarity diagnostics (lambda_1..lambda_K).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[4]
AUTOENCODER_DIR = REPO_ROOT / "face_embedding" / "gt_encdec" / "autoencoder"
if str(AUTOENCODER_DIR) not in sys.path:
    sys.path.append(str(AUTOENCODER_DIR))

from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset  # noqa: E402
from diffusion_autoencoder import DiffusionEncoderOnlyIntrinsec  # noqa: E402


SUBJECT_RE = re.compile(r"(id\d{4})", re.IGNORECASE)


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
    default_out = THIS_FILE.parent / "runs_hks_intrinsic_diagnostics"

    p = argparse.ArgumentParser(description="Run 4-step intrinsic diagnostics for HKS and latent embeddings.")
    p.add_argument("--data_dir", type=str, default=str(default_data_dir))
    p.add_argument("--dist_npz", type=str, default=str(default_dist))
    p.add_argument("--out_dir", type=str, default=str(default_out))

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max_subjects", type=int, default=0, help="0 = all overlapping GT subjects")
    p.add_argument("--max_meshes_per_subject", type=int, default=0, help="0 = all variants")

    p.add_argument("--n_hks", type=int, default=16)
    p.add_argument("--hks_k_high", type=int, default=50)
    p.add_argument("--hks_eps", type=float, default=1e-6)
    p.add_argument("--k_spec", type=int, default=50, help="Analyze lambda_1..lambda_k_spec")

    p.add_argument("--model_ckpt", type=str, default="", help="Optional checkpoint for Step 3 PCA")
    p.add_argument("--model_use_xyz", action="store_true")
    p.add_argument("--model_n_hks", type=int, default=-1, help="-1 => fallback to --n_hks")
    p.add_argument("--model_n_wks", type=int, default=0)
    p.add_argument("--model_latent_dim", type=int, default=256)
    p.add_argument("--model_width", type=int, default=128)
    p.add_argument("--model_n_blocks", type=int, default=4)
    p.add_argument("--model_dropout", type=float, default=0.1)
    p.add_argument("--model_hks_k_high", type=int, default=50)
    p.add_argument("--model_eps", type=float, default=1e-6)
    p.add_argument("--pca_components", type=int, default=3)

    return p.parse_args()


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


def choose_subjects(
    all_subjects: Sequence[str],
    gt_name_to_idx: Dict[str, int],
    max_subjects: int,
    seed: int,
) -> List[str]:
    overlap = sorted([s for s in all_subjects if s in gt_name_to_idx])
    if max_subjects <= 0 or len(overlap) <= max_subjects:
        return overlap
    rng = np.random.default_rng(seed)
    picked = rng.choice(np.asarray(overlap, dtype=object), size=max_subjects, replace=False)
    return sorted(picked.tolist())


def pick_mesh_indices(idxs: Sequence[int], max_meshes: int, seed: int) -> List[int]:
    if max_meshes <= 0 or len(idxs) <= max_meshes:
        return list(idxs)
    rng = np.random.default_rng(seed)
    picked = rng.choice(np.asarray(idxs), size=max_meshes, replace=False)
    return [int(i) for i in picked.tolist()]


def load_gt_distance_matrix(path: str) -> Tuple[np.ndarray, Dict[str, int]]:
    pack = np.load(path, allow_pickle=True)
    if "D_orig" not in pack or "names" not in pack:
        raise KeyError(f"{path} must contain D_orig and names. Found: {pack.files}")

    D = pack["D_orig"].astype(np.float64)
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


def to_device_tensor(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    if x.is_sparse:
        return x.coalesce().to(device)
    return x.to(device)


def load_ops_sample(sample: dict, device: torch.device) -> Tuple[torch.Tensor, ...]:
    keys = ("verts", "mass", "L", "evals", "evecs", "faces", "gradX", "gradY")
    vals = [to_device_tensor(sample[k], device) for k in keys]
    return tuple(vals)  # type: ignore[return-value]


def compute_hks(
    evals: torch.Tensor,
    evecs: torch.Tensor,
    n_hks: int,
    hks_k_high: int,
    eps: float,
) -> torch.Tensor:
    n_verts = evecs.shape[0]
    if n_hks <= 0:
        return torch.zeros(n_verts, 0, device=evecs.device, dtype=evecs.dtype)

    k = min(evals.numel(), evecs.shape[1])
    if k < 3:
        return torch.zeros(n_verts, n_hks, device=evecs.device, dtype=evecs.dtype)

    evals_k = evals[:k].clamp_min(eps)
    evecs_k = evecs[:, :k]

    lam_low = evals_k[1]
    lam_high = evals_k[min(k - 1, hks_k_high)]

    t_min = float((4.0 / lam_high).detach().cpu())
    t_max = float((4.0 / lam_low).detach().cpu())
    if not (np.isfinite(t_min) and np.isfinite(t_max) and t_min > 0 and t_max > t_min):
        t_min, t_max = 1e-2, 1e2

    t = torch.logspace(
        math.log10(t_min),
        math.log10(t_max),
        n_hks,
        device=evals.device,
        dtype=evals.dtype,
    )
    hks = (evecs_k ** 2) @ torch.exp(-evals_k[:, None] * t[None, :])
    return torch.log(hks + eps)


def mass_weighted_mean(x: torch.Tensor, mass: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    w = mass / (mass.sum() + eps)
    return (w[:, None] * x).sum(dim=0)


def take_or_pad_vec(x: torch.Tensor, k: int) -> torch.Tensor:
    x = x.flatten()
    if x.numel() >= k:
        return x[:k]
    pad = torch.zeros(k - x.numel(), device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=0)


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
    rx = rankdata_average_ties(np.asarray(x))
    ry = rankdata_average_ties(np.asarray(y))
    return pearson_corr(rx, ry)


def pairwise_l2_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    x2 = np.sum(X * X, axis=1, keepdims=True)
    d2 = np.maximum(x2 + x2.T - 2.0 * (X @ X.T), 0.0)
    return np.sqrt(d2)


def upper_triangular_values(M: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]


def collect_hks_and_spectrum(
    dataset: GTReadyDataset,
    subject_map: Dict[str, List[int]],
    subjects: Sequence[str],
    n_hks: int,
    hks_k_high: int,
    hks_eps: float,
    k_spec: int,
    max_meshes_per_subject: int,
    device: torch.device,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
    hks_by_subject: Dict[str, np.ndarray] = {}
    spec_by_subject: Dict[str, np.ndarray] = {}

    hks_min = float("inf")
    hks_max = float("-inf")
    total_mesh_used = 0
    total_mesh_failed = 0

    for s_idx, sid in enumerate(tqdm(subjects, desc="Collect HKS+Spectrum", dynamic_ncols=True)):
        mesh_idxs = pick_mesh_indices(
            idxs=subject_map[sid],
            max_meshes=max_meshes_per_subject,
            seed=seed + 1000 + s_idx,
        )

        hks_vecs: List[np.ndarray] = []
        spec_vecs: List[np.ndarray] = []

        for mesh_idx in mesh_idxs:
            try:
                sample = dataset[int(mesh_idx)]
                _, mass, _, evals, evecs, _, _, _ = load_ops_sample(sample, device)

                hks = compute_hks(
                    evals=evals,
                    evecs=evecs,
                    n_hks=n_hks,
                    hks_k_high=hks_k_high,
                    eps=hks_eps,
                )
                if hks.numel() == 0:
                    continue

                hks_min = min(hks_min, float(hks.min().item()))
                hks_max = max(hks_max, float(hks.max().item()))

                hks_vec = mass_weighted_mean(hks, mass).detach().cpu().numpy().astype(np.float64)
                spec_vec = take_or_pad_vec(evals[1:], k_spec).detach().cpu().numpy().astype(np.float64)

                hks_vecs.append(hks_vec)
                spec_vecs.append(spec_vec)
                total_mesh_used += 1
            except Exception:
                total_mesh_failed += 1

        if hks_vecs:
            hks_by_subject[sid] = np.stack(hks_vecs, axis=0)
        if spec_vecs:
            spec_by_subject[sid] = np.stack(spec_vecs, axis=0)

    misc = {
        "total_mesh_used": float(total_mesh_used),
        "total_mesh_failed": float(total_mesh_failed),
        "hks_global_min": hks_min if np.isfinite(hks_min) else float("nan"),
        "hks_global_max": hks_max if np.isfinite(hks_max) else float("nan"),
    }
    return hks_by_subject, spec_by_subject, misc


def step1_hks_vs_gt(
    hks_by_subject: Dict[str, np.ndarray],
    gt_matrix: np.ndarray,
    gt_name_to_idx: Dict[str, int],
) -> dict:
    subjects = sorted([s for s in hks_by_subject.keys() if s in gt_name_to_idx])
    if len(subjects) < 3:
        return {
            "n_subjects": len(subjects),
            "spearman": float("nan"),
            "pearson": float("nan"),
            "status": "insufficient_subjects",
        }

    X = np.stack([hks_by_subject[s].mean(axis=0) for s in subjects], axis=0)
    D_hks = pairwise_l2_matrix(X)

    gt_idx = np.array([gt_name_to_idx[s] for s in subjects], dtype=int)
    D_gt = gt_matrix[np.ix_(gt_idx, gt_idx)]

    hks_vals = upper_triangular_values(D_hks)
    gt_vals = upper_triangular_values(D_gt)

    return {
        "n_subjects": len(subjects),
        "spearman": spearman_corr(gt_vals, hks_vals),
        "pearson": pearson_corr(gt_vals, hks_vals),
        "status": "ok",
    }


def step2_hks_distribution(hks_by_subject: Dict[str, np.ndarray], hks_global_min: float, hks_global_max: float) -> dict:
    subjects = sorted(hks_by_subject.keys())
    if len(subjects) == 0:
        return {
            "n_subjects": 0,
            "hks_global_min": float("nan"),
            "hks_global_max": float("nan"),
            "mean_channel_variance_inter": float("nan"),
            "mean_channel_variance_intra": float("nan"),
            "inter_over_intra_ratio": float("nan"),
            "status": "no_data",
        }

    subj_means = np.stack([hks_by_subject[s].mean(axis=0) for s in subjects], axis=0)
    inter_var_by_channel = np.var(subj_means, axis=0, ddof=0)

    intra_list = []
    for s in subjects:
        Xs = hks_by_subject[s]
        if Xs.shape[0] < 2:
            continue
        intra_list.append(np.var(Xs, axis=0, ddof=0))

    if intra_list:
        intra_var_by_channel = np.mean(np.stack(intra_list, axis=0), axis=0)
    else:
        intra_var_by_channel = np.full(subj_means.shape[1], np.nan, dtype=np.float64)

    inter_m = float(np.nanmean(inter_var_by_channel))
    intra_m = float(np.nanmean(intra_var_by_channel))
    ratio = inter_m / max(intra_m, 1e-12) if np.isfinite(intra_m) else float("nan")

    return {
        "n_subjects": len(subjects),
        "hks_global_min": float(hks_global_min),
        "hks_global_max": float(hks_global_max),
        "mean_channel_variance_inter": inter_m,
        "mean_channel_variance_intra": intra_m,
        "inter_over_intra_ratio": float(ratio),
        "inter_var_by_channel": inter_var_by_channel.tolist(),
        "intra_var_by_channel": intra_var_by_channel.tolist(),
        "status": "ok",
    }


def pca_numpy(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    n = Xc.shape[0]
    if n > 1:
        var = (S ** 2) / (n - 1)
    else:
        var = S ** 2
    var_ratio = var / max(var.sum(), 1e-12)

    k = min(max(1, n_components), Vt.shape[0])
    comps = Vt[:k]
    coords = Xc @ comps.T
    return coords, comps, var_ratio


@torch.inference_mode()
def step3_latent_pca(
    model_ckpt: str,
    dataset: GTReadyDataset,
    subject_map: Dict[str, List[int]],
    subjects: Sequence[str],
    max_meshes_per_subject: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    if model_ckpt is None or model_ckpt.strip() == "":
        return {"status": "skipped", "reason": "no_checkpoint"}

    ckpt_path = Path(model_ckpt)
    if not ckpt_path.exists():
        return {"status": "skipped", "reason": f"checkpoint_not_found:{ckpt_path}"}

    model_n_hks = args.model_n_hks if args.model_n_hks >= 0 else args.n_hks

    model = DiffusionEncoderOnlyIntrinsec(
        latent_dim=args.model_latent_dim,
        width=args.model_width,
        n_blocks=args.model_n_blocks,
        dropout=args.model_dropout,
        use_xyz=args.model_use_xyz,
        n_hks=model_n_hks,
        n_wks=args.model_n_wks,
        hks_k_high=args.model_hks_k_high,
        eps=args.model_eps,
    ).to(device)

    payload = torch.load(str(ckpt_path), map_location=device)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload

    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        return {"status": "skipped", "reason": f"load_state_dict_failed:{e}"}

    model.eval()

    subj_embs: List[np.ndarray] = []
    subj_ids: List[str] = []

    for s_idx, sid in enumerate(tqdm(subjects, desc="Step3 latent extraction", dynamic_ncols=True)):
        mesh_idxs = pick_mesh_indices(
            idxs=subject_map[sid],
            max_meshes=max_meshes_per_subject,
            seed=args.seed + 9000 + s_idx,
        )
        z_list: List[np.ndarray] = []

        for idx in mesh_idxs:
            try:
                sample = dataset[int(idx)]
                V, mass, L, evals, evecs, faces, gradX, gradY = load_ops_sample(sample, device)
                zg = model(
                    V,
                    mass,
                    L,
                    evals,
                    evecs,
                    faces,
                    gradX,
                    gradY,
                    return_per_vertex=False,
                    add_noise=False,
                ).squeeze(0)
                z_list.append(zg.detach().cpu().numpy().astype(np.float64))
            except Exception:
                continue

        if not z_list:
            continue

        subj_embs.append(np.stack(z_list, axis=0).mean(axis=0))
        subj_ids.append(sid)

    if len(subj_embs) < 3:
        return {"status": "skipped", "reason": "insufficient_subjects_after_extraction"}

    X = np.stack(subj_embs, axis=0)
    coords, comps, var_ratio = pca_numpy(X, n_components=args.pca_components)

    return {
        "status": "ok",
        "n_subjects": len(subj_ids),
        "subject_ids": subj_ids,
        "coords": coords,
        "components": comps,
        "explained_variance_ratio": var_ratio,
    }


def step4_spectrum_analysis(
    spec_by_subject: Dict[str, np.ndarray],
    gt_matrix: np.ndarray,
    gt_name_to_idx: Dict[str, int],
) -> dict:
    subjects = sorted(spec_by_subject.keys())
    if len(subjects) == 0:
        return {"status": "no_data"}

    S_mean = np.stack([spec_by_subject[s].mean(axis=0) for s in subjects], axis=0)

    lam_mean = np.mean(S_mean, axis=0)
    lam_std = np.std(S_mean, axis=0)
    lam_min = np.min(S_mean, axis=0)
    lam_max = np.max(S_mean, axis=0)
    lam_cv = lam_std / np.maximum(np.abs(lam_mean), 1e-12)

    intra_spec_list = []
    for s in subjects:
        Xm = spec_by_subject[s]
        if Xm.shape[0] < 2:
            continue
        intra_spec_list.append(np.var(Xm, axis=0, ddof=0))
    if intra_spec_list:
        intra_spec_var_channel = np.mean(np.stack(intra_spec_list, axis=0), axis=0)
    else:
        intra_spec_var_channel = np.full(S_mean.shape[1], np.nan, dtype=np.float64)

    overlap = [s for s in subjects if s in gt_name_to_idx]
    if len(overlap) >= 3:
        X = np.stack([spec_by_subject[s].mean(axis=0) for s in overlap], axis=0)
        D_spec = pairwise_l2_matrix(X)
        idx = np.array([gt_name_to_idx[s] for s in overlap], dtype=int)
        D_gt = gt_matrix[np.ix_(idx, idx)]
        spec_vals = upper_triangular_values(D_spec)
        gt_vals = upper_triangular_values(D_gt)
        spec_spearman = spearman_corr(gt_vals, spec_vals)
        spec_pearson = pearson_corr(gt_vals, spec_vals)
    else:
        spec_spearman = float("nan")
        spec_pearson = float("nan")

    return {
        "status": "ok",
        "n_subjects": len(subjects),
        "lambda_mean": lam_mean.tolist(),
        "lambda_std": lam_std.tolist(),
        "lambda_min": lam_min.tolist(),
        "lambda_max": lam_max.tolist(),
        "lambda_cv": lam_cv.tolist(),
        "lambda_cv_mean": float(np.nanmean(lam_cv)),
        "lambda_cv_median": float(np.nanmedian(lam_cv)),
        "spec_intra_var_channel": intra_spec_var_channel.tolist(),
        "spec_intra_var_mean": float(np.nanmean(intra_spec_var_channel)),
        "spec_vs_gt_spearman": spec_spearman,
        "spec_vs_gt_pearson": spec_pearson,
    }


def write_subject_hks_csv(path: Path, hks_by_subject: Dict[str, np.ndarray]) -> None:
    rows = []
    for sid in sorted(hks_by_subject.keys()):
        rows.append((sid, hks_by_subject[sid].mean(axis=0)))
    if not rows:
        return

    n_hks = rows[0][1].shape[0]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject_id"] + [f"hks_{i}" for i in range(n_hks)])
        for sid, vec in rows:
            w.writerow([sid] + [float(v) for v in vec])


def write_channel_stats_csv(path: Path, inter_var: Sequence[float], intra_var: Sequence[float]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["channel", "inter_var", "intra_var", "inter_over_intra"])
        for i, (iv, av) in enumerate(zip(inter_var, intra_var)):
            ratio = float(iv) / max(float(av), 1e-12) if np.isfinite(av) else float("nan")
            w.writerow([i, float(iv), float(av), ratio])


def write_lambda_stats_csv(path: Path, step4: dict) -> None:
    lam_mean = step4.get("lambda_mean", [])
    lam_std = step4.get("lambda_std", [])
    lam_min = step4.get("lambda_min", [])
    lam_max = step4.get("lambda_max", [])
    lam_cv = step4.get("lambda_cv", [])

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k", "lambda_mean", "lambda_std", "lambda_min", "lambda_max", "lambda_cv"])
        for i in range(len(lam_mean)):
            w.writerow([i + 1, lam_mean[i], lam_std[i], lam_min[i], lam_max[i], lam_cv[i]])


def write_latent_pca_csv(path: Path, step3: dict) -> None:
    if step3.get("status") != "ok":
        return
    subj_ids = step3["subject_ids"]
    coords = np.asarray(step3["coords"])

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject_id"] + [f"pc{i+1}" for i in range(coords.shape[1])])
        for sid, row in zip(subj_ids, coords):
            w.writerow([sid] + [float(x) for x in row])


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Data: {args.data_dir}")
    print(f"GT dist: {args.dist_npz}")

    dataset = GTReadyDataset(args.data_dir)
    subject_map = build_subject_map(dataset.files)
    if not subject_map:
        raise RuntimeError("No valid subject ids parsed from dataset files")

    gt_matrix, gt_name_to_idx = load_gt_distance_matrix(args.dist_npz)
    subjects = choose_subjects(
        all_subjects=sorted(subject_map.keys()),
        gt_name_to_idx=gt_name_to_idx,
        max_subjects=args.max_subjects,
        seed=args.seed,
    )
    if len(subjects) < 3:
        raise RuntimeError(f"Need at least 3 overlapping subjects, got {len(subjects)}")

    print(f"Subjects used: {len(subjects)}")

    hks_by_subject, spec_by_subject, misc = collect_hks_and_spectrum(
        dataset=dataset,
        subject_map=subject_map,
        subjects=subjects,
        n_hks=args.n_hks,
        hks_k_high=args.hks_k_high,
        hks_eps=args.hks_eps,
        k_spec=args.k_spec,
        max_meshes_per_subject=args.max_meshes_per_subject,
        device=device,
        seed=args.seed,
    )

    step1 = step1_hks_vs_gt(
        hks_by_subject=hks_by_subject,
        gt_matrix=gt_matrix,
        gt_name_to_idx=gt_name_to_idx,
    )

    step2 = step2_hks_distribution(
        hks_by_subject=hks_by_subject,
        hks_global_min=misc["hks_global_min"],
        hks_global_max=misc["hks_global_max"],
    )

    step3 = step3_latent_pca(
        model_ckpt=args.model_ckpt,
        dataset=dataset,
        subject_map=subject_map,
        subjects=subjects,
        max_meshes_per_subject=args.max_meshes_per_subject,
        args=args,
        device=device,
    )

    step4 = step4_spectrum_analysis(
        spec_by_subject=spec_by_subject,
        gt_matrix=gt_matrix,
        gt_name_to_idx=gt_name_to_idx,
    )

    write_subject_hks_csv(out_dir / "step1_subject_hks_mean.csv", hks_by_subject)

    if step2.get("status") == "ok":
        write_channel_stats_csv(
            out_dir / "step2_hks_channel_variances.csv",
            step2["inter_var_by_channel"],
            step2["intra_var_by_channel"],
        )

    if step4.get("status") == "ok":
        write_lambda_stats_csv(out_dir / "step4_lambda_stats.csv", step4)

    write_latent_pca_csv(out_dir / "step3_latent_pca_subjects.csv", step3)

    summary = {
        "config": vars(args),
        "device": str(device),
        "n_subjects_requested": len(subjects),
        "collection_misc": misc,
        "step1_hks_vs_gt": step1,
        "step2_hks_distribution": {k: v for k, v in step2.items() if not k.endswith("_by_channel")},
        "step3_latent_pca": {
            k: v
            for k, v in step3.items()
            if k
            in (
                "status",
                "reason",
                "n_subjects",
                "explained_variance_ratio",
            )
        },
        "step4_spectrum": {
            k: v
            for k, v in step4.items()
            if k
            in (
                "status",
                "n_subjects",
                "lambda_cv_mean",
                "lambda_cv_median",
                "spec_intra_var_mean",
                "spec_vs_gt_spearman",
                "spec_vs_gt_pearson",
            )
        },
        "outputs": {
            "step1_subject_hks_mean_csv": str(out_dir / "step1_subject_hks_mean.csv"),
            "step2_channel_variances_csv": str(out_dir / "step2_hks_channel_variances.csv"),
            "step3_latent_pca_csv": str(out_dir / "step3_latent_pca_subjects.csv"),
            "step4_lambda_stats_csv": str(out_dir / "step4_lambda_stats.csv"),
            "summary_json": str(out_dir / "diagnostics_summary.json"),
        },
    }

    summary = sanitize_for_json(summary)
    with open(out_dir / "diagnostics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== STEP 1 (HKS vs GT) ===")
    print(step1)
    print("\n=== STEP 2 (HKS distribution) ===")
    print({k: v for k, v in step2.items() if not k.endswith("_by_channel")})
    print("\n=== STEP 3 (Latent PCA) ===")
    if step3.get("status") == "ok":
        print(
            {
                "status": step3["status"],
                "n_subjects": step3["n_subjects"],
                "explained_variance_ratio": step3["explained_variance_ratio"],
            }
        )
    else:
        print(step3)
    print("\n=== STEP 4 (Spectrum analysis) ===")
    print(
        {
            k: v
            for k, v in step4.items()
            if k
            in (
                "status",
                "n_subjects",
                "lambda_cv_mean",
                "lambda_cv_median",
                "spec_intra_var_mean",
                "spec_vs_gt_spearman",
                "spec_vs_gt_pearson",
            )
        }
    )
    print(f"\nSaved report in: {out_dir}")


if __name__ == "__main__":
    main()
