#!/usr/bin/env python3
"""
Triplet-ranking training for intrinsic/global face embeddings with DiffusionNet.

Goal:
  Learn subject-level embeddings z that preserve the ranking induced by D_GT.

Core training objective:
  if D_GT(i,j) < D_GT(i,k), enforce ||z_i - z_j|| < ||z_i - z_k|| + margin

Examples:
  # HKS + WKS (default)
  python3 train_intrinsic_triplet.py --device cuda --max_subjects 300

  # xyz-only baseline
  python3 train_intrinsic_triplet.py --feature_preset xyz --device cuda --max_subjects 300

  # xyz + HKS baseline with attention pooling
  python3 train_intrinsic_triplet.py \
      --feature_preset xyz_hks --pooling attention --device cuda --max_subjects 300
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


# ------------------------------------------------------------
# Paths & imports
# ------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[4]
AUTOENCODER_DIR = REPO_ROOT / "face_embedding" / "gt_encdec" / "autoencoder"

for p in (
    str(AUTOENCODER_DIR),
    "/equilibrium/lpampaloni/diffusion-net/src",
    "/home/pampaj/diffusion-net/src",
    "/seidenas/users/lpampaloni/diffusion-net/src",
):
    if p not in sys.path:
        sys.path.append(p)

from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset  # noqa: E402

try:  # noqa: E402
    import diffusion_net

    DiffusionNet = diffusion_net.layers.DiffusionNet
except Exception:  # noqa: E402
    from diffusion_net import DiffusionNet


SUBJECT_RE = re.compile(r"(id\d{4})", re.IGNORECASE)


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
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
    default_out = THIS_FILE.parent / "runs_triplet"

    parser = argparse.ArgumentParser(
        description="Train intrinsic DiffusionNet embeddings with triplet ranking supervision."
    )
    parser.add_argument("--data_dir", type=str, default=str(default_data_dir))
    parser.add_argument("--dist_npz", type=str, default=str(default_dist))
    parser.add_argument("--out_dir", type=str, default=str(default_out))

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_subjects", type=int, default=0, help="0 = all")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--eval_subject_cap", type=int, default=96, help="0 = all eval subjects")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_subjects", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--l2_normalize", action="store_true")

    parser.add_argument(
        "--feature_preset",
        type=str,
        default="hks_wks",
        choices=("hks_wks", "hks", "xyz", "xyz_hks", "custom"),
    )
    parser.add_argument("--use_xyz", action="store_true", help="Only used for --feature_preset custom")
    parser.add_argument("--n_hks", type=int, default=16)
    parser.add_argument("--n_wks", type=int, default=16)
    parser.add_argument("--hks_k_high", type=int, default=50)
    parser.add_argument("--feature_eps", type=float, default=1e-6)

    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=("mean", "attention", "spectral"),
    )
    parser.add_argument("--spectral_k", type=int, default=16)

    parser.add_argument("--triplet_margin", type=float, default=0.2)
    parser.add_argument("--min_gt_gap", type=float, default=1e-5)
    parser.add_argument("--max_triplets_per_anchor", type=int, default=128, help="0 = all valid")
    parser.add_argument("--lambda_intra", type=float, default=0.05)

    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=5)
    return parser.parse_args()


def apply_feature_preset(args: argparse.Namespace) -> None:
    if args.feature_preset == "hks_wks":
        args.use_xyz = False
        args.n_hks = max(args.n_hks, 1)
        args.n_wks = max(args.n_wks, 1)
    elif args.feature_preset == "hks":
        args.use_xyz = False
        args.n_hks = max(args.n_hks, 1)
        args.n_wks = 0
    elif args.feature_preset == "xyz":
        args.use_xyz = True
        args.n_hks = 0
        args.n_wks = 0
    elif args.feature_preset == "xyz_hks":
        args.use_xyz = True
        args.n_hks = max(args.n_hks, 1)
        args.n_wks = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
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
        subj = extract_subject_id(fname)
        if subj is None:
            continue
        out.setdefault(subj, []).append(idx)
    return out


def split_subjects(
    subjects: Sequence[str],
    val_fraction: float,
    seed: int,
    max_subjects: int,
) -> Tuple[List[str], List[str]]:
    subjects_arr = np.array(sorted(subjects), dtype=object)
    rng = np.random.default_rng(seed)

    if max_subjects > 0 and len(subjects_arr) > max_subjects:
        pick = rng.choice(len(subjects_arr), size=max_subjects, replace=False)
        subjects_arr = subjects_arr[np.sort(pick)]

    if len(subjects_arr) < 6:
        raise ValueError(f"Need at least 6 subjects; found {len(subjects_arr)}")

    rng.shuffle(subjects_arr)
    n_eval = int(round(val_fraction * len(subjects_arr)))
    n_eval = max(3, n_eval)
    n_eval = min(n_eval, len(subjects_arr) - 3)

    eval_subjects = sorted(subjects_arr[:n_eval].tolist())
    train_subjects = sorted(subjects_arr[n_eval:].tolist())
    return train_subjects, eval_subjects


def load_gt_distance_matrix(path: str) -> Tuple[np.ndarray, Dict[str, int]]:
    pack = np.load(path, allow_pickle=True)
    if "D_orig" not in pack:
        raise KeyError(f"'D_orig' not found in {path}; available keys: {pack.files}")
    D = pack["D_orig"].astype(np.float32)

    # normalize to [0, 1] on off-diagonal support
    mask = D > 0
    if mask.any():
        D = D / float(D[mask].max())

    if "names" not in pack:
        raise KeyError(f"'names' not found in {path}; available keys: {pack.files}")

    names_raw = pack["names"]
    name_to_idx: Dict[str, int] = {}
    for i, name in enumerate(names_raw):
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="ignore")
        sid = extract_subject_id(str(name))
        if sid is not None:
            name_to_idx[sid] = i

    if not name_to_idx:
        raise RuntimeError(f"Failed to parse subject ids from {path}")
    return D, name_to_idx


def to_device_tensor(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    if x.is_sparse:
        return x.coalesce().to(device)
    return x.to(device)


def load_ops_sample(sample: dict, device: torch.device) -> Tuple[torch.Tensor, ...]:
    keys = ("verts", "mass", "L", "evals", "evecs", "faces", "gradX", "gradY")
    vals = [to_device_tensor(sample[k], device) for k in keys]
    return tuple(vals)  # type: ignore[return-value]


def upper_triangular_values(M: torch.Tensor) -> torch.Tensor:
    iu = torch.triu_indices(M.shape[0], M.shape[1], offset=1, device=M.device)
    return M[iu[0], iu[1]]


def rankdata_average_ties(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]

    ranks = np.empty(values.shape[0], dtype=np.float64)
    i = 0
    while i < sorted_vals.shape[0]:
        j = i + 1
        while j < sorted_vals.shape[0] and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata_average_ties(np.asarray(x))
    ry = rankdata_average_ties(np.asarray(y))
    return pearson_corr(rx, ry)


def nearest_neighbor_match_rate(D_gt: torch.Tensor, D_emb: torch.Tensor) -> float:
    n = D_gt.shape[0]
    if n < 2:
        return float("nan")
    eye = torch.eye(n, dtype=torch.bool, device=D_gt.device)
    gt_masked = D_gt.masked_fill(eye, float("inf"))
    emb_masked = D_emb.masked_fill(eye, float("inf"))
    nn_gt = gt_masked.argmin(dim=1)
    nn_emb = emb_masked.argmin(dim=1)
    return float((nn_gt == nn_emb).float().mean().item())


# ------------------------------------------------------------
# Features + model
# ------------------------------------------------------------
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
) -> torch.Tensor:
    n_verts = evecs.shape[0]
    if n_hks <= 0:
        return torch.zeros(n_verts, 0, dtype=evecs.dtype, device=evecs.device)

    k = min(evals.numel(), evecs.shape[1])
    if k < 3:
        return torch.zeros(n_verts, n_hks, dtype=evecs.dtype, device=evecs.device)

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


def compute_wks(
    evals: torch.Tensor,
    evecs: torch.Tensor,
    n_wks: int,
    eps: float,
) -> torch.Tensor:
    n_verts = evecs.shape[0]
    if n_wks <= 0:
        return torch.zeros(n_verts, 0, dtype=evecs.dtype, device=evecs.device)

    k = min(evals.numel(), evecs.shape[1])
    if k < 3:
        return torch.zeros(n_verts, n_wks, dtype=evecs.dtype, device=evecs.device)

    evals_k = evals[:k].clamp_min(eps)
    evecs_k = evecs[:, :k]
    log_ev = torch.log(evals_k)

    e1 = log_ev[1]
    eN = log_ev[-1]

    energies0 = torch.linspace(
        float(e1.detach().cpu()),
        float(eN.detach().cpu()),
        n_wks,
        device=evals.device,
        dtype=evals.dtype,
    )
    if n_wks > 1:
        delta = float((energies0[1] - energies0[0]).abs().detach().cpu())
    else:
        delta = float((eN - e1).abs().detach().cpu())
    sigma = 7.0 * max(delta, 1e-6)

    emin = e1 + 2.0 * sigma
    emax = eN - 2.0 * sigma
    if float((emax - emin).detach().cpu()) <= 0:
        emin, emax = e1, eN

    energies = torch.linspace(
        float(emin.detach().cpu()),
        float(emax.detach().cpu()),
        n_wks,
        device=evals.device,
        dtype=evals.dtype,
    )

    diff = log_ev[:, None] - energies[None, :]
    kernel = torch.exp(-(diff**2) / (2.0 * sigma**2 + 1e-12))
    kernel = kernel / (kernel.sum(dim=0, keepdim=True) + 1e-8)

    wks = (evecs_k ** 2) @ kernel
    return torch.log(wks + eps)


class DiffusionRankingEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        width: int,
        n_blocks: int,
        dropout: float,
        use_xyz: bool,
        n_hks: int,
        n_wks: int,
        pooling: str,
        spectral_k: int,
        hks_k_high: int,
        eps: float,
        noise_std: float,
    ) -> None:
        super().__init__()
        self.use_xyz = use_xyz
        self.n_hks = n_hks
        self.n_wks = n_wks
        self.pooling = pooling
        self.spectral_k = spectral_k
        self.hks_k_high = hks_k_high
        self.eps = eps
        self.noise_std = noise_std

        c_in = (3 if use_xyz else 0) + n_hks + n_wks
        if c_in <= 0:
            raise ValueError("Input feature channels are zero. Enable xyz and/or HKS/WKS.")

        self.encoder = DiffusionNet(
            C_in=c_in,
            C_out=latent_dim,
            C_width=width,
            N_block=n_blocks,
            with_gradient_features=True,
            dropout=0.0,
        )
        self.vertex_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, latent_dim),
        )

        if pooling == "attention":
            self.att_logits = nn.Linear(latent_dim, 1)
        elif pooling == "spectral":
            self.spectral_head = nn.Linear(spectral_k * latent_dim, latent_dim)

    def build_vertex_features(
        self,
        V: torch.Tensor,
        mass: torch.Tensor,
        evals: torch.Tensor,
        evecs: torch.Tensor,
    ) -> torch.Tensor:
        feats = []
        if self.use_xyz:
            feats.append(V)
        if self.n_hks > 0:
            feats.append(compute_hks(evals, evecs, self.n_hks, self.hks_k_high, self.eps))
        if self.n_wks > 0:
            feats.append(compute_wks(evals, evecs, self.n_wks, self.eps))

        x = torch.cat(feats, dim=1)
        x = mass_weighted_standardize(x, mass, eps=self.eps)
        return x

    def pool(
        self,
        z_vertex: torch.Tensor,
        mass: torch.Tensor,
        evecs: torch.Tensor,
    ) -> torch.Tensor:
        if self.pooling == "mean":
            w = mass / (mass.sum() + self.eps)
            return (w[:, None] * z_vertex).sum(dim=0, keepdim=True)

        if self.pooling == "attention":
            w = mass / (mass.sum() + self.eps)
            logits = self.att_logits(z_vertex).squeeze(-1) + torch.log(w.clamp_min(self.eps))
            alpha = torch.softmax(logits, dim=0)
            return (alpha[:, None] * z_vertex).sum(dim=0, keepdim=True)

        # spectral pooling
        n_verts, c = z_vertex.shape
        k = min(self.spectral_k, evecs.shape[1])
        basis = evecs[:, :k]
        if k < self.spectral_k:
            pad = torch.zeros(n_verts, self.spectral_k - k, device=evecs.device, dtype=evecs.dtype)
            basis = torch.cat([basis, pad], dim=1)
        w = mass / (mass.sum() + self.eps)
        coeff = basis.T @ (w[:, None] * z_vertex)  # [K, C]
        coeff = coeff.reshape(1, self.spectral_k * c)
        return self.spectral_head(coeff)

    def forward(
        self,
        V: torch.Tensor,
        mass: torch.Tensor,
        L: torch.Tensor,
        evals: torch.Tensor,
        evecs: torch.Tensor,
        faces: torch.Tensor,
        gradX: torch.Tensor,
        gradY: torch.Tensor,
        add_noise: bool = True,
        return_per_vertex: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        x = self.build_vertex_features(V, mass, evals, evecs)
        z_vertex = self.encoder(x, mass, L, evals, evecs, faces=faces, gradX=gradX, gradY=gradY)
        z_vertex = self.vertex_head(z_vertex)

        if add_noise and self.noise_std > 0.0:
            z_vertex = z_vertex + self.noise_std * torch.randn_like(z_vertex)

        z_global = self.pool(z_vertex, mass, evecs)
        if return_per_vertex:
            return z_vertex, z_global
        return z_global


# ------------------------------------------------------------
# Triplets + evaluation
# ------------------------------------------------------------
def build_triplets_from_gt(
    D_gt: torch.Tensor,
    min_gt_gap: float,
    max_triplets_per_anchor: int,
) -> List[Tuple[int, int, int]]:
    d = D_gt.detach().cpu().numpy()
    n = d.shape[0]
    triplets: List[Tuple[int, int, int]] = []

    for a in range(n):
        order = np.argsort(d[a])
        order = order[order != a]
        if order.size < 2:
            continue

        added = 0
        for pi in range(order.size - 1):
            p = int(order[pi])
            dp = float(d[a, p])
            for ni in range(pi + 1, order.size):
                neg = int(order[ni])
                dn = float(d[a, neg])
                if dp + min_gt_gap >= dn:
                    continue
                triplets.append((a, p, neg))
                added += 1
                if max_triplets_per_anchor > 0 and added >= max_triplets_per_anchor:
                    break
            if max_triplets_per_anchor > 0 and added >= max_triplets_per_anchor:
                break
    return triplets


def triplet_ranking_loss(
    Z: torch.Tensor,
    triplets: List[Tuple[int, int, int]],
    margin: float,
) -> Tuple[torch.Tensor, float, float, float]:
    if not triplets:
        zero = Z.new_zeros(())
        return zero, float("nan"), float("nan"), float("nan")

    trip_idx = torch.tensor(triplets, dtype=torch.long, device=Z.device)
    a = trip_idx[:, 0]
    p = trip_idx[:, 1]
    n = trip_idx[:, 2]

    d_ap = torch.norm(Z[a] - Z[p], dim=1)
    d_an = torch.norm(Z[a] - Z[n], dim=1)
    violations = F.relu(d_ap - d_an + margin)
    loss = violations.mean()

    active_frac = float((violations > 0).float().mean().item())
    return loss, active_frac, float(d_ap.mean().item()), float(d_an.mean().item())


def encode_subject(
    model: nn.Module,
    dataset: GTReadyDataset,
    mesh_indices: Sequence[int],
    device: torch.device,
    add_noise: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
    z_list: List[torch.Tensor] = []

    for mesh_idx in mesh_indices:
        try:
            sample = dataset[int(mesh_idx)]
        except Exception:
            continue

        V, mass, L, evals, evecs, faces, gradX, gradY = load_ops_sample(sample, device)
        z = model(
            V,
            mass,
            L,
            evals,
            evecs,
            faces,
            gradX,
            gradY,
            add_noise=add_noise,
            return_per_vertex=False,
        )
        z_list.append(z.squeeze(0))

    if not z_list:
        return None

    z_stack = torch.stack(z_list, dim=0)
    z_mean = z_stack.mean(dim=0)
    if z_stack.shape[0] > 1:
        intra = ((z_stack - z_mean[None, :]) ** 2).mean()
    else:
        intra = z_mean.new_zeros(())

    return z_mean, intra, int(z_stack.shape[0])


@torch.no_grad()
def evaluate_ranking(
    model: nn.Module,
    dataset: GTReadyDataset,
    subject_map: Dict[str, List[int]],
    eval_subjects: Sequence[str],
    gt_matrix: np.ndarray,
    gt_name_to_idx: Dict[str, int],
    device: torch.device,
    l2_normalize: bool,
) -> Optional[dict]:
    model.eval()

    subj_embs: List[torch.Tensor] = []
    subj_ids: List[str] = []
    intra_vals: List[float] = []

    for sid in eval_subjects:
        if sid not in gt_name_to_idx:
            continue
        packed = encode_subject(
            model=model,
            dataset=dataset,
            mesh_indices=subject_map[sid],
            device=device,
            add_noise=False,
        )
        if packed is None:
            continue
        z_mean, intra, _ = packed
        subj_embs.append(z_mean)
        subj_ids.append(sid)
        intra_vals.append(float(intra.item()))

    if len(subj_embs) < 3:
        return None

    Z = torch.stack(subj_embs, dim=0)
    if l2_normalize:
        Z = F.normalize(Z, dim=1)

    gt_idx = np.array([gt_name_to_idx[s] for s in subj_ids], dtype=int)
    D_gt = torch.tensor(gt_matrix[np.ix_(gt_idx, gt_idx)], device=device, dtype=Z.dtype)
    D_emb = torch.cdist(Z, Z, p=2)

    gt_vals = upper_triangular_values(D_gt).detach().cpu().numpy()
    emb_vals = upper_triangular_values(D_emb).detach().cpu().numpy()

    return {
        "n_subjects_eval": len(subj_ids),
        "pearson": pearson_corr(gt_vals, emb_vals),
        "spearman": spearman_corr(gt_vals, emb_vals),
        "nn_match": nearest_neighbor_match_rate(D_gt, D_emb),
        "intra_mean": float(np.mean(intra_vals)) if intra_vals else float("nan"),
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    args = parse_args()
    apply_feature_preset(args)
    seed_everything(args.seed)

    if (not args.use_xyz) and args.n_hks <= 0 and args.n_wks <= 0:
        raise ValueError("No features enabled. Use xyz and/or HKS/WKS.")

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        f"Features: preset={args.feature_preset}, xyz={args.use_xyz}, "
        f"hks={args.n_hks}, wks={args.n_wks}"
    )
    print(
        f"Pooling: {args.pooling} | latent_dim={args.latent_dim}, width={args.width}, "
        f"blocks={args.n_blocks}"
    )

    dataset = GTReadyDataset(args.data_dir)
    subject_map = build_subject_map(dataset.files)
    if not subject_map:
        raise RuntimeError(f"No subject ids parsed from dataset files in {args.data_dir}")

    gt_matrix, gt_name_to_idx = load_gt_distance_matrix(args.dist_npz)

    usable_subjects = sorted([s for s in subject_map.keys() if s in gt_name_to_idx])
    if len(usable_subjects) < 6:
        raise RuntimeError(
            f"Only {len(usable_subjects)} subjects overlap with GT matrix; need >= 6"
        )

    train_subjects, eval_subjects_full = split_subjects(
        subjects=usable_subjects,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_subjects=args.max_subjects,
    )

    if args.eval_subject_cap > 0 and len(eval_subjects_full) > args.eval_subject_cap:
        rng = np.random.default_rng(args.seed + 99)
        chosen = rng.choice(eval_subjects_full, size=args.eval_subject_cap, replace=False)
        eval_subjects = sorted(chosen.tolist())
    else:
        eval_subjects = list(eval_subjects_full)

    print(
        f"Subjects: total={len(usable_subjects)} "
        f"train={len(train_subjects)} eval={len(eval_subjects)}"
    )
    print(f"Dataset meshes: {len(dataset.files)}")

    model = DiffusionRankingEncoder(
        latent_dim=args.latent_dim,
        width=args.width,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        use_xyz=args.use_xyz,
        n_hks=args.n_hks,
        n_wks=args.n_wks,
        pooling=args.pooling,
        spectral_k=args.spectral_k,
        hks_k_high=args.hks_k_high,
        eps=args.feature_eps,
        noise_std=args.noise_std,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    with open(Path(args.out_dir) / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    log_csv = Path(args.out_dir) / "train_log.csv"
    with open(log_csv, "w", encoding="utf-8") as f:
        f.write(
            "epoch,loss,triplet,intra,active_frac,n_triplets,dist_ap,dist_an,lr,"
            "spearman,pearson,nn_match,intra_eval,n_eval\n"
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = np.random.default_rng(args.seed + epoch)
        train_perm = rng.permutation(np.array(train_subjects, dtype=object))

        epoch_loss = 0.0
        epoch_trip = 0.0
        epoch_intra = 0.0
        epoch_active = 0.0
        epoch_triplet_count = 0.0
        epoch_dap = 0.0
        epoch_dan = 0.0
        n_steps = 0

        pbar = tqdm(
            range(0, len(train_perm), args.batch_subjects),
            desc=f"Epoch {epoch}/{args.epochs}",
        )

        for start in pbar:
            batch_subjects = train_perm[start : start + args.batch_subjects].tolist()

            subj_embeddings: List[torch.Tensor] = []
            subj_intra: List[torch.Tensor] = []
            gt_indices: List[int] = []

            for sid in batch_subjects:
                packed = encode_subject(
                    model=model,
                    dataset=dataset,
                    mesh_indices=subject_map[sid],
                    device=device,
                    add_noise=True,
                )
                if packed is None:
                    continue
                z_mean, intra, _ = packed
                subj_embeddings.append(z_mean)
                subj_intra.append(intra)
                gt_indices.append(gt_name_to_idx[sid])

            if len(subj_embeddings) < 3:
                continue

            Z = torch.stack(subj_embeddings, dim=0)
            if args.l2_normalize:
                Z = F.normalize(Z, dim=1)

            D_gt_batch = torch.tensor(
                gt_matrix[np.ix_(gt_indices, gt_indices)],
                device=device,
                dtype=Z.dtype,
            )

            triplets = build_triplets_from_gt(
                D_gt=D_gt_batch,
                min_gt_gap=args.min_gt_gap,
                max_triplets_per_anchor=args.max_triplets_per_anchor,
            )
            if not triplets:
                continue

            optimizer.zero_grad(set_to_none=True)

            loss_triplet, active_frac, mean_dap, mean_dan = triplet_ranking_loss(
                Z=Z,
                triplets=triplets,
                margin=args.triplet_margin,
            )
            loss_intra = torch.stack(subj_intra).mean() if subj_intra else Z.new_zeros(())

            loss = loss_triplet + args.lambda_intra * loss_intra
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            n_steps += 1
            epoch_loss += float(loss.item())
            epoch_trip += float(loss_triplet.item())
            epoch_intra += float(loss_intra.item())
            epoch_active += active_frac
            epoch_triplet_count += len(triplets)
            epoch_dap += mean_dap
            epoch_dan += mean_dan

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                trip=f"{loss_triplet.item():.4f}",
                intra=f"{loss_intra.item():.4f}",
                act=f"{active_frac:.2f}",
                triplets=str(len(triplets)),
            )

        if n_steps == 0:
            raise RuntimeError("No valid optimization step in epoch. Check dataset/mapping settings.")

        epoch_loss /= n_steps
        epoch_trip /= n_steps
        epoch_intra /= n_steps
        epoch_active /= n_steps
        epoch_triplet_count /= n_steps
        epoch_dap /= n_steps
        epoch_dan /= n_steps

        scheduler.step(epoch_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        metrics = None
        if (epoch % args.eval_every) == 0:
            metrics = evaluate_ranking(
                model=model,
                dataset=dataset,
                subject_map=subject_map,
                eval_subjects=eval_subjects,
                gt_matrix=gt_matrix,
                gt_name_to_idx=gt_name_to_idx,
                device=device,
                l2_normalize=args.l2_normalize,
            )

        if metrics is None:
            metrics = {
                "spearman": float("nan"),
                "pearson": float("nan"),
                "nn_match": float("nan"),
                "intra_mean": float("nan"),
                "n_subjects_eval": 0,
            }

        print(
            f"Epoch {epoch:03d} | loss={epoch_loss:.5f} trip={epoch_trip:.5f} "
            f"intra={epoch_intra:.5f} active={epoch_active:.3f} "
            f"spearman={metrics['spearman']:.4f} nn={metrics['nn_match']:.4f} lr={lr_now:.2e}"
        )

        with open(log_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},"
                f"{epoch_loss:.6f},{epoch_trip:.6f},{epoch_intra:.6f},{epoch_active:.6f},"
                f"{epoch_triplet_count:.1f},{epoch_dap:.6f},{epoch_dan:.6f},{lr_now:.2e},"
                f"{metrics['spearman']:.6f},{metrics['pearson']:.6f},{metrics['nn_match']:.6f},"
                f"{metrics['intra_mean']:.6f},{int(metrics['n_subjects_eval'])}\n"
            )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = Path(args.out_dir) / f"encoder_triplet_epoch{epoch}.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"Saved checkpoint: {ckpt}")

    print("Training completed.")


if __name__ == "__main__":
    main()
