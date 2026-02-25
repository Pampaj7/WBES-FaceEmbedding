#!/usr/bin/env python3
"""
Baseline intrinseco senza DiffusionNet:
  1) HKS per vertice
  2) Proiezione in base spettrale (Phi^T M HKS)
  3) Flatten
  4) Distanza L2 tra embedding globali

Valuta il ranking rispetto a D_GT (Spearman/Pearson + NN match rate).
"""

from __future__ import annotations
from tqdm import tqdm
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[4]
AUTOENCODER_DIR = REPO_ROOT / "face_embedding" / "gt_encdec" / "autoencoder"

if str(AUTOENCODER_DIR) not in sys.path:
    sys.path.append(str(AUTOENCODER_DIR))

from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset  # noqa: E402


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
    default_out = THIS_FILE.parent / "runs_hks_spectral_l2"

    parser = argparse.ArgumentParser(description="HKS + spectral projection + flatten + L2 baseline.")
    parser.add_argument("--data_dir", type=str, default=str(default_data_dir))
    parser.add_argument("--dist_npz", type=str, default=str(default_dist))
    parser.add_argument("--out_dir", type=str, default=str(default_out))

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_subjects", type=int, default=300, help="0 = tutti i soggetti")

    parser.add_argument("--n_hks", type=int, default=16)
    parser.add_argument("--hks_k_high", type=int, default=50)
    parser.add_argument("--proj_k", type=int, default=32, help="numero modi per proiezione spettrale")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--hks_times_mode",
        type=str,
        default="spectral",
        choices=("spectral", "autoscale"),
        help="spectral: tempi legati a lambda; autoscale: logspace fisso [-2,0] come nel codice diffusion-net",
    )
    parser.add_argument("--no_standardize_hks", action="store_true")
    parser.add_argument("--l2_normalize_embedding", action="store_true")
    parser.add_argument("--abs_coeffs", action="store_true", help="usa valore assoluto dei coefficienti spettrali")
    parser.add_argument(
        "--fix_evec_sign",
        action="store_true",
        help="canonizza il segno degli autovettori per ridurre instabilita inter-mesh",
    )
    return parser.parse_args()


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
        raise KeyError(f"{path} deve contenere 'D_orig' e 'names'. Trovate: {pack.files}")

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
        raise RuntimeError("Impossibile parsare subject ids da names in dist_npz.")
    return D, name_to_idx


def to_device_tensor(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    if x.is_sparse:
        return x.coalesce().to(device)
    return x.to(device)


def load_ops_sample(sample: dict, device: torch.device) -> Tuple[torch.Tensor, ...]:
    keys = ("verts", "mass", "L", "evals", "evecs", "faces", "gradX", "gradY")
    vals = [to_device_tensor(sample[k], device) for k in keys]
    return tuple(vals)  # type: ignore[return-value]


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
    times_mode: str = "spectral",
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
        # Equivalente a compute_hks_autoscale() nel codice diffusion-net.
        t = torch.logspace(
            -2.0,
            0.0,
            n_hks,
            device=evals.device,
            dtype=evals.dtype,
        )
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
            n_hks,
            device=evals.device,
            dtype=evals.dtype,
        )
    hks = (evecs_k ** 2) @ torch.exp(-evals_k[:, None] * t[None, :])
    return torch.log(hks + eps)


def canonicalize_evec_sign(evecs: torch.Tensor, mass: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Risolve l'ambiguita di segno +/- per ogni autovettore usando il terzo momento:
      s_k = sign(sum_i m_i * phi_{ik}^3)
    """
    moments = (mass[:, None] * (evecs ** 3)).sum(dim=0)
    signs = torch.where(moments >= 0, torch.ones_like(moments), -torch.ones_like(moments))

    # fallback raro: momento ~0
    near_zero = moments.abs() < eps
    if near_zero.any():
        alt = evecs.abs().argmax(dim=0)
        alt_sign = torch.sign(evecs[alt, torch.arange(evecs.shape[1], device=evecs.device)])
        alt_sign = torch.where(alt_sign == 0, torch.ones_like(alt_sign), alt_sign)
        signs = torch.where(near_zero, alt_sign, signs)

    return evecs * signs[None, :]


def to_basis(values: torch.Tensor, basis: torch.Tensor, massvec: torch.Tensor) -> torch.Tensor:
    """
    Stessa logica del codice diffusion-net:
      basis^T (values * mass)
    values: (V,D), basis: (V,K), massvec: (V)
    out: (K,D)
    """
    return basis.transpose(-2, -1) @ (values * massvec[:, None])


def project_hks_spectral(
    hks: torch.Tensor,
    evecs: torch.Tensor,
    mass: torch.Tensor,
    proj_k: int,
    abs_coeffs: bool,
) -> torch.Tensor:
    n_verts, _ = hks.shape
    k = min(proj_k, evecs.shape[1])
    phi = evecs[:, :k]

    coeff = to_basis(hks, phi, mass)  # [k, T]
    if abs_coeffs:
        coeff = coeff.abs()

    if k < proj_k:
        pad = torch.zeros(proj_k - k, coeff.shape[1], device=hks.device, dtype=hks.dtype)
        coeff = torch.cat([coeff, pad], dim=0)

    return coeff.reshape(-1)  # flatten


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


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = GTReadyDataset(args.data_dir)
    subject_map = build_subject_map(dataset.files)
    if not subject_map:
        raise RuntimeError("Nessun subject id parsato dai file dataset.")

    D_gt_full, gt_name_to_idx = load_gt_distance_matrix(args.dist_npz)
    shared_subjects = sorted([s for s in subject_map if s in gt_name_to_idx])
    if len(shared_subjects) < 3:
        raise RuntimeError(f"Soggetti in comune con D_GT insufficienti: {len(shared_subjects)}")

    if args.max_subjects > 0 and len(shared_subjects) > args.max_subjects:
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(shared_subjects, size=args.max_subjects, replace=False)
        shared_subjects = sorted(pick.tolist())

    print(f"Subjects usati: {len(shared_subjects)}")
    print(
        f"Pipeline: HKS(T={args.n_hks}) -> spectral proj(K={args.proj_k}) -> flatten -> L2 "
        f"(standardize_hks={not args.no_standardize_hks}, abs_coeffs={args.abs_coeffs}, "
        f"hks_times_mode={args.hks_times_mode}, fix_evec_sign={args.fix_evec_sign})"
    )

    embeddings = []
    valid_subjects = []

    for sid in tqdm(shared_subjects, desc="Encoding subjects", ncols=100):        
        mesh_embs = []
        for mesh_idx in subject_map[sid]:
            try:
                sample = dataset[int(mesh_idx)]
            except Exception:
                continue

            V, mass, _, evals, evecs, _, _, _ = load_ops_sample(sample, device)
            del V

            hks = compute_hks(
                evals=evals,
                evecs=evecs,
                n_hks=args.n_hks,
                hks_k_high=args.hks_k_high,
                eps=args.eps,
                times_mode=args.hks_times_mode,
            )
            if not args.no_standardize_hks:
                hks = mass_weighted_standardize(hks, mass, eps=args.eps)

            evecs_proj = canonicalize_evec_sign(evecs, mass, eps=args.eps) if args.fix_evec_sign else evecs
            z = project_hks_spectral(
                hks=hks,
                evecs=evecs_proj,
                mass=mass,
                proj_k=args.proj_k,
                abs_coeffs=args.abs_coeffs,
            )
            if args.l2_normalize_embedding:
                z = torch.nn.functional.normalize(z[None, :], dim=1).squeeze(0)

            mesh_embs.append(z)

        if not mesh_embs:
            continue

        z_subject = torch.stack(mesh_embs, dim=0).mean(dim=0)
        embeddings.append(z_subject)
        valid_subjects.append(sid)

    if len(embeddings) < 3:
        raise RuntimeError(f"Embedding validi insufficienti: {len(embeddings)}")

    Z = torch.stack(embeddings, dim=0)
    D_emb = torch.cdist(Z, Z, p=2)

    gt_idx = np.array([gt_name_to_idx[s] for s in valid_subjects], dtype=int)
    D_gt = torch.tensor(D_gt_full[np.ix_(gt_idx, gt_idx)], device=device, dtype=Z.dtype)

    gt_vals = upper_triangular_values(D_gt).detach().cpu().numpy()
    emb_vals = upper_triangular_values(D_emb).detach().cpu().numpy()

    metrics = {
        "n_subjects": int(len(valid_subjects)),
        "spearman": spearman_corr(gt_vals, emb_vals),
        "pearson": pearson_corr(gt_vals, emb_vals),
        "nn_match_rate": nn_match_rate(D_gt, D_emb),
    }

    print(
        f"Spearman={metrics['spearman']:.4f} | "
        f"Pearson={metrics['pearson']:.4f} | "
        f"NN-match={metrics['nn_match_rate']:.4f}"
    )

    np.savez(
        Path(args.out_dir) / "baseline_outputs.npz",
        subjects=np.array(valid_subjects, dtype=object),
        Z=Z.detach().cpu().numpy(),
        D_emb=D_emb.detach().cpu().numpy(),
        D_gt=D_gt.detach().cpu().numpy(),
    )
    with open(Path(args.out_dir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(Path(args.out_dir) / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Salvato in: {args.out_dir}")


if __name__ == "__main__":
    main()
