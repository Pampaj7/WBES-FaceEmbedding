#!/usr/bin/env python3
import os
import re
import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import sys

sys.path.append(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/gt_encdec/autoencoder"
)

from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset
from diffusion_autoencoder import DiffusionEncoderXYZSpectrum, DiffusionEncoderOnlyIntrinsec
from latent_loss import stress_loss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    # NOTE: deprecation warning in recent PyTorch; still works for now.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ============================================================
# RUN DIR (config-based)
# ============================================================

RUNS_ROOT = Path("runs_diffusion_xyz_spectrum")

# ============================================================
# ARGPARSE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_xyz", action="store_true")
    parser.add_argument("--use_spectrum", action="store_true")
    parser.add_argument(
    "--model",
    type=str,
    default="xyz_spectrum",
    choices=["xyz_spectrum", "hkswks"],
    help="Which encoder to train",
    )

    # Args usati solo da xyz_spectrum (li hai già, li lasci)
    # --use_spectrum --k_spec --log_input

    # Args usati solo da hkswks
    parser.add_argument("--n_hks", type=int, default=16)
    parser.add_argument("--n_wks", type=int, default=16)
    parser.add_argument("--eig_k", type=int, default=300)
    parser.add_argument("--pool_mode", type=str, default="meanmax", choices=["mean", "meanmax"])
    
    parser.add_argument("--k_spec", type=int, default=100)
    parser.add_argument("--log_input", action="store_true")
    parser.add_argument("--eps", type=float, default=1e-8)

    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_subjects", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--lambda_stress", type=float, default=0.3)
    parser.add_argument("--lambda_id", type=float, default=0.1)

    parser.add_argument("--max_meshes_per_subject_train", type=int, default=0)
    parser.add_argument("--max_meshes_per_subject_eval", type=int, default=0)

    parser.add_argument("--lambda_rank", type=float, default=0.3)
    parser.add_argument("--rank_margin", type=float, default=0.05)
    parser.add_argument("--rank_pairs", type=int, default=2048)
    parser.add_argument("--rank_tau", type=float, default=0.02)
    parser.add_argument("--rank_hard_frac", type=float, default=0.7)
    
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--eval_every", type=int, default=1)

    # Preflight checks (fail-fast)
    parser.add_argument("--preflight", action="store_true", help="Run preflight checks before training")
    parser.add_argument("--preflight_samples", type=int, default=500, help="How many meshes to sample for preflight")
    parser.add_argument("--preflight_subjects", type=int, default=300, help="How many subjects for GT-alignment baseline")
    parser.add_argument("--preflight_eps_range", type=float, default=1e-3, help="Range threshold to consider a channel 'almost constant' across meshes")
    parser.add_argument("--preflight_dead_frac_warn", type=float, default=0.30, help="Warn if fraction of constant channels exceeds this")
    parser.add_argument("--preflight_dead_frac_stop", type=float, default=0.50, help="Stop if fraction of constant channels exceeds this")
    
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# CONFIG PATHS
# ============================================================

DATA_DIR = "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/REMESH/npz_data_topo_500_withops"

DIST_PATH = (
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/"
    "gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/"
    "normalized_matrix_distances.npz"
)

VARIANT_RE = re.compile(r"^(id\d+)_.*\.npz$")


# ============================================================
# SUBJECT GROUPING
# ============================================================

def build_subject_map(dataset):
    subj_to_idxs = {}
    for idx, fname in enumerate(dataset.files):
        m = VARIANT_RE.match(fname)
        subj = fname.split("_")[0] if m is None else m.group(1)
        subj_to_idxs.setdefault(subj, []).append(idx)
    return subj_to_idxs

def pairwise_rank_loss(
    D_lat: torch.Tensor,
    D_gt: torch.Tensor,
    n_pairs: int = 2048,
    margin: float = 0.05,
    tau: float = 0.02,
    hard_frac: float = 0.7,
) -> torch.Tensor:
    """
    Spearman-surrogate ranking loss on distances (robust + "hard" sampling).

    Goal:
      Preserve relative ordering of distances:
        if D_gt[i,j] > D_gt[i,k] + tau  =>  D_lat[i,j] > D_lat[i,k] + margin

    Key improvements vs naive version:
      1) Ambiguity filtering (tau): ignores nearly-equal GT comparisons that are mostly noise.
      2) Hard sampling (hard_frac): often compares a "far" vs a "near" target for the same anchor,
         increasing signal when batch size is small.

    Args:
      D_lat, D_gt: [B,B] symmetric, diagonal ~0
      n_pairs: number of constraints sampled
      margin: hinge margin in latent distance units
      tau: GT separation threshold (in GT distance units, your D_gt is ~[0,1])
      hard_frac: fraction of constraints built with hard (near vs far) sampling.
                remaining constraints are random (for coverage).

    Returns:
      Scalar tensor loss.
    """
    B = int(D_gt.size(0))
    if B < 3:
        return torch.zeros((), device=D_gt.device, dtype=D_gt.dtype)

    device = D_gt.device
    dtype = D_gt.dtype

    # How many hard vs random constraints
    n_hard = int(round(float(n_pairs) * float(hard_frac)))
    n_rand = int(n_pairs) - n_hard

    losses = []

    # ----------------------------
    # (A) Hard constraints
    # ----------------------------
    if n_hard > 0:
        # pick anchors
        i = torch.randint(0, B, (n_hard,), device=device)

        # For each anchor, select a "near" k and a "far" j using GT distances.
        # We'll avoid self by masking the diagonal to -inf / +inf.
        # d: [n_hard, B]
        d = D_gt[i]  # gather rows

        # mask self
        idx = torch.arange(B, device=device).unsqueeze(0)  # [1,B]
        self_mask = idx == i.unsqueeze(1)
        d_near = d.masked_fill(self_mask, float("inf"))
        d_far = d.masked_fill(self_mask, float("-inf"))

        # choose among top-k extremes (adds some stochasticity)
        # k_candidates and j_candidates are indices in [0,B)
        k_top = min(3, B - 1)  # near candidates
        j_top = min(3, B - 1)  # far candidates

        k_candidates = torch.topk(d_near, k=k_top, largest=False).indices  # [n_hard,k_top]
        j_candidates = torch.topk(d_far, k=j_top, largest=True).indices    # [n_hard,j_top]

        kk = torch.randint(0, k_top, (n_hard,), device=device)
        jj = torch.randint(0, j_top, (n_hard,), device=device)
        k = k_candidates[torch.arange(n_hard, device=device), kk]
        j = j_candidates[torch.arange(n_hard, device=device), jj]

        # Ensure j != k (rare but possible when B small and candidates overlap)
        j = torch.where(j == k, (j + 1) % B, j)

        dgj = D_gt[i, j]
        dgk = D_gt[i, k]
        dlj = D_lat[i, j]
        dlk = D_lat[i, k]

        mask = dgj > (dgk + tau)
        if mask.any():
            losses.append(torch.relu(margin - (dlj[mask] - dlk[mask])).mean())

    # ----------------------------
    # (B) Random constraints (coverage)
    # ----------------------------
    if n_rand > 0:
        i = torch.randint(0, B, (n_rand,), device=device)
        j = torch.randint(0, B, (n_rand,), device=device)
        k = torch.randint(0, B, (n_rand,), device=device)

        # avoid trivial equal indices
        j = torch.where(j == i, (j + 1) % B, j)
        k = torch.where(k == i, (k + 2) % B, k)
        k = torch.where(k == j, (k + 1) % B, k)

        dgj = D_gt[i, j]
        dgk = D_gt[i, k]
        dlj = D_lat[i, j]
        dlk = D_lat[i, k]

        mask = dgj > (dgk + tau)
        if mask.any():
            losses.append(torch.relu(margin - (dlj[mask] - dlk[mask])).mean())

    if not losses:
        return torch.zeros((), device=device, dtype=dtype)

    return torch.stack(losses).mean()
def _slug(x: str) -> str:
    x = x.strip().lower()
    x = re.sub(r"[^a-z0-9._-]+", "-", x)
    x = re.sub(r"-+", "-", x).strip("-")
    return x

def make_run_dir(args: argparse.Namespace) -> Path:
    """
    Create a run directory based on a stable fingerprint of the args.
    Saves config.json and creates checkpoints/ folder.
    """
    fp = {
        "use_xyz": bool(args.use_xyz),
        "use_spectrum": bool(args.use_spectrum),
        "k_spec": int(args.k_spec),
        "log_input": bool(args.log_input),
        "eps": float(args.eps),

        "latent_dim": int(args.latent_dim),
        "width": int(args.width),
        "n_blocks": int(args.n_blocks),
        "dropout": float(args.dropout),

        "epochs": int(args.epochs),
        "batch_subjects": int(args.batch_subjects),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),

        "lambda_stress": float(args.lambda_stress),
        "lambda_id": float(args.lambda_id),

        "max_meshes_per_subject_train": int(args.max_meshes_per_subject_train),
        "max_meshes_per_subject_eval": int(args.max_meshes_per_subject_eval),

        "seed": int(args.seed),
        "save_every": int(args.save_every),
        "eval_every": int(args.eval_every),
    }

    fp_json = json.dumps(fp, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha1(fp_json.encode("utf-8")).hexdigest()[:8]

    name = (
        f"xyz{int(fp['use_xyz'])}_spec{int(fp['use_spectrum'])}"
        f"_k{fp['k_spec']}_log{int(fp['log_input'])}"
        f"_z{fp['latent_dim']}_w{fp['width']}_b{fp['n_blocks']}"
        f"_bs{fp['batch_subjects']}_lr{fp['lr']:.1e}_wd{fp['weight_decay']:.1e}"
        f"_ls{fp['lambda_stress']:.2g}_li{fp['lambda_id']:.2g}"
        f"_seed{fp['seed']}"
        f"__{h}"
    )

    run_dir = RUNS_ROOT / _slug(name)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    # Save config
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "fingerprint": fp}, f, indent=2, sort_keys=True)

    # Optional timestamp file (doesn't affect folder name)
    (run_dir / "run_started.txt").write_text(datetime.now().isoformat(), encoding="utf-8")

    return run_dir

def pick_mesh_indices(
    idxs: Sequence[int],
    max_meshes: int,
    rng: Optional[np.random.Generator],
) -> List[int]:
    if max_meshes <= 0 or len(idxs) <= max_meshes:
        return list(idxs)
    if rng is None:
        return list(idxs[:max_meshes])
    picked = rng.choice(np.asarray(idxs), size=max_meshes, replace=False)
    return [int(i) for i in picked.tolist()]

def _rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    order = a.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    return ranks

def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = _rankdata(x)
    yr = _rankdata(y)
    if xr.std() < 1e-12 or yr.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])

def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def spectrum_vector_from_evals(
    evals: torch.Tensor,
    k_spec: int,
    log_input: bool,
    eps: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    EXACTLY matches DiffusionEncoderXYZSpectrum._spectrum_vector():
      ev = evals.flatten()[1:]
      spec = ev[:k_spec] padded with zeros
      if log_input: log(clamp_min(eps))
    Returns: spec [k_spec]
    """
    ev = evals.flatten()
    if ev.numel() <= 1:
        return torch.zeros(k_spec, device=ev.device, dtype=dtype)

    ev = ev[1:].to(dtype)

    if ev.numel() >= k_spec:
        spec = ev[:k_spec]
    else:
        pad = torch.zeros(k_spec - ev.numel(), device=ev.device, dtype=dtype)
        spec = torch.cat([ev, pad], dim=0)

    if log_input:
        spec = torch.log(spec.clamp_min(eps))

    return spec

@torch.inference_mode()
def preflight_spectrum_sanity(
    dataset,
    run_dir: Path,
    k_spec: int,
    log_input: bool,
    eps: float,
    n_meshes: int,
    seed: int,
    eps_range: float,
    dead_frac_warn: float,
    dead_frac_stop: float,
) -> Dict[str, object]:
    """
    Sanity check for *your* spectrum:
    - Because spectrum is replicated per-vertex, per-vertex std is always 0 (not useful).
    - So we check per-channel variation ACROSS meshes.
    - We log per-channel p1/p50/p99 across sampled meshes and mark channels with (p99-p1) < eps_range as 'almost constant'.
    """
    pre_dir = run_dir / "preflight"
    pre_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    n_total = len(dataset)
    n_take = min(int(n_meshes), n_total)
    idxs = rng.choice(np.arange(n_total), size=n_take, replace=False)

    specs = []
    for i in idxs:
        sample = dataset[int(i)]
        spec = spectrum_vector_from_evals(
            sample["evals"],
            k_spec=k_spec,
            log_input=log_input,
            eps=eps,
            dtype=torch.float32,
        )
        specs.append(spec.detach().cpu().numpy())

    S = np.stack(specs, axis=0)  # [N,k]
    # per-channel distribution across meshes
    mean = S.mean(axis=0)
    std = S.std(axis=0)
    p1 = np.percentile(S, 1, axis=0)
    p50 = np.percentile(S, 50, axis=0)
    p99 = np.percentile(S, 99, axis=0)
    rng99 = p99 - p1

    const_mask = rng99 < float(eps_range)
    const_frac = float(const_mask.mean())

    # write CSV
    csv_path = pre_dir / "preflight_spectrum_stats.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("channel,mean,std,p1,p50,p99,p99_minus_p1,is_almost_constant\n")
        for c in range(S.shape[1]):
            f.write(
                f"{c},{mean[c]:.8e},{std[c]:.8e},{p1[c]:.8e},{p50[c]:.8e},{p99[c]:.8e},{rng99[c]:.8e},{int(const_mask[c])}\n"
            )

    out = {
        "n_meshes_sampled": int(n_take),
        "k_spec_effective": int(S.shape[1]),
        "log_input": bool(log_input),
        "eps": float(eps),
        "eps_range": float(eps_range),
        "almost_constant_channels": int(const_mask.sum()),
        "almost_constant_fraction": float(const_frac),
        "range_p99_p1_min": float(rng99.min()),
        "range_p99_p1_median": float(np.median(rng99)),
        "range_p99_p1_max": float(rng99.max()),
        "csv_path": str(csv_path),
        "status": "ok",
    }

    # warn/stop policy
    if const_frac > dead_frac_stop:
        out["status"] = "stop"
        out["reason"] = (
            f"Too many spectrum channels are almost constant across meshes "
            f"(fraction={const_frac:.2f} > stop={dead_frac_stop:.2f}). "
            f"This usually means your spectrum carries little discriminative signal "
            f"(or is overly squashed by log/scale)."
        )
    elif const_frac > dead_frac_warn:
        out["status"] = "warn"
        out["reason"] = (
            f"Many spectrum channels are almost constant across meshes "
            f"(fraction={const_frac:.2f} > warn={dead_frac_warn:.2f})."
        )

    json_path = pre_dir / "preflight_spectrum_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    out["json_path"] = str(json_path)

    return out

@torch.inference_mode()
def preflight_gt_alignment_baseline(
    dataset,
    subj_map: Dict[str, List[int]],
    subjects: Sequence[str],
    name_to_idx: Dict[str, int],
    D_orig: np.ndarray,
    run_dir: Path,
    k_spec: int,
    log_input: bool,
    eps: float,
    n_subjects: int,
    seed: int,
) -> Dict[str, object]:
    """
    Baseline: build one embedding per subject using ONLY the spectrum vector.
    - For each subject, average spec vectors over a few meshes (here: all available indices, could subsample if huge).
    - Then compute D_embed in embedding space (L2 and cosine) and correlate with D_orig.
    """
    pre_dir = run_dir / "preflight"
    pre_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    subs = list(subjects)
    if len(subs) > n_subjects:
        subs = rng.choice(np.asarray(subs), size=int(n_subjects), replace=False).tolist()
        subs = sorted([str(s) for s in subs])

    # build subject embeddings
    kept = []
    Em = []

    for subj in subs:
        if subj not in subj_map or subj not in name_to_idx:
            continue
        idxs = subj_map[subj]
        if not idxs:
            continue

        # average spectrum across that subject's meshes (robust + cheap)
        specs = []
        for idx in idxs:
            sample = dataset[int(idx)]
            spec = spectrum_vector_from_evals(
                sample["evals"],
                k_spec=k_spec,
                log_input=log_input,
                eps=eps,
                dtype=torch.float32,
            )
            specs.append(spec)

        if not specs:
            continue
        spec_mean = torch.stack(specs, dim=0).mean(dim=0)  # [k]
        kept.append(subj)
        Em.append(spec_mean.detach().cpu().numpy())

    if len(kept) < 6:
        out = {"status": "skip", "reason": f"Too few subjects for baseline ({len(kept)})."}
        json_path = pre_dir / "preflight_gt_alignment.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        return out

    E = np.stack(Em, axis=0)  # [N,k]
    idx = np.array([name_to_idx[s] for s in kept], dtype=int)
    D_gt = D_orig[np.ix_(idx, idx)]

    # upper triangle
    iu = np.triu_indices(D_gt.shape[0], 1)    
    gt = D_gt[iu]

    # L2 distances
    # (vectorized cdist)
    diff = E[:, None, :] - E[None, :, :]
    D_l2 = np.sqrt((diff * diff).sum(axis=-1) + 1e-12)
    l2 = D_l2[iu]

    # cosine distances
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    D_cos = 1.0 - (En @ En.T)
    cos = D_cos[iu]

    out = {
        "status": "ok",
        "n_subjects_used": int(len(kept)),
        "k_spec": int(k_spec),
        "log_input": bool(log_input),
        "eps": float(eps),
        "pearson_l2": _pearson(gt, l2),
        "spearman_l2": _spearman(gt, l2),
        "pearson_cos": _pearson(gt, cos),
        "spearman_cos": _spearman(gt, cos),
        "note": (
            "This measures whether your *spectrum-only* embedding already aligns with D_orig. "
            "If correlations are ~0, either spectrum is not informative for D_orig or scaling is off."
        ),
    }

    json_path = pre_dir / "preflight_gt_alignment.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    out["json_path"] = str(json_path)
    return out

@torch.inference_mode()
def eval_latent_structure(
    model,
    dataset,
    subj_map,
    eval_subjects,
    name_to_idx,
    D_orig,
    max_meshes_per_subject_eval=0,
):
    model.eval()
    subj_mean: Dict[str, torch.Tensor] = {}
    intra_vals: List[float] = []

    for subj in eval_subjects:
        idxs = pick_mesh_indices(subj_map[subj], max_meshes_per_subject_eval, None)
        if not idxs:
            continue

        z_list: List[torch.Tensor] = []
        for idx in idxs:
            sample = dataset[int(idx)]
            Zg = model(
                sample["verts"].to(DEVICE),
                sample["mass"].to(DEVICE),
                sample["L"].to(DEVICE),
                sample["evals"].to(DEVICE),
                sample["evecs"].to(DEVICE),
                sample["faces"].to(DEVICE),
                sample["gradX"].to(DEVICE),
                sample["gradY"].to(DEVICE),
                return_per_vertex=False,
                add_noise=False,
            ).squeeze(0)
            z_list.append(Zg)

        if not z_list:
            continue

        Zs = torch.stack(z_list, dim=0)
        zm = Zs.mean(dim=0)
        subj_mean[subj] = zm
        intra_vals.append(float(((Zs - zm) ** 2).mean().item()))

    kept = [s for s in eval_subjects if s in name_to_idx and s in subj_mean]
    if len(kept) < 3:
        return None

    Zmat = torch.stack([subj_mean[s] for s in kept], dim=0)
    idx = np.array([name_to_idx[s] for s in kept], dtype=int)

    D_gt = torch.tensor(D_orig[np.ix_(idx, idx)], device=DEVICE, dtype=Zmat.dtype)

    iu = torch.triu_indices(D_gt.size(0), D_gt.size(1), offset=1)
    gt = D_gt[iu[0], iu[1]].detach().cpu().numpy()

    # 1) L2
    D_l2 = torch.cdist(Zmat, Zmat, p=2)
    lat_l2 = D_l2[iu[0], iu[1]].detach().cpu().numpy()
    pearson_l2 = np.corrcoef(gt, lat_l2)[0, 1]
    spearman_l2 = np.corrcoef(gt.argsort().argsort(), lat_l2.argsort().argsort())[0, 1]

    # 2) Cosine distance
    Z_norm = Zmat / (Zmat.norm(dim=1, keepdim=True) + 1e-12)
    D_cos = 1.0 - (Z_norm @ Z_norm.T)
    lat_cos = D_cos[iu[0], iu[1]].detach().cpu().numpy()
    pearson_cos = np.corrcoef(gt, lat_cos)[0, 1]
    spearman_cos = np.corrcoef(gt.argsort().argsort(), lat_cos.argsort().argsort())[0, 1]

    # 3) L2 after unit norm
    D_l2_unit = torch.cdist(Z_norm, Z_norm, p=2)
    lat_l2_unit = D_l2_unit[iu[0], iu[1]].detach().cpu().numpy()
    pearson_l2_unit = np.corrcoef(gt, lat_l2_unit)[0, 1]
    spearman_l2_unit = np.corrcoef(gt.argsort().argsort(), lat_l2_unit.argsort().argsort())[0, 1]

    # 4) L2 after z-score per dim
    Z_np = Zmat.detach().cpu().numpy()
    Z_z = (Z_np - Z_np.mean(axis=0, keepdims=True)) / (Z_np.std(axis=0, keepdims=True) + 1e-12)
    Z_z = torch.tensor(Z_z, device=DEVICE, dtype=Zmat.dtype)
    D_l2_z = torch.cdist(Z_z, Z_z, p=2)
    lat_l2_z = D_l2_z[iu[0], iu[1]].detach().cpu().numpy()
    pearson_l2_z = np.corrcoef(gt, lat_l2_z)[0, 1]
    spearman_l2_z = np.corrcoef(gt.argsort().argsort(), lat_l2_z.argsort().argsort())[0, 1]

    return {
        "pearson_l2": float(pearson_l2),
        "spearman_l2": float(spearman_l2),
        "pearson_cos": float(pearson_cos),
        "spearman_cos": float(spearman_cos),
        "pearson_l2_unit": float(pearson_l2_unit),
        "spearman_l2_unit": float(spearman_l2_unit),
        "pearson_l2_z": float(pearson_l2_z),
        "spearman_l2_z": float(spearman_l2_z),
        "intra_mean": float(np.mean(intra_vals)) if intra_vals else float("nan"),
        "n_eval": int(len(kept)),
    }

def main():
    args = parse_args()
    seed_everything(args.seed)

    # ------------------------------------------------------------
    # Basic run setup
    # ------------------------------------------------------------
    run_dir = make_run_dir(args)
    log_csv = run_dir / "train_log.csv"

    print(f"Device={DEVICE}")
    print(f"Run dir: {run_dir}")
    print(f"Model={args.model}")

    # Model-specific sanity (inputs)
    if args.model == "xyz_spectrum":
        if not (args.use_xyz or args.use_spectrum):
            raise ValueError("For --model xyz_spectrum enable --use_xyz and/or --use_spectrum")
        print(
            f"Inputs: XYZ={args.use_xyz} Spectrum={args.use_spectrum} "
            f"k={args.k_spec} log={args.log_input}"
        )
    elif args.model == "hkswks":
        if not (args.use_xyz or args.n_hks > 0 or args.n_wks > 0):
            raise ValueError("For --model hkswks enable --use_xyz and/or set --n_hks/--n_wks > 0")
        print(
            f"Inputs: XYZ={args.use_xyz} HKS={args.n_hks} WKS={args.n_wks} "
            f"eig_k={args.eig_k} pool={args.pool_mode}"
        )
    else:
        raise RuntimeError(f"Unknown model: {args.model}")

    # ------------------------------------------------------------
    # Dataset + GT
    # ------------------------------------------------------------
    dataset = GTReadyDataset(DATA_DIR)
    subj_map = build_subject_map(dataset)

    D_pack = np.load(DIST_PATH, allow_pickle=True)
    D_orig = D_pack["D_orig"].astype(np.float64)
    D_orig /= np.max(D_orig[D_orig > 0])

    names = [str(n) for n in D_pack["names"]]
    name_to_idx = {
        re.search(r"(id\d{4})", n).group(1): i
        for i, n in enumerate(names)
        if re.search(r"(id\d{4})", n)
    }

    subjects = sorted([s for s in subj_map.keys() if s in name_to_idx])
    if len(subjects) < 6:
        raise RuntimeError(f"Need at least 6 subjects overlapping GT matrix, found {len(subjects)}")

    rng_split = np.random.default_rng(args.seed)
    n_eval = max(3, int(0.2 * len(subjects)))
    n_eval = min(n_eval, len(subjects) - 3)
    eval_subjects = sorted(rng_split.choice(subjects, n_eval, replace=False).tolist())
    train_subjects = sorted([s for s in subjects if s not in set(eval_subjects)])

    # ------------------------------------------------------------
    # Preflight (only meaningful for xyz_spectrum for now)
    # ------------------------------------------------------------
    if args.preflight:
        if args.model != "xyz_spectrum":
            print("\n⚠️  Preflight currently supports only --model xyz_spectrum. Skipping preflight.\n")
        else:
            print("\n🧪 Running preflight checks...")

            sanity = preflight_spectrum_sanity(
                dataset=dataset,
                run_dir=run_dir,
                k_spec=args.k_spec,
                log_input=args.log_input,
                eps=args.eps,
                n_meshes=args.preflight_samples,
                seed=args.seed,
                eps_range=args.preflight_eps_range,
                dead_frac_warn=args.preflight_dead_frac_warn,
                dead_frac_stop=args.preflight_dead_frac_stop,
            )
            print(
                "Preflight spectrum sanity:",
                {k: sanity[k] for k in ["status", "almost_constant_fraction", "almost_constant_channels", "range_p99_p1_median"] if k in sanity},
            )

            if sanity.get("status") == "stop":
                print("\n⛔ PREFLIGHT STOP:", sanity.get("reason", "Unknown reason"))
                print(f"See: {sanity.get('json_path','(missing)')}")
                raise SystemExit(2)

            align = preflight_gt_alignment_baseline(
                dataset=dataset,
                subj_map=subj_map,
                subjects=subjects,
                name_to_idx=name_to_idx,
                D_orig=D_orig,
                run_dir=run_dir,
                k_spec=args.k_spec,
                log_input=args.log_input,
                eps=args.eps,
                n_subjects=args.preflight_subjects,
                seed=args.seed + 123,
            )
            print(
                "Preflight GT alignment (spectrum-only):",
                {k: align.get(k) for k in ["status", "spearman_l2", "spearman_cos", "pearson_l2", "pearson_cos", "n_subjects_used"]},
            )

            if align.get("status") == "ok":
                rho = align.get("spearman_l2", float("nan"))
                if isinstance(rho, (float, int)) and not np.isnan(rho) and abs(rho) < 0.05:
                    print("\n⚠️  PREFLIGHT WARNING: spectrum-only baseline has ~0 correlation with D_orig.")
                    print("   This often means your target is extrinsic or spectrum scaling carries little signal.\n")

    # ------------------------------------------------------------
    # Build model (switch by CLI)
    # ------------------------------------------------------------
    if args.model == "xyz_spectrum":
        model = DiffusionEncoderXYZSpectrum(
            latent_dim=args.latent_dim,
            width=args.width,
            n_blocks=args.n_blocks,
            dropout=args.dropout,
            use_xyz=args.use_xyz,
            use_spectrum=args.use_spectrum,
            k_spec=args.k_spec,
            log_input=args.log_input,
            eps=args.eps,
        ).to(DEVICE)

    elif args.model == "hkswks":
        model = DiffusionEncoderOnlyIntrinsec(
            latent_dim=args.latent_dim,
            width=args.width,
            n_blocks=args.n_blocks,
            dropout=args.dropout,
            use_xyz=args.use_xyz,
            n_hks=args.n_hks,
            n_wks=args.n_wks,
            eig_k=args.eig_k,
            eps=args.eps,
            pool_mode=args.pool_mode,
        ).to(DEVICE)

    else:
        raise RuntimeError(f"Unknown model: {args.model}")

    # ------------------------------------------------------------
    # Optim
    # ------------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3)

    # ------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------
    with open(log_csv, "w", encoding="utf-8") as f:
        f.write(
            "epoch,loss,stress,rank,lr,"
            "spearman_l2,spearman_cos,spearman_l2_unit,spearman_l2_z,"
            "pearson_l2,pearson_cos,pearson_l2_unit,pearson_l2_z,"
            "intra,n_eval\n"
        )

    last_metrics = {
        "spearman_l2": float("nan"),
        "spearman_cos": float("nan"),
        "spearman_l2_unit": float("nan"),
        "spearman_l2_z": float("nan"),
        "pearson_l2": float("nan"),
        "pearson_cos": float("nan"),
        "pearson_l2_unit": float("nan"),
        "pearson_l2_z": float("nan"),
        "intra_mean": float("nan"),
        "n_eval": 0,
    }

    best_spearman = -1e9
    best_epoch = -1
    best_ckpt_path = run_dir / "checkpoints" / "best_by_spearman.pth"

    # ------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = np.random.default_rng(args.seed + 999 + epoch)
        subjects_shuf = rng.permutation(train_subjects)

        epoch_loss = 0.0
        epoch_stress = 0.0
        epoch_rank = 0.0
        n_steps = 0

        pbar = tqdm(
            range(0, len(subjects_shuf), args.batch_subjects),
            desc=f"Epoch {epoch}/{args.epochs}",
            dynamic_ncols=True,
        )

        for start in pbar:
            batch_subjects = subjects_shuf[start : start + args.batch_subjects]
            if len(batch_subjects) < 2:
                continue

            optimizer.zero_grad(set_to_none=True)

            subj_means: List[torch.Tensor] = []
            subj_gt_idx: List[int] = []

            for subj in batch_subjects:
                idxs = pick_mesh_indices(
                    subj_map[subj],
                    args.max_meshes_per_subject_train,
                    rng,
                )
                if not idxs:
                    continue

                z_list: List[torch.Tensor] = []
                for idx in idxs:
                    sample = dataset[int(idx)]
                    Zg = model(
                        sample["verts"].to(DEVICE),
                        sample["mass"].to(DEVICE),
                        sample["L"].to(DEVICE),
                        sample["evals"].to(DEVICE),
                        sample["evecs"].to(DEVICE),
                        sample["faces"].to(DEVICE),
                        sample["gradX"].to(DEVICE),
                        sample["gradY"].to(DEVICE),
                        return_per_vertex=False,
                        add_noise=True,
                    ).squeeze(0)
                    z_list.append(Zg)

                if not z_list:
                    continue

                Zs = torch.stack(z_list, dim=0)
                subj_means.append(Zs.mean(dim=0))
                subj_gt_idx.append(name_to_idx[subj])

            if len(subj_means) < 2:
                continue

            Z_batch = torch.stack(subj_means, dim=0)
            idx_np = np.array(subj_gt_idx, dtype=int)

            D_batch = torch.tensor(
                D_orig[np.ix_(idx_np, idx_np)],
                device=DEVICE,
                dtype=Z_batch.dtype,
            )

            loss_stress = stress_loss(Z_batch, D_batch)

            D_lat = torch.cdist(Z_batch, Z_batch, p=2)
            loss_rank = pairwise_rank_loss(
                D_lat,
                D_batch,
                n_pairs=args.rank_pairs,
                margin=args.rank_margin,
                tau=args.rank_tau,
                hard_frac=args.rank_hard_frac,
            )

            loss = args.lambda_stress * loss_stress + args.lambda_rank * loss_rank

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            loss_item = float(loss.item())
            stress_item = float(loss_stress.item())
            rank_item = float(loss_rank.item())

            epoch_loss += loss_item
            epoch_stress += stress_item
            epoch_rank += rank_item
            n_steps += 1

            pbar.set_postfix(loss=f"{loss_item:.4f}", stress=f"{stress_item:.4f}", rank=f"{rank_item:.4f}")

        if n_steps == 0:
            raise RuntimeError("No valid optimization step in epoch. Check split/settings.")

        epoch_loss /= n_steps
        epoch_stress /= n_steps
        epoch_rank /= n_steps

        scheduler.step(epoch_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        do_eval = (args.eval_every > 0) and (epoch % args.eval_every == 0 or epoch == args.epochs)
        if do_eval:
            metrics = eval_latent_structure(
                model=model,
                dataset=dataset,
                subj_map=subj_map,
                eval_subjects=eval_subjects,
                name_to_idx=name_to_idx,
                D_orig=D_orig,
                max_meshes_per_subject_eval=args.max_meshes_per_subject_eval,
            )
            if metrics is not None:
                last_metrics = metrics

        metrics = last_metrics

        print(f"\nEpoch {epoch}")
        print(
            {
                "loss": epoch_loss,
                "stress": epoch_stress,
                "rank": epoch_rank,
                "lr": lr_now,
                "spearman_l2": metrics["spearman_l2"],
                "spearman_cos": metrics["spearman_cos"],
                "spearman_l2_unit": metrics["spearman_l2_unit"],
                "spearman_l2_z": metrics["spearman_l2_z"],
                "n_eval": metrics["n_eval"],
            }
        )

        # Save best checkpoint by spearman_l2_z
        if do_eval:
            score = float(metrics.get("spearman_l2_z", float("nan")))
            if not np.isnan(score) and score > best_spearman:
                best_spearman = score
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": vars(args),
                        "best_spearman_l2_z": best_spearman,
                    },
                    best_ckpt_path,
                )
                (run_dir / "best_by_spearman.txt").write_text(
                    f"best_epoch={best_epoch}\nbest_spearman_l2_z={best_spearman}\n",
                    encoding="utf-8",
                )
                print(f"🏁 New best spearman_l2_z={best_spearman:.4f} @ epoch {best_epoch} -> {best_ckpt_path}")

        with open(log_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{epoch_loss:.6f},{epoch_stress:.6f},{epoch_rank:.6f},{lr_now:.2e},"
                f"{metrics['spearman_l2']:.6f},{metrics['spearman_cos']:.6f},"
                f"{metrics['spearman_l2_unit']:.6f},{metrics['spearman_l2_z']:.6f},"
                f"{metrics['pearson_l2']:.6f},{metrics['pearson_cos']:.6f},"
                f"{metrics['pearson_l2_unit']:.6f},{metrics['pearson_l2_z']:.6f},"
                f"{metrics['intra_mean']:.6e},{int(metrics['n_eval'])}\n"
            )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = run_dir / "checkpoints" / f"epoch{epoch:03d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                },
                ckpt,
            )

    print("✅ DONE.")
    print(f"Saved in: {run_dir}")
    print(f"Best spearman_l2_z={best_spearman:.4f} at epoch {best_epoch} -> {best_ckpt_path}")


if __name__ == "__main__":
    main()