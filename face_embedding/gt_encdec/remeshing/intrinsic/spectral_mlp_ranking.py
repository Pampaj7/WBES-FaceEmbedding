#!/usr/bin/env python3
"""
Spectral/XYZ/Hybrid MLP training for subject-ranking experiments.

Modalities per mesh:
  - spectral: lambda_1..lambda_K (exclude lambda_0), optional log
  - xyz: flattened vertex coordinates
  - hybrid: two-branch fusion of xyz and spectral

Training losses:
  - stress (distance matching)
  - ranking hinge (order preservation)
  - identity compactness (intra-subject variance)

Optional diagnostics:
  - GT MDS spectrum analysis
"""

from __future__ import annotations

import argparse
import csv
import json
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


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[4]
AUTOENCODER_DIR = REPO_ROOT / "face_embedding" / "gt_encdec" / "autoencoder"

if str(AUTOENCODER_DIR) not in sys.path:
    sys.path.append(str(AUTOENCODER_DIR))

from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset  # noqa: E402
from latent_loss import stress_loss  # noqa: E402


SUBJECT_RE = re.compile(r"(id\d{4})", re.IGNORECASE)


class SpectralMLP(nn.Module):
    """Single-branch MLP for spectral-only inputs."""

    def __init__(
        self,
        in_dim: int,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        mlp_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        if mlp_layers < 2:
            raise ValueError("mlp_layers must be >= 2")

        layers: List[nn.Module] = []
        cur = in_dim
        for _ in range(mlp_layers - 1):
            layers.append(nn.Linear(cur, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            cur = hidden_dim
        layers.append(nn.Linear(cur, latent_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x_spec: torch.Tensor) -> torch.Tensor:
        return self.net(x_spec)


class MultiModalFusionMLP(nn.Module):
    """
    Two-branch fusion:
      Branch 1: XYZ -> h_xyz
      Branch 2: Spectrum -> h_spec
      Fusion: [h_xyz, h_spec] -> z

    Works also in single-modality mode (xyz-only or spec-only) by enabling one branch.
    """

    def __init__(
        self,
        use_xyz: bool,
        use_spec: bool,
        xyz_dim: int,
        spec_dim: int,
        branch_dim: int = 256,
        fusion_hidden_dim: int = 256,
        latent_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not (use_xyz or use_spec):
            raise ValueError("At least one branch must be enabled")

        self.use_xyz = use_xyz
        self.use_spec = use_spec

        if self.use_xyz:
            self.xyz_branch = nn.Sequential(
                nn.Linear(xyz_dim, branch_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(branch_dim, branch_dim),
                nn.ReLU(inplace=True),
            )

        if self.use_spec:
            self.spec_branch = nn.Sequential(
                nn.Linear(spec_dim, branch_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(branch_dim, branch_dim),
                nn.ReLU(inplace=True),
            )

        fusion_in = branch_dim * int(self.use_xyz) + branch_dim * int(self.use_spec)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(fusion_hidden_dim, latent_dim),
        )

    def forward(
        self,
        x_xyz: Optional[torch.Tensor] = None,
        x_spec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feats: List[torch.Tensor] = []
        if self.use_xyz:
            if x_xyz is None:
                raise ValueError("x_xyz is required when use_xyz=True")
            feats.append(self.xyz_branch(x_xyz))

        if self.use_spec:
            if x_spec is None:
                raise ValueError("x_spec is required when use_spec=True")
            feats.append(self.spec_branch(x_spec))

        h = feats[0] if len(feats) == 1 else torch.cat(feats, dim=1)
        return self.fusion(h)


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
    default_out = THIS_FILE.parent / "runs_spectral_mlp_ranking"

    p = argparse.ArgumentParser(description="Train spectral/xyz/hybrid MLPs with stress/ranking losses.")
    p.add_argument("--data_dir", type=str, default=str(default_data_dir))
    p.add_argument("--dist_npz", type=str, default=str(default_dist))
    p.add_argument("--out_dir", type=str, default=str(default_out))

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max_subjects", type=int, default=300, help="0 = all")
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--eval_subject_cap", type=int, default=96, help="0 = all eval subjects")

    p.add_argument("--input_mode", type=str, default="hybrid", choices=("spectral", "xyz", "hybrid"))

    p.add_argument("--k_spec", type=int, default=100)
    p.add_argument("--log_input", action="store_true")
    p.add_argument("--input_eps", type=float, default=1e-8)

    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=256, help="Used by spectral-only MLP")
    p.add_argument("--mlp_layers", type=int, default=3, help="Used by spectral-only MLP")
    p.add_argument("--branch_dim", type=int, default=256, help="Used by xyz/hybrid branches")
    p.add_argument("--fusion_hidden_dim", type=int, default=256, help="Used by xyz/hybrid fusion")
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_subjects", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--eval_every", type=int, default=1)

    p.add_argument("--max_meshes_per_subject_train", type=int, default=0, help="0 = all")
    p.add_argument("--max_meshes_per_subject_eval", type=int, default=0, help="0 = all")

    # Default to stress-only first, as requested for stable baseline.
    p.add_argument("--loss_mode", type=str, default="stress", choices=("stress", "ranking", "both"))
    p.add_argument("--lambda_stress", type=float, default=0.3)
    p.add_argument("--lambda_rank", type=float, default=0.3)
    p.add_argument("--lambda_id", type=float, default=0.1)

    p.add_argument("--rank_margin", type=float, default=0.0)
    p.add_argument("--rank_gt_eps", type=float, default=1e-8)
    p.add_argument("--rank_max_triplets", type=int, default=20000, help="0 = use all valid triplets")

    p.add_argument("--latent_noise_std", type=float, default=0.0)

    p.add_argument("--run_mds", action="store_true")
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


def split_subjects(
    subjects: Sequence[str],
    val_fraction: float,
    seed: int,
    max_subjects: int,
) -> Tuple[List[str], List[str]]:
    arr = np.array(sorted(subjects), dtype=object)
    rng = np.random.default_rng(seed)

    if max_subjects > 0 and len(arr) > max_subjects:
        pick = rng.choice(len(arr), size=max_subjects, replace=False)
        arr = arr[np.sort(pick)]

    if len(arr) < 6:
        raise ValueError(f"Need at least 6 subjects, found {len(arr)}")

    rng.shuffle(arr)
    n_eval = int(round(val_fraction * len(arr)))
    n_eval = max(3, n_eval)
    n_eval = min(n_eval, len(arr) - 3)

    eval_subj = sorted(arr[:n_eval].tolist())
    train_subj = sorted(arr[n_eval:].tolist())
    return train_subj, eval_subj


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
        raise RuntimeError(f"Could not parse subject ids from names in {path}")
    return D, name_to_idx


def get_spectral_vector(
    evals: torch.Tensor,
    k_spec: int,
    log_input: bool,
    eps: float,
) -> torch.Tensor:
    ev = evals.flatten().float()
    if ev.numel() > 1:
        ev = ev[1:]  # exclude lambda_0

    if ev.numel() >= k_spec:
        x = ev[:k_spec]
    else:
        x = torch.cat([ev, torch.zeros(k_spec - ev.numel(), dtype=ev.dtype)], dim=0)

    if log_input:
        x = torch.log(x.clamp_min(eps))
    return x


def get_xyz_vector(verts: torch.Tensor) -> torch.Tensor:
    return verts.float().reshape(-1)


def sample_mesh_indices(
    idxs: Sequence[int],
    max_meshes: int,
    rng: np.random.Generator,
) -> List[int]:
    if max_meshes <= 0 or len(idxs) <= max_meshes:
        return [int(i) for i in idxs]
    picked = rng.choice(np.asarray(idxs), size=max_meshes, replace=False)
    return [int(i) for i in picked.tolist()]


def precompute_mesh_features(
    dataset: GTReadyDataset,
    subject_map: Dict[str, List[int]],
    subjects: Sequence[str],
    use_xyz: bool,
    use_spec: bool,
    k_spec: int,
    log_input: bool,
    eps: float,
) -> Tuple[Dict[str, List[dict]], int, int, dict]:
    features_by_subject: Dict[str, List[dict]] = {}

    xyz_dim = -1
    spec_dim = k_spec if use_spec else 0

    kept_mesh = 0
    skipped_dim = 0
    skipped_empty = 0

    for sid in tqdm(subjects, desc="Precompute mesh features", dynamic_ncols=True):
        entries: List[dict] = []
        for idx in subject_map[sid]:
            sample = dataset[int(idx)]
            entry: Dict[str, torch.Tensor] = {}

            if use_spec:
                entry["spec"] = get_spectral_vector(
                    evals=sample["evals"],
                    k_spec=k_spec,
                    log_input=log_input,
                    eps=eps,
                ).cpu()

            if use_xyz:
                xyz = get_xyz_vector(sample["verts"]).cpu()
                if xyz_dim < 0:
                    xyz_dim = int(xyz.numel())
                if int(xyz.numel()) != xyz_dim:
                    skipped_dim += 1
                    continue
                entry["xyz"] = xyz

            if len(entry) == 0:
                skipped_empty += 1
                continue

            entries.append(entry)
            kept_mesh += 1

        if entries:
            features_by_subject[sid] = entries

    misc = {
        "kept_meshes": kept_mesh,
        "skipped_dim_mismatch": skipped_dim,
        "skipped_empty": skipped_empty,
    }
    return features_by_subject, xyz_dim, spec_dim, misc


def compute_norm_stats(
    features_by_subject: Dict[str, List[dict]],
    train_subjects: Sequence[str],
    use_xyz: bool,
    use_spec: bool,
    eps: float = 1e-8,
) -> dict:
    stats = {}

    if use_xyz:
        xyz_list: List[torch.Tensor] = []
        for sid in train_subjects:
            xyz_list.extend([e["xyz"] for e in features_by_subject.get(sid, []) if "xyz" in e])
        if not xyz_list:
            raise RuntimeError("No XYZ training features available for normalization")
        X = torch.stack(xyz_list, dim=0)
        mu = X.mean(dim=0)
        std = X.std(dim=0, unbiased=False).clamp_min(eps)
        stats["xyz_mean"] = mu
        stats["xyz_std"] = std

    if use_spec:
        spec_list: List[torch.Tensor] = []
        for sid in train_subjects:
            spec_list.extend([e["spec"] for e in features_by_subject.get(sid, []) if "spec" in e])
        if not spec_list:
            raise RuntimeError("No spectral training features available for normalization")
        X = torch.stack(spec_list, dim=0)
        mu = X.mean(dim=0)
        std = X.std(dim=0, unbiased=False).clamp_min(eps)
        stats["spec_mean"] = mu
        stats["spec_std"] = std

    return stats


def apply_norm_stats(
    features_by_subject: Dict[str, List[dict]],
    stats: dict,
    use_xyz: bool,
    use_spec: bool,
) -> None:
    for sid, entries in features_by_subject.items():
        for e in entries:
            if use_xyz and "xyz" in e:
                e["xyz"] = (e["xyz"] - stats["xyz_mean"]) / stats["xyz_std"]
            if use_spec and "spec" in e:
                e["spec"] = (e["spec"] - stats["spec_mean"]) / stats["spec_std"]


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


def nearest_neighbor_match_rate(D_gt: torch.Tensor, D_emb: torch.Tensor) -> float:
    n = D_gt.shape[0]
    if n < 2:
        return float("nan")
    eye = torch.eye(n, dtype=torch.bool, device=D_gt.device)
    nn_gt = D_gt.masked_fill(eye, float("inf")).argmin(dim=1)
    nn_emb = D_emb.masked_fill(eye, float("inf")).argmin(dim=1)
    return float((nn_gt == nn_emb).float().mean().item())


def ranking_hinge_loss(
    Z: torch.Tensor,
    D_gt: torch.Tensor,
    margin: float = 0.0,
    gt_eps: float = 1e-8,
    max_triplets: int = 0,
) -> torch.Tensor:
    """
    Enforce order consistency:
      if D_gt(i,j) + gt_eps < D_gt(i,k), then D_lat(i,j) + margin < D_lat(i,k)
    """
    n = Z.shape[0]
    if n < 3:
        return torch.tensor(0.0, device=Z.device, dtype=Z.dtype)

    D_lat = torch.cdist(Z, Z, p=2)

    gt_ij = D_gt.unsqueeze(2)  # [i,j,1]
    gt_ik = D_gt.unsqueeze(1)  # [i,1,k]
    order_mask = gt_ij + gt_eps < gt_ik

    eye = torch.eye(n, dtype=torch.bool, device=Z.device)
    i_eq_j = eye.unsqueeze(2)
    i_eq_k = eye.unsqueeze(1)
    j_eq_k = eye.unsqueeze(0)

    valid = order_mask & (~i_eq_j) & (~i_eq_k) & (~j_eq_k)
    if not valid.any():
        return torch.tensor(0.0, device=Z.device, dtype=Z.dtype)

    violations = margin + D_lat.unsqueeze(2) - D_lat.unsqueeze(1)
    active = F.relu(violations[valid])
    if active.numel() == 0:
        return torch.tensor(0.0, device=Z.device, dtype=Z.dtype)

    if max_triplets > 0 and active.numel() > max_triplets:
        perm = torch.randperm(active.numel(), device=active.device)[:max_triplets]
        active = active[perm]

    return active.mean()


def sid_seed(sid: str) -> int:
    return int(sum((i + 1) * ord(c) for i, c in enumerate(sid)) % 1_000_003)


def forward_entries(
    model: nn.Module,
    entries: List[dict],
    input_mode: str,
    device: torch.device,
) -> torch.Tensor:
    if input_mode == "spectral":
        x_spec = torch.stack([e["spec"] for e in entries], dim=0).to(device)
        return model(x_spec)
    if input_mode == "xyz":
        x_xyz = torch.stack([e["xyz"] for e in entries], dim=0).to(device)
        return model(x_xyz=x_xyz, x_spec=None)

    # hybrid
    x_xyz = torch.stack([e["xyz"] for e in entries], dim=0).to(device)
    x_spec = torch.stack([e["spec"] for e in entries], dim=0).to(device)
    return model(x_xyz=x_xyz, x_spec=x_spec)


@torch.inference_mode()
def evaluate_ranking(
    model: nn.Module,
    features_by_subject: Dict[str, List[dict]],
    eval_subjects: Sequence[str],
    gt_matrix: np.ndarray,
    gt_name_to_idx: Dict[str, int],
    input_mode: str,
    device: torch.device,
    max_meshes_per_subject_eval: int,
    seed: int,
) -> Optional[dict]:
    model.eval()

    subj_embs: List[torch.Tensor] = []
    subj_ids: List[str] = []
    intra_vals: List[float] = []

    for sid in eval_subjects:
        if sid not in gt_name_to_idx:
            continue
        feat_list = features_by_subject.get(sid)
        if not feat_list:
            continue

        if max_meshes_per_subject_eval > 0 and len(feat_list) > max_meshes_per_subject_eval:
            rng = np.random.default_rng(seed + sid_seed(sid))
            pick = rng.choice(len(feat_list), size=max_meshes_per_subject_eval, replace=False)
            chosen = [feat_list[int(i)] for i in pick.tolist()]
        else:
            chosen = feat_list

        Zs = forward_entries(model, chosen, input_mode=input_mode, device=device)
        Zm = Zs.mean(dim=0)

        subj_embs.append(Zm)
        subj_ids.append(sid)
        intra_vals.append(float(((Zs - Zm[None, :]) ** 2).mean().item()))

    if len(subj_embs) < 3:
        return None

    Z = torch.stack(subj_embs, dim=0)
    gt_idx = np.array([gt_name_to_idx[s] for s in subj_ids], dtype=int)

    D_gt = torch.tensor(gt_matrix[np.ix_(gt_idx, gt_idx)], device=device, dtype=Z.dtype)
    D_emb = torch.cdist(Z, Z, p=2)

    iu = torch.triu_indices(D_gt.shape[0], D_gt.shape[1], offset=1, device=device)
    gt_vals = D_gt[iu[0], iu[1]].detach().cpu().numpy()
    em_vals = D_emb[iu[0], iu[1]].detach().cpu().numpy()

    return {
        "n_subjects_eval": len(subj_ids),
        "spearman": spearman_corr(gt_vals, em_vals),
        "pearson": pearson_corr(gt_vals, em_vals),
        "nn_match": nearest_neighbor_match_rate(D_gt, D_emb),
        "intra_mean": float(np.mean(intra_vals)) if intra_vals else float("nan"),
    }


def run_gt_mds(
    gt_matrix: np.ndarray,
    gt_name_to_idx: Dict[str, int],
    subjects: Sequence[str],
    out_dir: Path,
) -> dict:
    overlap = [s for s in subjects if s in gt_name_to_idx]
    if len(overlap) < 3:
        return {"status": "skipped", "reason": "insufficient_subjects"}

    idx = np.array([gt_name_to_idx[s] for s in overlap], dtype=int)
    D = gt_matrix[np.ix_(idx, idx)].astype(np.float64)

    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J

    evals = np.linalg.eigvalsh(B)[::-1]
    pos = evals[evals > 0]
    neg = evals[evals < 0]

    pos_sum = float(pos.sum()) if pos.size else 0.0
    neg_abs_sum = float(np.abs(neg).sum()) if neg.size else 0.0

    if pos_sum > 0:
        pos_ratio = pos / pos_sum
        cum = np.cumsum(pos_ratio)
        d90 = int(np.searchsorted(cum, 0.90) + 1)
        d95 = int(np.searchsorted(cum, 0.95) + 1)
    else:
        pos_ratio = np.array([], dtype=np.float64)
        cum = np.array([], dtype=np.float64)
        d90, d95 = 0, 0

    mds_csv = out_dir / "gt_mds_eigenspectrum.csv"
    with open(mds_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["component", "eigenvalue", "is_positive", "pos_var_ratio", "pos_cum_var_ratio"])
        pos_i = 0
        for i, ev in enumerate(evals, start=1):
            is_pos = ev > 0
            if is_pos and pos_i < len(pos_ratio):
                r = float(pos_ratio[pos_i])
                c = float(cum[pos_i])
                pos_i += 1
            else:
                r, c = 0.0, 0.0
            w.writerow([i, float(ev), int(is_pos), r, c])

    return {
        "status": "ok",
        "n_subjects": n,
        "positive_eigs": int((evals > 0).sum()),
        "negative_eigs": int((evals < 0).sum()),
        "positive_var_sum": pos_sum,
        "negative_abs_sum": neg_abs_sum,
        "negative_over_positive": (neg_abs_sum / max(pos_sum, 1e-12)) if pos_sum > 0 else float("nan"),
        "dims_for_90pct_positive_var": d90,
        "dims_for_95pct_positive_var": d95,
        "csv": str(mds_csv),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    use_spec = args.input_mode in ("spectral", "hybrid")
    use_xyz = args.input_mode in ("xyz", "hybrid")

    print(f"Device: {device}")
    print(
        f"Config: input_mode={args.input_mode} use_xyz={use_xyz} use_spec={use_spec} "
        f"k_spec={args.k_spec} log_input={args.log_input} loss_mode={args.loss_mode}"
    )

    dataset = GTReadyDataset(args.data_dir)
    subject_map = build_subject_map(dataset.files)
    if not subject_map:
        raise RuntimeError(f"No valid subject ids parsed from dataset files in {args.data_dir}")

    gt_matrix, gt_name_to_idx = load_gt_distance_matrix(args.dist_npz)
    usable_subjects = sorted([s for s in subject_map.keys() if s in gt_name_to_idx])
    if len(usable_subjects) < 6:
        raise RuntimeError(f"Only {len(usable_subjects)} subjects overlap with GT matrix (need >= 6)")

    train_subjects, eval_subjects_full = split_subjects(
        subjects=usable_subjects,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_subjects=args.max_subjects,
    )

    if args.eval_subject_cap > 0 and len(eval_subjects_full) > args.eval_subject_cap:
        rng_eval = np.random.default_rng(args.seed + 99)
        eval_subjects = sorted(
            rng_eval.choice(eval_subjects_full, size=args.eval_subject_cap, replace=False).tolist()
        )
    else:
        eval_subjects = list(eval_subjects_full)

    all_used_subjects = sorted(set(train_subjects) | set(eval_subjects))

    print(
        f"Subjects: total_overlap={len(usable_subjects)} train={len(train_subjects)} "
        f"eval={len(eval_subjects)}"
    )

    features_by_subject, xyz_dim, spec_dim, precomp_misc = precompute_mesh_features(
        dataset=dataset,
        subject_map=subject_map,
        subjects=all_used_subjects,
        use_xyz=use_xyz,
        use_spec=use_spec,
        k_spec=args.k_spec,
        log_input=args.log_input,
        eps=args.input_eps,
    )

    # Keep only subjects with at least one valid mesh feature.
    train_subjects = [s for s in train_subjects if s in features_by_subject]
    eval_subjects = [s for s in eval_subjects if s in features_by_subject]

    if len(train_subjects) < 3 or len(eval_subjects) < 3:
        raise RuntimeError(
            f"Insufficient subjects after feature precompute: train={len(train_subjects)} eval={len(eval_subjects)}"
        )

    norm_stats = compute_norm_stats(
        features_by_subject=features_by_subject,
        train_subjects=train_subjects,
        use_xyz=use_xyz,
        use_spec=use_spec,
        eps=args.input_eps,
    )
    apply_norm_stats(
        features_by_subject=features_by_subject,
        stats=norm_stats,
        use_xyz=use_xyz,
        use_spec=use_spec,
    )

    if args.input_mode == "spectral":
        model: nn.Module = SpectralMLP(
            in_dim=spec_dim,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            mlp_layers=args.mlp_layers,
            dropout=args.dropout,
        ).to(device)
    else:
        model = MultiModalFusionMLP(
            use_xyz=use_xyz,
            use_spec=use_spec,
            xyz_dim=xyz_dim,
            spec_dim=spec_dim,
            branch_dim=args.branch_dim,
            fusion_hidden_dim=args.fusion_hidden_dim,
            latent_dim=args.latent_dim,
            dropout=args.dropout,
        ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    config_path = out_dir / "config.json"
    cfg = vars(args).copy()
    cfg["use_xyz"] = use_xyz
    cfg["use_spec"] = use_spec
    cfg["xyz_dim"] = int(xyz_dim)
    cfg["spec_dim"] = int(spec_dim)
    cfg["precompute_misc"] = precomp_misc
    if "xyz_mean" in norm_stats:
        cfg["xyz_mean"] = norm_stats["xyz_mean"].tolist()
        cfg["xyz_std"] = norm_stats["xyz_std"].tolist()
    if "spec_mean" in norm_stats:
        cfg["spec_mean"] = norm_stats["spec_mean"].tolist()
        cfg["spec_std"] = norm_stats["spec_std"].tolist()
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    if args.run_mds:
        mds_stats = run_gt_mds(
            gt_matrix=gt_matrix,
            gt_name_to_idx=gt_name_to_idx,
            subjects=all_used_subjects,
            out_dir=out_dir,
        )
        with open(out_dir / "gt_mds_summary.json", "w", encoding="utf-8") as f:
            json.dump(mds_stats, f, indent=2)
        print(f"GT MDS: {mds_stats}")

    log_csv = out_dir / "train_log.csv"
    with open(log_csv, "w", encoding="utf-8") as f:
        f.write("epoch,loss,stress,rank,id,lr,spearman,pearson,nn_match,intra_eval,n_eval\n")

    last_metrics = {
        "spearman": float("nan"),
        "pearson": float("nan"),
        "nn_match": float("nan"),
        "intra_mean": float("nan"),
        "n_subjects_eval": 0,
    }

    for epoch in range(1, args.epochs + 1):
        model.train()

        rng = np.random.default_rng(args.seed + epoch)
        train_perm = rng.permutation(np.array(train_subjects, dtype=object))

        epoch_loss = 0.0
        epoch_stress = 0.0
        epoch_rank = 0.0
        epoch_id = 0.0
        n_steps = 0

        pbar = tqdm(
            range(0, len(train_perm), args.batch_subjects),
            desc=f"Epoch {epoch}/{args.epochs}",
            dynamic_ncols=True,
        )
        for start in pbar:
            batch_subjects = train_perm[start : start + args.batch_subjects].tolist()
            if len(batch_subjects) < 2:
                continue

            subj_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
            for sid in batch_subjects:
                feat_list = features_by_subject.get(sid)
                if not feat_list:
                    continue

                pick_idx = sample_mesh_indices(
                    idxs=list(range(len(feat_list))),
                    max_meshes=args.max_meshes_per_subject_train,
                    rng=rng,
                )
                chosen = [feat_list[i] for i in pick_idx]

                Zs = forward_entries(model, chosen, input_mode=args.input_mode, device=device)
                if args.latent_noise_std > 0:
                    Zs = Zs + args.latent_noise_std * torch.randn_like(Zs)
                Zm = Zs.mean(dim=0)
                subj_stats[sid] = (Zs, Zm)

            if len(subj_stats) < 2:
                continue

            loss_id = torch.tensor(0.0, device=device)
            count_id = 0
            for Zs, Zm in subj_stats.values():
                if Zs.shape[0] < 2:
                    continue
                loss_id = loss_id + ((Zs - Zm.unsqueeze(0)) ** 2).mean()
                count_id += 1
            if count_id > 0:
                loss_id = loss_id / count_id

            subj_means: List[torch.Tensor] = []
            gt_idx: List[int] = []
            for sid in batch_subjects:
                packed = subj_stats.get(sid)
                if packed is None:
                    continue
                i_gt = gt_name_to_idx.get(sid)
                if i_gt is None:
                    continue
                _, Zm = packed
                subj_means.append(Zm)
                gt_idx.append(i_gt)

            if len(subj_means) >= 2:
                Z_batch = torch.stack(subj_means, dim=0)
                idx_np = np.array(gt_idx, dtype=int)
                D_batch = torch.tensor(
                    gt_matrix[np.ix_(idx_np, idx_np)],
                    device=device,
                    dtype=Z_batch.dtype,
                )
            else:
                Z_batch = torch.empty(0, args.latent_dim, device=device)
                D_batch = torch.empty(0, 0, device=device)

            if args.loss_mode in ("stress", "both") and Z_batch.shape[0] >= 2:
                loss_stress = stress_loss(Z_batch, D_batch)
            else:
                loss_stress = torch.tensor(0.0, device=device)

            if args.loss_mode in ("ranking", "both") and Z_batch.shape[0] >= 3:
                loss_rank = ranking_hinge_loss(
                    Z=Z_batch,
                    D_gt=D_batch,
                    margin=args.rank_margin,
                    gt_eps=args.rank_gt_eps,
                    max_triplets=args.rank_max_triplets,
                )
            else:
                loss_rank = torch.tensor(0.0, device=device)

            loss = args.lambda_id * loss_id
            if args.loss_mode in ("stress", "both"):
                loss = loss + args.lambda_stress * loss_stress
            if args.loss_mode in ("ranking", "both"):
                loss = loss + args.lambda_rank * loss_rank

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            n_steps += 1
            epoch_loss += float(loss.item())
            epoch_stress += float(loss_stress.item())
            epoch_rank += float(loss_rank.item())
            epoch_id += float(loss_id.item())

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                stress=f"{loss_stress.item():.4f}",
                rank=f"{loss_rank.item():.4f}",
                ident=f"{loss_id.item():.4f}",
            )

        if n_steps == 0:
            raise RuntimeError("No valid optimization step in epoch. Check split/settings.")

        epoch_loss /= n_steps
        epoch_stress /= n_steps
        epoch_rank /= n_steps
        epoch_id /= n_steps

        scheduler.step(epoch_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        if (epoch % args.eval_every) == 0 or epoch == args.epochs:
            metrics = evaluate_ranking(
                model=model,
                features_by_subject=features_by_subject,
                eval_subjects=eval_subjects,
                gt_matrix=gt_matrix,
                gt_name_to_idx=gt_name_to_idx,
                input_mode=args.input_mode,
                device=device,
                max_meshes_per_subject_eval=args.max_meshes_per_subject_eval,
                seed=args.seed,
            )
            if metrics is not None:
                last_metrics = metrics

        metrics = last_metrics

        print(
            f"Epoch {epoch:03d} | loss={epoch_loss:.5f} stress={epoch_stress:.5f} "
            f"rank={epoch_rank:.5f} id={epoch_id:.5f} "
            f"spearman={metrics['spearman']:.4f} pearson={metrics['pearson']:.4f} "
            f"nn={metrics['nn_match']:.4f} lr={lr_now:.2e}"
        )

        with open(log_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{epoch_loss:.6f},{epoch_stress:.6f},{epoch_rank:.6f},{epoch_id:.6f},{lr_now:.2e},"
                f"{metrics['spearman']:.6f},{metrics['pearson']:.6f},{metrics['nn_match']:.6f},"
                f"{metrics['intra_mean']:.6f},{int(metrics['n_subjects_eval'])}\n"
            )

        if (epoch % args.save_every) == 0 or epoch == args.epochs:
            ckpt = out_dir / f"spectral_mlp_epoch{epoch}.pth"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "config": vars(args),
                },
                ckpt,
            )
            print(f"Saved checkpoint: {ckpt}")

    print("Training completed.")


if __name__ == "__main__":
    main()
