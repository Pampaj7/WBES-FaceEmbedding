from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Sequence, Tuple, Union

import numpy as np
import torch


SUBJECT_RE_4DIGIT = re.compile(r"(id\d{4})", re.IGNORECASE)
SUBJECT_RE_ANY = re.compile(r"(id\d+)", re.IGNORECASE)


def slugify_token(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_subject_id(name: str, subject_re: Pattern[str] = SUBJECT_RE_4DIGIT) -> Optional[str]:
    m = subject_re.search(str(name))
    return m.group(1).lower() if m else None


def build_subject_map(
    files: Sequence[str],
    subject_re: Pattern[str] = SUBJECT_RE_4DIGIT,
) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for idx, fname in enumerate(files):
        sid = extract_subject_id(str(fname), subject_re=subject_re)
        if sid is None:
            continue
        out.setdefault(sid, []).append(idx)
    return out


def split_subjects(
    subjects: Sequence[str],
    val_fraction: float,
    seed: int,
    max_subjects: int = 0,
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


def load_gt_distance_matrix(
    path: str,
    subject_re: Pattern[str] = SUBJECT_RE_4DIGIT,
    dtype=np.float32,
) -> Tuple[np.ndarray, Dict[str, int]]:
    pack = np.load(path, allow_pickle=True)
    if "D_orig" not in pack or "names" not in pack:
        raise KeyError(f"{path} must contain D_orig and names. Found: {pack.files}")

    D = pack["D_orig"].astype(dtype)
    mask = D > 0
    if mask.any():
        D = D / float(D[mask].max())

    name_to_idx: Dict[str, int] = {}
    for i, n in enumerate(pack["names"]):
        if isinstance(n, bytes):
            n = n.decode("utf-8", errors="ignore")
        sid = extract_subject_id(str(n), subject_re=subject_re)
        if sid is not None:
            name_to_idx[sid] = i

    if not name_to_idx:
        raise RuntimeError(f"Could not parse subject ids from names in {path}")
    return D, name_to_idx


def sample_mesh_indices(
    idxs: Sequence[int],
    max_meshes: int,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> List[int]:
    idx_list = [int(i) for i in idxs]
    if max_meshes <= 0 or len(idx_list) <= max_meshes:
        return idx_list

    if rng is None and seed is not None:
        rng = np.random.default_rng(seed)

    if rng is None:
        return idx_list[:max_meshes]

    picked = rng.choice(np.asarray(idx_list), size=max_meshes, replace=False)
    return [int(i) for i in picked.tolist()]


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
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_corr(rankdata_average_ties(np.asarray(x)), rankdata_average_ties(np.asarray(y)))


def upper_triangular_values(M: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(M, torch.Tensor):
        iu = torch.triu_indices(M.shape[0], M.shape[1], offset=1, device=M.device)
        return M[iu[0], iu[1]]

    arr = np.asarray(M)
    iu = np.triu_indices(arr.shape[0], k=1)
    return arr[iu]


def pairwise_distance_matrix(Z: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "l2":
        return torch.cdist(Z, Z, p=2)
    if mode == "cosine":
        Zn = torch.nn.functional.normalize(Z, dim=1)
        sim = (Zn @ Zn.T).clamp(-1.0, 1.0)
        return (1.0 - sim).clamp_min(0.0)
    raise ValueError(f"Unknown distance mode: {mode}")


def nn_match_rate(D_gt: torch.Tensor, D_emb: torch.Tensor) -> float:
    n = D_gt.shape[0]
    if n < 2:
        return float("nan")
    eye = torch.eye(n, dtype=torch.bool, device=D_gt.device)
    nn_gt = D_gt.masked_fill(eye, float("inf")).argmin(dim=1)
    nn_emb = D_emb.masked_fill(eye, float("inf")).argmin(dim=1)
    return float((nn_gt == nn_emb).float().mean().item())


def spectrum_vector_from_evals(
    evals: torch.Tensor,
    k_spec: int,
    log_input: bool,
    eps: float,
    dtype: torch.dtype,
) -> torch.Tensor:
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


def pairwise_rank_loss(
    D_lat: torch.Tensor,
    D_gt: torch.Tensor,
    n_pairs: int = 2048,
    margin: float = 0.05,
    tau: float = 0.02,
    hard_frac: float = 0.7,
) -> torch.Tensor:
    B = int(D_gt.size(0))
    if B < 3:
        return torch.zeros((), device=D_gt.device, dtype=D_gt.dtype)

    device = D_gt.device
    dtype = D_gt.dtype

    n_hard = int(round(float(n_pairs) * float(hard_frac)))
    n_rand = int(n_pairs) - n_hard

    losses = []

    if n_hard > 0:
        i = torch.randint(0, B, (n_hard,), device=device)
        d = D_gt[i]

        idx = torch.arange(B, device=device).unsqueeze(0)
        self_mask = idx == i.unsqueeze(1)
        d_near = d.masked_fill(self_mask, float("inf"))
        d_far = d.masked_fill(self_mask, float("-inf"))

        k_top = min(3, B - 1)
        j_top = min(3, B - 1)

        k_candidates = torch.topk(d_near, k=k_top, largest=False).indices
        j_candidates = torch.topk(d_far, k=j_top, largest=True).indices

        kk = torch.randint(0, k_top, (n_hard,), device=device)
        jj = torch.randint(0, j_top, (n_hard,), device=device)
        k = k_candidates[torch.arange(n_hard, device=device), kk]
        j = j_candidates[torch.arange(n_hard, device=device), jj]

        j = torch.where(j == k, (j + 1) % B, j)

        dgj = D_gt[i, j]
        dgk = D_gt[i, k]
        dlj = D_lat[i, j]
        dlk = D_lat[i, k]

        mask = dgj > (dgk + tau)
        if mask.any():
            losses.append(torch.relu(margin - (dlj[mask] - dlk[mask])).mean())

    if n_rand > 0:
        i = torch.randint(0, B, (n_rand,), device=device)
        j = torch.randint(0, B, (n_rand,), device=device)
        k = torch.randint(0, B, (n_rand,), device=device)

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

    S = np.stack(specs, axis=0)
    mean = S.mean(axis=0)
    std = S.std(axis=0)
    p1 = np.percentile(S, 1, axis=0)
    p50 = np.percentile(S, 50, axis=0)
    p99 = np.percentile(S, 99, axis=0)
    rng99 = p99 - p1

    const_mask = rng99 < float(eps_range)
    const_frac = float(const_mask.mean())

    csv_path = pre_dir / "preflight_spectrum_stats.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("channel,mean,std,p1,p50,p99,p99_minus_p1,is_almost_constant\n")
        for c in range(S.shape[1]):
            f.write(
                f"{c},{mean[c]:.8e},{std[c]:.8e},{p1[c]:.8e},{p50[c]:.8e},{p99[c]:.8e},{rng99[c]:.8e},{int(const_mask[c])}\n"
            )

    out: Dict[str, object] = {
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
    pre_dir = run_dir / "preflight"
    pre_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    subs = list(subjects)
    if len(subs) > n_subjects:
        subs = rng.choice(np.asarray(subs), size=int(n_subjects), replace=False).tolist()
        subs = sorted([str(s) for s in subs])

    kept: List[str] = []
    emb_rows: List[np.ndarray] = []

    for subj in subs:
        if subj not in subj_map or subj not in name_to_idx:
            continue
        idxs = subj_map[subj]
        if not idxs:
            continue

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

        spec_mean = torch.stack(specs, dim=0).mean(dim=0)
        kept.append(subj)
        emb_rows.append(spec_mean.detach().cpu().numpy())

    if len(kept) < 6:
        out = {"status": "skip", "reason": f"Too few subjects for baseline ({len(kept)})."}
        json_path = pre_dir / "preflight_gt_alignment.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        return out

    E = np.stack(emb_rows, axis=0)
    idx = np.array([name_to_idx[s] for s in kept], dtype=int)
    D_gt = D_orig[np.ix_(idx, idx)]

    iu = np.triu_indices(D_gt.shape[0], 1)
    gt = D_gt[iu]

    diff = E[:, None, :] - E[None, :, :]
    D_l2 = np.sqrt((diff * diff).sum(axis=-1) + 1e-12)
    l2 = D_l2[iu]

    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    D_cos = 1.0 - (En @ En.T)
    cos = D_cos[iu]

    out = {
        "status": "ok",
        "n_subjects_used": int(len(kept)),
        "k_spec": int(k_spec),
        "log_input": bool(log_input),
        "eps": float(eps),
        "pearson_l2": pearson_corr(gt, l2),
        "spearman_l2": spearman_corr(gt, l2),
        "pearson_cos": pearson_corr(gt, cos),
        "spearman_cos": spearman_corr(gt, cos),
        "note": (
            "This measures whether your spectrum-only embedding already aligns with D_orig. "
            "If correlations are ~0, either spectrum is not informative for D_orig or scaling is off."
        ),
    }

    json_path = pre_dir / "preflight_gt_alignment.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    out["json_path"] = str(json_path)
    return out
