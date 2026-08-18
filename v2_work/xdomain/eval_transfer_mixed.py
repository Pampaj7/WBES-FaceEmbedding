#!/usr/bin/env python
"""The headline mixed cell: one side of every pair in BFM topology, the other in FLAME
topology, same identity set -- the pairing eval_transfer.py cannot express (it pairs
samples drawn from a single data-dir/topology-label space). Sibling script per the task
brief: eval_transfer.py is not edited, this reuses its primitives (dataset, model, GT
loading, correlation) unmodified.

GT reference: the native-BFM GT distance matrix (per-vertex mean L2 in the 23,470-vertex
BFM crop space) is the one canonical D_GT for the shared identity set -- topology of
measurement doesn't change which identity is which, only how much the retopologized
mesh can express about it (that's the cross-topology GT-agreement question task 2
answers separately).

Usage:
  .conda_env/bin/python v2_work/xdomain/eval_transfer_mixed.py \
      --checkpoint <run_dir or .pth> \
      --native-data-dir datasets/REMESH/npz_data_topo_500_withops \
      --flame-data-dir v2_work/xdomain/bfm_in_flame_withops \
      --dist-npz face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz \
      --n-subjects 100 --use-eval-split --tag bfm_mixed_bfm_x_flame
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
INTRINSIC = REPO_ROOT / "face_embedding/gt_encdec/remeshing/intrinsic"
sys.path.insert(0, str(INTRINSIC))
sys.path.insert(0, str(REPO_ROOT / "face_embedding/gt_encdec/autoencoder"))
sys.path.insert(0, str(REPO_ROOT / "diffusion-net/src"))
sys.path.insert(0, str(REPO_ROOT / "v2_work" / "transfer"))

from robustness.data_utils import GTReadyDataset, rebuild_subject_split, sample_to_device  # noqa: E402
from robustness.model_helpers import build_model, forward_model  # noqa: E402
from intrinsic_utils import (  # noqa: E402
    SUBJECT_RE_ANY, build_subject_map, load_gt_distance_matrix, pearson_corr, spearman_corr,
)
from eval_transfer import resolve_checkpoint  # noqa: E402  (import freely, do not modify)


def embed_all(dataset: GTReadyDataset, idxs: list[int], model, device) -> torch.Tensor:
    Z = []
    with torch.inference_mode():
        for idx in idxs:
            s = sample_to_device(dataset[int(idx)], device=device)
            z, _ = forward_model(model=model, sample_dict=s, V_in=s["verts"],
                                 return_gate_info=False, add_noise=False)
            Z.append(z.squeeze(0))
    return torch.stack(Z, dim=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--native-data-dir", type=Path, required=True, help="BFM-native topology withops dir")
    ap.add_argument("--flame-data-dir", type=Path, required=True, help="BFM-in-FLAME topology withops dir")
    ap.add_argument("--dist-npz", type=Path, required=True, help="native-BFM GT matrix (canonical D_GT)")
    ap.add_argument("--n-subjects", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", type=Path, default=THIS_DIR / "results")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--use-eval-split", action="store_true",
                    help="restrict to the training run's held-out subjects -- for an "
                         "apples-to-apples comparison with the native-BFM held-out cell.")
    args = ap.parse_args()

    ckpt_path, cfg = resolve_checkpoint(args.checkpoint)
    device = torch.device(args.device)
    model = build_model(args=SimpleNamespace(**cfg), device=device)
    pack = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = pack["state_dict"] if isinstance(pack, dict) and "state_dict" in pack else pack
    model.load_state_dict(state, strict=True)
    model.eval()

    dsA = GTReadyDataset(str(args.native_data_dir))  # BFM topology
    dsB = GTReadyDataset(str(args.flame_data_dir))   # FLAME topology
    subjA = build_subject_map(dsA.files, subject_re=SUBJECT_RE_ANY)
    subjB = build_subject_map(dsB.files, subject_re=SUBJECT_RE_ANY)
    gt_matrix, name_to_idx = load_gt_distance_matrix(str(args.dist_npz), dtype=np.float64)

    subjects = sorted(set(subjA) & set(subjB) & set(name_to_idx))
    if not subjects:
        raise SystemExit("no subject overlap between the two data dirs and the GT matrix")
    split_note = "random subset"
    if args.use_eval_split:
        _, subjects = rebuild_subject_split(
            subjects=subjects, eval_fraction=0.2, seed=int(cfg.get("seed", args.seed)), max_subjects=0,
        )
        split_note = f"held-out eval split of the training run ({len(subjects)} subjects)"

    rng = np.random.default_rng(args.seed)
    if 0 < args.n_subjects < len(subjects):
        pick = np.sort(rng.choice(len(subjects), size=args.n_subjects, replace=False))
        subjects = [subjects[int(i)] for i in pick]

    idxA = [i for sid in subjects for i in subjA[sid]]
    sidA = [sid for sid in subjects for _ in subjA[sid]]
    idxB = [i for sid in subjects for i in subjB[sid]]
    sidB = [sid for sid in subjects for _ in subjB[sid]]
    print(f"[{args.tag}] subjects={len(subjects)} native_samples={len(idxA)} "
          f"flame_samples={len(idxB)}", flush=True)

    t0 = time.time()
    ZA = embed_all(dsA, idxA, model, device)
    ZB = embed_all(dsB, idxB, model, device)
    embed_seconds = time.time() - t0

    D = torch.cdist(ZA, ZB, p=2).cpu().numpy().astype(np.float64)  # (nA, nB)
    sidA_arr, sidB_arr = np.asarray(sidA), np.asarray(sidB)
    cross_subject_mask = sidA_arr[:, None] != sidB_arr[None, :]

    gt_idx_a = np.asarray([name_to_idx[s] for s in sidA], dtype=int)
    gt_idx_b = np.asarray([name_to_idx[s] for s in sidB], dtype=int)
    gt_full = gt_matrix[np.ix_(gt_idx_a, gt_idx_b)]

    lat = D[cross_subject_mask]
    gt = gt_full[cross_subject_mask]

    out = {
        "tag": args.tag,
        "checkpoint": str(ckpt_path),
        "trained_on": cfg.get("data_dir", "?"),
        "native_data_dir": str(args.native_data_dir),
        "flame_data_dir": str(args.flame_data_dir),
        "dist_npz": str(args.dist_npz),
        "model": cfg.get("model", "?"),
        "pair_mode": "cross_domain_mixed (one side BFM topology, other FLAME topology, "
                     "different identity per pair -- matches the cross_topology protocol "
                     "used elsewhere, generalized to two data dirs)",
        "subject_selection": split_note,
        "n_subjects": int(len(subjects)),
        "n_native_samples": int(len(idxA)),
        "n_flame_samples": int(len(idxB)),
        "n_pairs": int(lat.size),
        "spearman": float(spearman_corr(gt, lat)),
        "pearson": float(pearson_corr(gt, lat)),
        "embed_seconds": round(embed_seconds, 1),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
