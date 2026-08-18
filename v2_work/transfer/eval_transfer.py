#!/usr/bin/env python
"""One cell of the cross-model transfer matrix: a checkpoint evaluated on a domain.

Protocol matches the v1 paper's headline number: cross-topology mesh pairs,
mesh_pair aggregation, Spearman of latent distance against the GT distance matrix.
Runs on CPU (inference only) so it does not compete with training for the GPU.

Usage:
  .conda_env/bin/python v2_work/transfer/eval_transfer.py \
      --checkpoint <run_dir or .pth> --data-dir <withops dir> --dist-npz <gt npz> \
      --n-subjects 100 --tag bfm_model_on_flame
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

from robustness.data_utils import (  # noqa: E402
    GTReadyDataset, build_eval_plan, build_sample_eval_records, sample_to_device,
)
from robustness.eval_utils import build_pair_eval_context, aggregate_pair_observations  # noqa: E402
from robustness.model_helpers import build_model, forward_model  # noqa: E402
from intrinsic_utils import (  # noqa: E402
    SUBJECT_RE_ANY, build_subject_map, load_gt_distance_matrix, pearson_corr, spearman_corr,
)


def resolve_checkpoint(path: Path) -> tuple[Path, dict]:
    """Accept a run dir or an explicit .pth; return (ckpt_path, args_dict)."""
    if path.is_dir():
        for cand in ("best_by_xtopo_mesh_clean.pth", "best_by_clean.pth", "best_by_auc.pth"):
            p = path / "checkpoints" / cand
            if p.exists():
                path = p
                break
        else:
            epochs = sorted(path.glob("checkpoints/epoch*.pth"))
            if not epochs:
                raise SystemExit(f"no checkpoint under {path}")
            path = epochs[-1]
    pack = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(pack.get("args", {})) if isinstance(pack, dict) else {}
    if not cfg:
        run_dir = path.parent.parent if path.parent.name == "checkpoints" else path.parent
        cj = run_dir / "config.json"
        if cj.exists():
            blob = json.loads(cj.read_text())
            cfg = blob.get("args", blob)
    return path, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--dist-npz", type=Path, required=True)
    ap.add_argument("--n-subjects", type=int, default=100)
    ap.add_argument("--pair-mode", default="cross_topology",
                    choices=["cross_topology", "within_topology", "all"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", type=Path, default=THIS_DIR / "results")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--use-eval-split",
        action="store_true",
        help="Restrict to the training run's held-out subjects (rebuild_subject_split, "
             "eval_fraction 0.2, same seed). REQUIRED for in-domain cells: without it a "
             "random subject subset overlaps the training set and the diagonal of the "
             "transfer matrix is inflated. Off-domain cells do not need it.",
    )
    args = ap.parse_args()

    ckpt_path, cfg = resolve_checkpoint(args.checkpoint)
    device = torch.device(args.device)
    model = build_model(args=SimpleNamespace(**cfg), device=device)
    pack = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = pack["state_dict"] if isinstance(pack, dict) and "state_dict" in pack else pack
    model.load_state_dict(state, strict=True)
    model.eval()

    dataset = GTReadyDataset(str(args.data_dir))
    subj_map = build_subject_map(dataset.files, subject_re=SUBJECT_RE_ANY)
    gt_matrix, name_to_idx = load_gt_distance_matrix(str(args.dist_npz), dtype=np.float64)
    subjects = sorted(set(subj_map) & set(name_to_idx))
    if not subjects:
        raise SystemExit("no subject overlap between data dir and GT matrix")
    split_note = "random subset (out-of-domain: no train overlap possible)"
    if args.use_eval_split:
        from robustness.data_utils import rebuild_subject_split

        _, subjects = rebuild_subject_split(
            subjects=subjects, eval_fraction=0.2, seed=int(cfg.get("seed", args.seed)),
            max_subjects=0,
        )
        split_note = f"held-out eval split of the training run ({len(subjects)} subjects)"

    rng = np.random.default_rng(args.seed)
    if 0 < args.n_subjects < len(subjects):
        pick = np.sort(rng.choice(len(subjects), size=args.n_subjects, replace=False))
        subjects = [subjects[int(i)] for i in pick]

    plan = build_eval_plan(subj_map=subj_map, eval_subjects=subjects,
                           max_meshes_per_subject_eval=0, seed=args.seed)
    records = build_sample_eval_records(dataset=dataset, eval_plan=plan,
                                        eval_subjects=subjects, sample_cache=None)
    ctx = build_pair_eval_context(sample_records=records, name_to_idx=name_to_idx,
                                  gt_matrix=gt_matrix, device=device,
                                  pair_mode=args.pair_mode, aggregation_level="mesh_pair")
    print(f"[{args.tag}] subjects={ctx.n_subjects} samples={ctx.n_samples} "
          f"topologies={ctx.n_topology_labels} pairs={ctx.pair_count}", flush=True)

    t0 = time.time()
    Z = []
    with torch.inference_mode():
        for i, rec in enumerate(ctx.sample_records):
            s = sample_to_device(dataset[int(rec.dataset_idx)], device=device)
            z, _ = forward_model(model=model, sample_dict=s, V_in=s["verts"],
                                 return_gate_info=False, add_noise=False)
            Z.append(z.squeeze(0))
            if (i + 1) % 200 == 0:
                print(f"[{args.tag}] embed {i+1}/{ctx.n_samples} "
                      f"({(i+1)/(time.time()-t0):.1f}/s)", flush=True)
    Z = torch.stack(Z, dim=0)

    d = torch.linalg.vector_norm(Z.index_select(0, ctx.pair_i) - Z.index_select(0, ctx.pair_j), dim=1)
    lat = aggregate_pair_observations(d.cpu().numpy().astype(np.float64), pair_ctx=ctx)
    gt = np.asarray(ctx.gt_vals, dtype=np.float64)

    out = {
        "tag": args.tag,
        "checkpoint": str(ckpt_path),
        "trained_on": cfg.get("data_dir", "?"),
        "eval_data_dir": str(args.data_dir),
        "eval_dist_npz": str(args.dist_npz),
        "model": cfg.get("model", "?"),
        "pair_mode": args.pair_mode,
        "subject_selection": split_note,
        "n_subjects": int(ctx.n_subjects),
        "n_samples": int(ctx.n_samples),
        "n_pairs": int(ctx.pair_count),
        "spearman": float(spearman_corr(gt, lat)),
        "pearson": float(pearson_corr(gt, lat)),
        "embed_seconds": round(time.time() - t0, 1),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
