#!/usr/bin/env python
"""Per-topology-group evaluation — the measurement the potential-well prediction needs.

`eval_transfer.py` pools every cross-topology pair into one number, which cannot test the
prediction: the well is expected to help where the *boundary* changes and to do nothing (or
slightly hurt) where only the *sampling* changes. Those two effects cancel in a pooled score.

This script therefore reports Spearman vs D_GT separately for:

    crop        pairs where either side is the `crop` realisation (boundary moved)
    noisy       pairs involving `noisy` (surface area changes, boundary intact)
    resample    pairs only among original/remesh/down8k/up60k (sampling changes only)
    all         the pooled number, for continuity with earlier tables

Usage:
    .conda_env/bin/python v2_work/potential/eval_by_topology.py \
        --checkpoint <run dir or .pth> --data-dir <withops> \
        --dist-npz <gt npz> --tag pot_well
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
sys.path.insert(0, str(REPO_ROOT / "face_embedding/gt_encdec/remeshing/intrinsic"))
sys.path.insert(0, str(REPO_ROOT / "face_embedding/gt_encdec/autoencoder"))
sys.path.insert(0, str(REPO_ROOT / "diffusion-net/src"))
sys.path.insert(0, str(REPO_ROOT / "v2_work/transfer"))

from robustness.data_utils import (  # noqa: E402
    GTReadyDataset, build_eval_plan, build_sample_eval_records, sample_to_device,
)
from robustness.model_helpers import build_model, forward_model  # noqa: E402
from intrinsic_utils import (  # noqa: E402
    SUBJECT_RE_ANY, build_subject_map, load_gt_distance_matrix, spearman_corr,
)
from eval_transfer import resolve_checkpoint  # noqa: E402

RESAMPLE = {"original", "remesh", "down8k", "up60k"}


def group_of(ta: str, tb: str) -> str | None:
    """Which reported group a topology pair belongs to (None = same topology, excluded)."""
    if ta == tb:
        return None
    if "crop" in (ta, tb):
        return "crop"
    if "noisy" in (ta, tb):
        return "noisy"
    if ta in RESAMPLE and tb in RESAMPLE:
        return "resample"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--dist-npz", type=Path, required=True)
    ap.add_argument("--n-subjects", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", type=Path, default=THIS_DIR / "results")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--use-eval-split", action="store_true",
                    help="restrict to the run's held-out subjects (required in-domain)")
    ap.add_argument("--masked-pooling", action="store_true",
                    help="REQUIRED for checkpoints trained with --masked-pooling. Masking adds "
                         "no parameters, so a masked checkpoint loads into the unmasked model "
                         "with strict=True and no error -- it would simply be evaluated "
                         "without its mask, silently measuring the wrong model.")
    ap.add_argument("--roi-threshold", type=float, default=0.5)
    ap.add_argument("--point-backbone", action="store_true",
                    help="REQUIRED for checkpoints trained with --point-backbone. The point "
                         "encoder has a different architecture entirely, so loading its "
                         "weights into xyz_dn fails loudly -- but the reverse (scoring a "
                         "DiffusionNet checkpoint while this flag is on) would not, so the "
                         "flag is still the caller's responsibility.")
    ap.add_argument("--point-samples", type=int, default=2048)
    ap.add_argument("--point-knn", type=int, default=20)
    ap.add_argument("--frame", default="current", choices=["current", "rms", "area"],
                    help="REQUIRED to match the frame a checkpoint was trained with. A frame "
                         "mismatch changes no tensor shape and raises no error, so it is the "
                         "caller's responsibility -- the tag is echoed into the result file so "
                         "a mismatch is at least discoverable after the fact.")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "v2_work/pointnet"))
    from frames import reframe as _reframe
    if args.frame != "current":
        print(f"[frame] valutazione con frame {args.frame}", flush=True)

    if args.masked_pooling and args.point_backbone:
        raise SystemExit("--masked-pooling and --point-backbone are mutually exclusive")

    if args.masked_pooling or args.point_backbone:
        sys.path.insert(0, str(REPO_ROOT / "v2_work/fastio"))
        if args.masked_pooling:
            from train_fast import install_masked_pooling
            install_masked_pooling(float(args.roi_threshold))
        else:
            from train_fast import install_point_backbone
            install_point_backbone(int(args.point_samples), int(args.point_knn))
        # This module did `from robustness.model_helpers import build_model, forward_model`
        # at import time, so it holds its own references and the patch above would not reach
        # them. Rebind them here to the patched objects.
        import robustness.model_helpers as _mh
        globals()["build_model"] = _mh.build_model
        globals()["forward_model"] = _mh.forward_model

    ckpt, cfg = resolve_checkpoint(args.checkpoint)
    device = torch.device(args.device)
    model = build_model(args=SimpleNamespace(**cfg), device=device)
    pack = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(pack["state_dict"] if "state_dict" in pack else pack, strict=True)
    model.eval()

    dataset = GTReadyDataset(str(args.data_dir))
    subj_map = build_subject_map(dataset.files, subject_re=SUBJECT_RE_ANY)
    gt, name_to_idx = load_gt_distance_matrix(str(args.dist_npz), dtype=np.float64)
    subjects = sorted(set(subj_map) & set(name_to_idx))
    if args.use_eval_split:
        from robustness.data_utils import rebuild_subject_split
        _, subjects = rebuild_subject_split(subjects=subjects, eval_fraction=0.2,
                                            seed=int(cfg.get("seed", args.seed)), max_subjects=0)
    rng = np.random.default_rng(args.seed)
    if 0 < args.n_subjects < len(subjects):
        pick = np.sort(rng.choice(len(subjects), size=args.n_subjects, replace=False))
        subjects = [subjects[int(i)] for i in pick]

    plan = build_eval_plan(subj_map=subj_map, eval_subjects=subjects,
                           max_meshes_per_subject_eval=0, seed=args.seed)
    records = build_sample_eval_records(dataset=dataset, eval_plan=plan,
                                        eval_subjects=subjects, sample_cache=None)
    print(f"[{args.tag}] {len(subjects)} subjects, {len(records)} meshes", flush=True)

    t0 = time.time()
    Z = []
    with torch.inference_mode():
        for i, rec in enumerate(records):
            s = sample_to_device(dataset[int(rec.dataset_idx)], device=device)
            V_in = _reframe(s["verts"], s["mass"], s["faces"], args.frame)
            z, _ = forward_model(model=model, sample_dict=s, V_in=V_in,
                                 return_gate_info=False, add_noise=False)
            Z.append(z.squeeze(0))
            if (i + 1) % 200 == 0:
                print(f"[{args.tag}] {i+1}/{len(records)} "
                      f"({(i+1)/(time.time()-t0):.1f}/s)", flush=True)
    Z = torch.stack(Z, dim=0).cpu().numpy().astype(np.float64)

    subj = np.array([r.subject_id for r in records], dtype=object)
    topo = np.array([r.topology_label for r in records], dtype=object)
    gt_idx = np.array([name_to_idx[r.subject_id] for r in records], dtype=int)

    iu, ju = np.triu_indices(len(records), 1)
    keep = subj[iu] != subj[ju]                      # cross-subject only
    iu, ju = iu[keep], ju[keep]
    d_lat = np.linalg.norm(Z[iu] - Z[ju], axis=1)
    d_gt = gt[gt_idx[iu], gt_idx[ju]]
    groups = np.array([group_of(a, b) for a, b in zip(topo[iu], topo[ju])], dtype=object)

    out = {"tag": args.tag, "checkpoint": str(ckpt), "data_dir": str(args.data_dir),
           "frame": args.frame, "point_backbone": bool(args.point_backbone),
           "n_subjects": len(subjects), "groups": {}}
    print(f"\n{'group':10s} {'pairs':>9s} {'Spearman':>10s}")
    for g in ("crop", "noisy", "resample", "other"):
        m = groups == g
        if m.sum() < 3:
            continue
        rho = float(spearman_corr(d_gt[m], d_lat[m]))
        out["groups"][g] = {"n_pairs": int(m.sum()), "spearman": rho}
        print(f"{g:10s} {int(m.sum()):9d} {rho:10.4f}")
    m_all = groups != None  # noqa: E711
    rho_all = float(spearman_corr(d_gt[m_all], d_lat[m_all]))
    out["groups"]["all"] = {"n_pairs": int(m_all.sum()), "spearman": rho_all}
    print(f"{'all':10s} {int(m_all.sum()):9d} {rho_all:10.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(f"\nwritten to {args.out_dir / f'{args.tag}.json'}")


if __name__ == "__main__":
    main()
