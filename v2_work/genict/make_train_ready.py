#!/usr/bin/env python
"""Expose the ICT half of REMESH-2 under the subject-id convention the trainer parses.

`intrinsic_utils.SUBJECT_RE_ANY` is `(id\\d+)`, so `ict0000_GTready_original.npz` yields
no subject id and the trainer silently sees zero subjects. This script builds a symlink
view named `idNNNN_GTready_<variant>.npz` plus a matching GT matrix, exactly like
`v2_work/genflame/make_train_ready.py` does for FLAME.

Subject ids are offset by 10000 (FLAME uses 1000) so the three 3DMMs occupy disjoint
ranges and a joint training run cannot alias them: BFM id0000-id0499, FLAME
id1000-id5999, ICT id10000-id14999.

Also writes the WS5 split: the first 4500 identities train, the last 500 held out.
The split is by index and not random -- the identities are i.i.d. draws from one
distribution, so a contiguous block is as unbiased as a shuffle and is reproducible
without carrying a second seed around.

Nothing is copied: the withops npz are tens of GB and symlinks are what
`assemble_faceverse_cross_topology_dataset.py` already does elsewhere in this repo.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
NAME_RE = re.compile(r"^ict(?P<num>\d+)_GTready_(?P<variant>.+)\.npz$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--withops-dir", type=Path, default=REPO_ROOT / "datasets/ICT/topo_withops")
    ap.add_argument("--gt-npz", type=Path,
                    default=REPO_ROOT / "datasets/ICT/gt/ict_matrix_distances_maxabs.npz")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "datasets/ICT/train_ready")
    ap.add_argument("--id-offset", type=int, default=10000)
    ap.add_argument("--n-heldout", type=int, default=500)
    args = ap.parse_args()

    data_out = args.out_dir / "npz_withops"
    data_out.mkdir(parents=True, exist_ok=True)

    with np.load(args.gt_npz, allow_pickle=True) as z:
        D = z["D_orig"]
        names = [str(n) for n in z["names"]]
    bad = [n for n in names if not re.fullmatch(r"ict\d+", n)]
    if bad:
        raise SystemExit(f"{len(bad)} GT names are not ictNNNN, e.g. {bad[:3]}")
    # The mapping comes from the GT names, not from the withops glob: this script is run
    # twice -- once right after the GT matrix, to publish the split before the (hour-long)
    # operator job, and once after it, to create the symlinks.
    mapping = {n: f"id{args.id_offset + int(n[3:]):04d}" for n in names}
    renamed = [mapping[n] for n in names]

    n_link = 0
    for p in sorted(args.withops_dir.glob("ict*_GTready_*.npz")):
        m = NAME_RE.match(p.name)
        if not m:
            continue
        sid_new = f"id{args.id_offset + int(m['num']):04d}"
        link = data_out / f"{sid_new}_GTready_{m['variant']}.npz"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(p.resolve())
        n_link += 1

    gt_out = args.out_dir / "gt_matrix.npz"
    np.savez(gt_out, D_orig=D, names=np.array(renamed))

    # WS5 split: contiguous tail held out, by ict index (== sorted order of `renamed`)
    order = sorted(renamed, key=lambda s: int(s[2:]))
    n_train = len(order) - args.n_heldout
    train_ids, heldout_ids = order[:n_train], order[n_train:]
    (args.out_dir / "split_train.txt").write_text("\n".join(train_ids) + "\n")
    (args.out_dir / "split_heldout.txt").write_text("\n".join(heldout_ids) + "\n")
    (args.out_dir / "splits.json").write_text(json.dumps(
        {"train": train_ids, "heldout": heldout_ids}, indent=2) + "\n")

    meta = {
        "source_withops": str(args.withops_dir),
        "source_gt": str(args.gt_npz),
        "id_offset": args.id_offset,
        "n_symlinks": n_link,
        "n_symlinks_expected": len(renamed) * 6,
        "n_subjects": len(renamed),
        "id_range": [order[0], order[-1]],
        "n_train": len(train_ids),
        "n_heldout": len(heldout_ids),
        "heldout_range": [heldout_ids[0], heldout_ids[-1]],
        "note": "symlink view for train_runner.py; ids offset to avoid BFM id0000-id0499 "
                "and FLAME id1000-id5999 collisions",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    if n_link != len(renamed) * 6:
        print(f"ATTENZIONE: {n_link} symlink su {len(renamed) * 6} attesi -- "
              f"operatori mancanti in {args.withops_dir}")


if __name__ == "__main__":
    main()
