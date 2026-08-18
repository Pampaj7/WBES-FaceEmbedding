#!/usr/bin/env python
"""Expose the FLAME half of REMESH-2 under the subject-id convention the trainer parses.

`intrinsic_utils.SUBJECT_RE_ANY` is `(id\\d+)`, so `flame0000_GTready_original.npz` yields
no subject id and the trainer silently sees zero subjects. This script builds a symlink
view named `idNNNN_GTready_<variant>.npz` plus a matching GT matrix.

Subject ids are offset (default 1000) so FLAME ids can never collide with the BFM
id0000-id0499 range — required for the later joint BFM+FLAME training.

Nothing is copied: the withops npz are ~10 GB and symlinks are what
`assemble_faceverse_cross_topology_dataset.py` already does elsewhere in this repo.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
NAME_RE = re.compile(r"^flame(?P<num>\d+)_GTready_(?P<variant>.+)\.npz$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--withops-dir", type=Path, default=THIS_DIR / "flame_topo_600_withops")
    ap.add_argument("--gt-npz", type=Path,
                    default=THIS_DIR / "flame_gt_distance_matrix" / "flame_matrix_distances_maxabs.npz")
    ap.add_argument("--out-dir", type=Path, default=THIS_DIR / "flame_train_ready")
    ap.add_argument("--id-offset", type=int, default=1000)
    args = ap.parse_args()

    data_out = args.out_dir / "npz_withops"
    data_out.mkdir(parents=True, exist_ok=True)

    n_link = 0
    mapping: dict[str, str] = {}
    for p in sorted(args.withops_dir.glob("flame*_GTready_*.npz")):
        m = NAME_RE.match(p.name)
        if not m:
            continue
        sid_new = f"id{args.id_offset + int(m['num']):04d}"
        mapping[f"flame{int(m['num']):04d}"] = sid_new
        link = data_out / f"{sid_new}_GTready_{m['variant']}.npz"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(p.resolve())
        n_link += 1

    with np.load(args.gt_npz, allow_pickle=True) as z:
        D = z["D_orig"]
        names = [str(n) for n in z["names"]]
    renamed = [mapping.get(n, n) for n in names]
    missing = [n for n, r in zip(names, renamed) if n == r]
    if missing:
        raise SystemExit(f"{len(missing)} GT names had no mesh counterpart, e.g. {missing[:3]}")

    gt_out = args.out_dir / "gt_matrix.npz"
    np.savez(gt_out, D_orig=D, names=np.array(renamed))

    meta = {
        "source_withops": str(args.withops_dir),
        "source_gt": str(args.gt_npz),
        "id_offset": args.id_offset,
        "n_symlinks": n_link,
        "n_subjects": len(renamed),
        "id_range": [renamed[0], renamed[-1]],
        "note": "symlink view for train_runner.py; ids offset to avoid BFM id0000-id0499 collision",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
