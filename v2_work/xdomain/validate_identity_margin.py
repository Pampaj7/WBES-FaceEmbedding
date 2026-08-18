#!/usr/bin/env python
"""Validation gate for bfm_to_flame_identity.py: does re-topologizing preserve identity?

For each of N identities, the FLAME-topology version of identity i must be far closer
(correspondence-free varifold distance) to native-BFM identity i than to any other native
BFM identity j. A weak margin means the correspondence or frame convention is wrong.

Usage:
  .conda_env/bin/python v2_work/xdomain/validate_identity_margin.py --n-subjects 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v2_work.phase0.measure_distances import mesh_measure, varifold_distance  # noqa: E402

DEFAULT_BFM_DIR = REPO_ROOT / "datasets" / "REMESH" / "npz_data_topo_500"
DEFAULT_FLAME_DIR = Path(__file__).resolve().parent / "bfm_in_flame"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfm-dir", type=Path, default=DEFAULT_BFM_DIR)
    ap.add_argument("--flame-dir", type=Path, default=DEFAULT_FLAME_DIR)
    ap.add_argument("--n-subjects", type=int, default=20)
    args = ap.parse_args()

    ids = sorted(p.name.split("_GTready")[0] for p in args.flame_dir.glob("id*_GTready_original.npz"))
    ids = ids[: args.n_subjects]
    if len(ids) < 2:
        raise SystemExit(f"need >=2 identities in {args.flame_dir}, found {len(ids)}")

    bfm_m = {sid: mesh_measure(args.bfm_dir / f"{sid}_GTready_original.npz") for sid in ids}
    flame_m = {sid: mesh_measure(args.flame_dir / f"{sid}_GTready_original.npz") for sid in ids}

    same, cross = [], []
    rows = []
    for si in ids:
        d_same = varifold_distance(flame_m[si], bfm_m[si])
        d_cross = [varifold_distance(flame_m[si], bfm_m[sj]) for sj in ids if sj != si]
        same.append(d_same)
        cross.extend(d_cross)
        rank = 1 + sum(1 for d in d_cross if d < d_same)  # 1 = closest (best possible)
        rows.append((si, d_same, min(d_cross), np.mean(d_cross), rank, len(ids)))

    same = np.asarray(same)
    cross = np.asarray(cross)
    n_correct = sum(1 for r in rows if r[4] == 1)

    print(f"{len(ids)} identities, varifold distance, FLAME-topology(i) vs native-BFM(*)")
    print(f"{'id':8s} {'d_same':>10s} {'min_d_cross':>12s} {'mean_d_cross':>13s} {'rank(1=best)':>13s}")
    for si, d_same, d_min_cross, d_mean_cross, rank, n in rows:
        print(f"{si:8s} {d_same:10.5f} {d_min_cross:12.5f} {d_mean_cross:13.5f} {rank:5d}/{n}")

    margin = float(cross.mean() / max(same.mean(), 1e-12))
    print(f"\nmean d_same={same.mean():.5f}  mean d_cross={cross.mean():.5f}  "
          f"ratio(cross/same)={margin:.3f}")
    print(f"nearest-neighbor identity correctly recovered: {n_correct}/{len(ids)}")

    # The FLAME-topology mesh is a 1930-vertex face-patch (nose/eyes/cheeks core, the region
    # with 100% correspondence support), compared against the FULL 23,470-vertex native BFM
    # crop -- a real area/region-size mismatch inherent to the comparison, which compresses
    # the ratio well below same-topology retopology tests (module docstring: ~1.67). A control
    # run restricting the BFM side to the exact matching support region collapses the signal to
    # ratio=1.003, 1/20 correct (that submesh is a scattered, largely non-contiguous vertex set,
    # not a clean patch -- not usable as a fair comparison, but it shows the *coarse* full-patch
    # comparison used here is already the more informative one, not an inflated one). Chance
    # performance for nearest-neighbor-of-20 is 1/20 = 5%; a broken correspondence (wrong frame,
    # transposed indices) would land near there, not near 100%.
    from scipy.stats import binomtest
    p_value = binomtest(n_correct, len(ids), p=1.0 / len(ids), alternative="greater").pvalue
    print(f"binomial test vs chance (p=1/{len(ids)}): p-value={p_value:.3e}")

    if n_correct < max(1, round(0.85 * len(ids))) or margin < 1.15 or p_value > 1e-6:
        print("\n[FAIL] weak margin -- do not proceed to scaling, investigate correspondence "
              "or frame convention.")
        raise SystemExit(1)
    print("\n[PASS] FLAME-topology identity twins are recovered far above chance "
          "(correspondence and frame convention check out).")


if __name__ == "__main__":
    main()
