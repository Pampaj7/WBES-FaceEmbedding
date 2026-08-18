#!/usr/bin/env python
"""Task 2: how much does GT identity ordering depend on the topology it's measured in?

Compares the two D_GT matrices (per-vertex mean L2, computed by the unmodified
v2_work/genflame/build_flame_gt_matrix.py) over the SAME 500 identities:
  - BFM topology  (23,470 verts, v2_work/xdomain/gt_matrices/bfm_native)
  - FLAME topology (1,930 verts, v2_work/xdomain/gt_matrices/bfm_in_flame)

Usage:
  .conda_env/bin/python v2_work/xdomain/gt_agreement_spearman.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

THIS_DIR = Path(__file__).resolve().parent


def load(path: Path) -> tuple[np.ndarray, list[str]]:
    d = np.load(path, allow_pickle=True)
    names = [str(n).split("_GTready")[0] for n in d["names"]]
    return d["D_orig"].astype(np.float64), names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfm-gt-dir", type=Path, default=THIS_DIR / "gt_matrices" / "bfm_native")
    ap.add_argument("--flame-gt-dir", type=Path, default=THIS_DIR / "gt_matrices" / "bfm_in_flame")
    ap.add_argument("--out", type=Path, default=THIS_DIR / "gt_matrices" / "agreement.json")
    args = ap.parse_args()

    result = {}
    for space in ("raw", "maxabs"):
        Dn, names_n = load(args.bfm_gt_dir / f"flame_matrix_distances_{space}.npz")
        Df, names_f = load(args.flame_gt_dir / f"flame_matrix_distances_{space}.npz")
        assert names_n == names_f, f"identity order mismatch in {space} space"
        n = len(names_n)
        iu = np.triu_indices(n, 1)
        rho = float(spearmanr(Dn[iu], Df[iu]).statistic)
        result[space] = {"n_identities": n, "n_pairs": int(len(iu[0])), "spearman": rho}
        print(f"[{space}] n={n} pairs={len(iu[0])} "
              f"spearman(BFM-topology GT, FLAME-topology GT) = {rho:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
