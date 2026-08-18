#!/usr/bin/env python
"""Re-express BFM identities in FLAME topology, via the published BFM<->FLAME correspondence.

Why: every GT distance matrix in this repo is per-domain (BFM-space or FLAME-space), so
D_GT(BFM subject, FLAME subject) is undefined -- that forced domain-homogeneous batching
in train_v2.py and left the cross-model transfer matrix disconnected. If the *same* BFM
identity is also expressed in FLAME topology, D_GT(i, j) is defined regardless of which
topology each side is measured in, because it's literally the same identity.

Method (see v2_work/genflame/make_flame_topologies.py module docstring for the full
derivation -- this script imports it unmodified and reuses its crop-index logic and
6-variant generation, it does not reimplement them):

  1. A BFM identity's `original` npz (23,470 verts) is the crop of the 53,215-vertex BFM
     GT mesh selected by WBES/utils/ix_23470_relative_to_53215.txt.
  2. Scatter those 23,470 vertices back into a length-53,215 array (zeros elsewhere).
  3. `BFM_to_FLAME/data/BFM_to_FLAME_corr.npz::BFM2009_cropped_corr['mtx']` is a sparse
     (5023, 2*53215) matrix with `mtx @ [V_bfm_53215; 0] = V_flame_5023` (the second half
     of the matrix, the per-vertex-normal term, is verified all-zero for this entry --
     see `verify_second_half_zero()` below -- so padding it with zeros is exact, not an
     approximation).
  4. The result is a full FLAME-topology mesh (5023 verts) carrying the BFM identity's
     shape. Write it as a `flameNNNN.npz`-shaped staging file and hand it to
     `make_flame_topologies.process_subject` unmodified: it applies the SAME fixed
     1930-vertex/3770-triangle face-region crop used for the synthetic FLAME set, and
     builds the same 6 topology variants (remesh/crop/noisy/down8k/up60k).

Weight-leakage check (must hold for step 2-3 to be exact, not assumed):
  every one of the 500*6=3000 nonzero BFM columns the correspondence matrix uses for the
  KEPT 1930-vertex set must lie inside the 23,470 crop -- otherwise scattering with zeros
  silently drops real signal into those vertices. Checked once, in `verify_weight_leakage`.

Usage:
  .conda_env/bin/python v2_work/xdomain/bfm_to_flame_identity.py \
      --bfm-dir datasets/REMESH/npz_data_topo_500 \
      --out-dir v2_work/xdomain/bfm_in_flame --n-subjects 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import v2_work.genflame.make_flame_topologies as mft  # noqa: E402  (import freely, do not modify)

DEFAULT_BFM_DIR = REPO_ROOT / "datasets" / "REMESH" / "npz_data_topo_500"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "bfm_in_flame"
DEFAULT_STAGING_DIR = Path(__file__).resolve().parent / "_staging_full5023"
# any flame*.npz has the same fixed FLAME topology (F is identity-independent)
DEFAULT_FLAME_TEMPLATE = REPO_ROOT / "v2_work" / "genflame" / "flame_identities" / "flame0000.npz"

N_BFM_VERTS = 53_215


def load_correspondence(corr_path: Path) -> sp.csr_matrix:
    """Position-weight half of BFM2009_cropped_corr: (5023, 53215) sparse."""
    corr = np.load(corr_path, allow_pickle=True, encoding="latin1")
    mtx = corr["BFM2009_cropped_corr"].item()["mtx"].tocsr()
    return mtx[:, :N_BFM_VERTS]


def verify_second_half_zero(corr_path: Path) -> None:
    """The docstring's `mtx @ [V; 0] = V_flame` assumes the normal-weight half is dead."""
    corr = np.load(corr_path, allow_pickle=True, encoding="latin1")
    mtx = corr["BFM2009_cropped_corr"].item()["mtx"]
    second_half = abs(mtx[:, N_BFM_VERTS:]).sum()
    if second_half > 0:
        raise RuntimeError(
            f"BFM2009_cropped_corr normal-weight half is not zero (sum={second_half}); "
            "the zero-pad convention no longer holds, investigate before proceeding."
        )
    print(f"[verify] second-half (normal-weight) mass = {second_half}: zero-pad convention holds")


def verify_weight_leakage(corr_path: Path, index_file: Path) -> None:
    """No FLAME vertex in the final 1930-vertex kept set may draw weight from outside crop."""
    frac = mft.in_crop_weight_fraction(corr_path, index_file)
    with np.load(DEFAULT_FLAME_TEMPLATE) as d:
        faces = np.asarray(d["F"], dtype=np.int32)
    crop_faces = mft.face_crop_faces(faces, corr_path, index_file)
    kept = np.unique(crop_faces)
    leak = 1.0 - frac[kept]
    max_leak = float(leak.max())
    print(f"[verify] {len(kept)} kept FLAME verts, max out-of-crop weight leakage = {max_leak:.3e}")
    if max_leak > 1e-6:
        raise RuntimeError(
            f"kept FLAME vertices draw {max_leak:.3e} weight from outside the BFM crop "
            "(zero-padding would silently distort them) -- investigate before scaling."
        )


def bfm_to_flame_full(V_crop: np.ndarray, mtx_pos: sp.csr_matrix, ix_23470: np.ndarray) -> np.ndarray:
    """Scatter a (23470, 3) BFM crop into 53215 and apply the correspondence -> (5023, 3)."""
    V53215 = np.zeros((N_BFM_VERTS, 3), dtype=np.float64)
    V53215[ix_23470] = V_crop
    return np.asarray(mtx_pos @ V53215)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bfm-dir", type=Path, default=DEFAULT_BFM_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING_DIR)
    ap.add_argument("--corr", type=Path, default=mft.DEFAULT_CORR)
    ap.add_argument("--index-file", type=Path, default=mft.DEFAULT_INDEX_FILE)
    ap.add_argument("--n-subjects", type=int, default=0, help="0 = all")
    ap.add_argument("--n-cores", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    verify_second_half_zero(args.corr)
    verify_weight_leakage(args.corr, args.index_file)

    subjects = sorted(args.bfm_dir.glob("id*_GTready_original.npz"))
    if not subjects:
        raise FileNotFoundError(f"no id*_GTready_original.npz in {args.bfm_dir}")
    if args.n_subjects:
        subjects = subjects[: args.n_subjects]

    mtx_pos = load_correspondence(args.corr)
    ix_23470 = np.loadtxt(args.index_file, dtype=np.int64)
    with np.load(DEFAULT_FLAME_TEMPLATE) as d:
        flame_faces = np.asarray(d["F"], dtype=np.int32)

    args.staging_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for p in subjects:
        subject = p.name.split("_GTready")[0]  # "idNNNN" -- keeps traceability to BFM GT matrix
        staged = args.staging_dir / f"{subject}.npz"
        if args.overwrite or not staged.exists():
            with np.load(p) as d:
                V_crop = np.asarray(d["V"], dtype=np.float64)
            V_flame_full = bfm_to_flame_full(V_crop, mtx_pos, ix_23470)
            np.savez(staged, V=V_flame_full, F=flame_faces)
        tasks.append((str(staged), str(args.out_dir), str(args.corr), str(args.index_file), args.overwrite))

    print(f"{len(tasks)} BFM identities {args.bfm_dir} -> {args.out_dir}  workers={max(1, args.n_cores)}",
          flush=True)

    if args.n_cores > 1:
        import multiprocessing as mp
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        with mp.Pool(processes=args.n_cores) as pool:
            results = list(pool.imap_unordered(mft.process_subject, tasks))
    else:
        results = [mft.process_subject(t) for t in tasks]

    tally = {"[ok]": 0, "[skip]": 0, "[fail]": 0}
    failures = []
    for i, (status, msg) in enumerate(results, start=1):
        tally[status] += 1
        if status == "[fail]":
            failures.append(msg)
        print(f"[{i}/{len(results)}] {status} {msg}", flush=True)

    print(f"\nDone. ok={tally['[ok]']} skip={tally['[skip]']} fail={tally['[fail]']}")
    for msg in failures[:20]:
        print(f"  - {msg}")


if __name__ == "__main__":
    main()
