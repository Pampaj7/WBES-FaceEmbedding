#!/usr/bin/env python
"""Ground-truth distance matrix for the ICT half of REMESH-2.

Mirrors the BFM protocol (autoencoder/latent_analysis/compute_gt_distance_matrix_normalized.py)
and its FLAME twin (`v2_work/genflame/build_flame_gt_matrix.py`): per-vertex mean
L2 between the `original` meshes of two identities, exploiting the dense
correspondence that holds by construction within one 3DMM. Two variants are
written so the space mismatch found in checking_assumptions cannot recur silently:

  raw    : distances on the meshes as generated (ICT units)
  maxabs : distances after the per-mesh normalization GTReadyDatasetNPZ applies
           (center on vertex mean, divide by max|coord|) -- the space the benchmark
           metrics actually live in

Output npz keys match the BFM matrix `normalized_matrix_distances.npz` so
downstream loaders work unchanged: `D_orig` (float32, symmetric, zero diagonal,
divided by its own max so the largest pair is exactly 1) and `names`.

The n^2 loop of the FLAME script is replaced by `pairdist.vertex_mean_l2_matrix`
(batched torch, GPU when available): at 5000 identities x 9409 vertices the
numpy version is hours.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pairdist import offdiag_stats, vertex_mean_l2_matrix  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def normalize_maxabs(V: np.ndarray) -> np.ndarray:
    Vn = V - V.mean(axis=0, keepdims=True)
    return Vn / max(float(np.abs(Vn).max()), 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo-dir", type=Path, default=REPO_ROOT / "datasets/ICT/topo")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "datasets/ICT/gt")
    ap.add_argument("--variant", default="original")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    files = sorted(args.topo_dir.glob(f"*_GTready_{args.variant}.npz"))
    if not files:
        raise SystemExit(f"no {args.variant} meshes in {args.topo_dir}")

    names, raw, nrm = [], [], []
    for p in files:
        with np.load(p) as d:
            V = (d["verts"] if "verts" in d else d["V"]).astype(np.float64)
        names.append(p.name.split("_GTready")[0])
        raw.append(V)
        nrm.append(normalize_maxabs(V))
    n = len(names)
    print(f"{n} identities, {raw[0].shape[0]} verts each", flush=True)

    out = {}
    for tag, verts in (("raw", raw), ("maxabs", nrm)):
        D = vertex_mean_l2_matrix(np.stack(verts, axis=0), device=args.device)
        scale = float(D[D > 0].max())
        out[tag] = (D / scale, scale)
        s = offdiag_stats(D)
        print(f"[{tag}] max={s['max']:.6g} mean={np.mean(D[np.triu_indices(n, 1)]):.6g} "
              f"min={s['min']:.6g} p1={s['p1']:.6g}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for tag, (Dn, scale) in out.items():
        np.savez(
            args.out_dir / f"ict_matrix_distances_{tag}.npz",
            D_orig=Dn.astype(np.float32),
            names=np.array(names),
        )

    # cross-space agreement: the confound this file exists to expose
    from scipy.stats import spearmanr
    iu = np.triu_indices(n, 1)
    rho = float(spearmanr(out["raw"][0][iu], out["maxabs"][0][iu]).statistic)
    meta = {
        "n_identities": n,
        "variant": args.variant,
        "normalization_scale": {k: v[1] for k, v in out.items()},
        "spearman_raw_vs_maxabs": rho,
        "closest_pair_frac_of_median": float(
            out["raw"][0][iu].min() / np.median(out["raw"][0][iu])
        ),
        "offdiag_stats_unnormalized": {k: offdiag_stats(v[0] * v[1]) for k, v in out.items()},
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
