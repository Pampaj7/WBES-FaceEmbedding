#!/usr/bin/env python
"""Ground-truth distance matrix for the FLAME half of REMESH-2.

Mirrors the BFM protocol (autoencoder/latent_analysis/compute_gt_distance_matrix_normalized.py):
per-vertex mean L2 between the `original` meshes of two identities, exploiting the
dense correspondence that holds by construction within one 3DMM. Two variants are
written so the space mismatch found in checking_assumptions cannot recur silently:

  raw    : distances on the meshes as generated (metres, FLAME units)
  maxabs : distances after the per-mesh normalization GTReadyDatasetNPZ applies
           (center on vertex mean, divide by max|coord|) — the space the benchmark
           metrics actually live in

Output npz keys match the BFM matrix so downstream loaders work unchanged:
  D_orig (normalized to max 1), names.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent


def normalize_maxabs(V: np.ndarray) -> np.ndarray:
    Vn = V - V.mean(axis=0, keepdims=True)
    return Vn / max(float(np.abs(Vn).max()), 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo-dir", type=Path, default=THIS_DIR / "flame_topo_600")
    ap.add_argument("--out-dir", type=Path, default=THIS_DIR / "flame_gt_distance_matrix")
    ap.add_argument("--variant", default="original")
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
    print(f"{n} identities, {raw[0].shape[0]} verts each")

    out = {}
    for tag, verts in (("raw", raw), ("maxabs", nrm)):
        V = np.stack(verts, axis=0)                      # (n, nv, 3)
        D = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            diff = V[i][None] - V                        # (n, nv, 3)
            D[i] = np.sqrt((diff ** 2).sum(-1)).mean(-1)
        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)
        scale = float(D[D > 0].max())
        out[tag] = (D / scale, scale)
        iu = np.triu_indices(n, 1)
        v = D[iu]
        print(f"[{tag}] max={scale:.6g} mean={v.mean():.6g} min={v.min():.6g} "
              f"p1={np.percentile(v,1):.6g}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for tag, (Dn, scale) in out.items():
        np.savez(
            args.out_dir / f"flame_matrix_distances_{tag}.npz",
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
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
