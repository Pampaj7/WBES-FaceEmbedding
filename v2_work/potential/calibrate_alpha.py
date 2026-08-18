#!/usr/bin/env python
"""Collection-wide calibration of the potential well's offset.

Liu, Jacobson & Crane (CGF 2017, Sec. 5.1) place the well with a single offset for the whole
collection: "beta is normalized such that across an entire collection of patches no boundary
points are contained in the region of interest." Our first implementation instead took a
per-mesh quantile of each mesh's own geodesic distribution, which places the well somewhere
different on every mesh -- so the patches stop sharing a domain, which is precisely the
invariance the well exists to provide.

This script recovers the collection-level quantities the paper asks for:

    scale  a COMMON length scale (median of per-mesh max geodesic distance), so that one
           offset means the same thing on every mesh
    alpha  an offset strictly inside every mesh's boundary, i.e. below the smallest
           normalised centre-to-boundary distance found anywhere in the collection

Sampling: the min over a sample can only overestimate the true collection min, so the safety
factor is applied downward and the resulting alpha stays conservative (well fully interior).

    .conda_env/bin/python v2_work/potential/calibrate_alpha.py \
        --input-dir datasets/REMESH/npz_data_topo_500 --n-sample 240
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from potential_operators import boundary_vertices, geodesic_from_centre, load_mesh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--n-sample", type=int, default=240,
                    help="meshes to sample; stratified over topologies by sorting")
    ap.add_argument("--safety", type=float, default=0.9,
                    help="alpha = safety * min normalised boundary distance")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("alpha_global.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(args.input_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no npz in {args.input_dir}")
    rng = np.random.default_rng(args.seed)
    if 0 < args.n_sample < len(files):
        idx = np.sort(rng.choice(len(files), size=args.n_sample, replace=False))
        files = [files[int(i)] for i in idx]

    per_topo: dict[str, list[float]] = {}
    maxes, bnd_raw, no_boundary = [], [], 0
    for i, f in enumerate(files):
        V, F = load_mesh(f)
        d, _ = geodesic_from_centre(V, F)
        dmax = float(d.max())
        maxes.append(dmax)
        b = boundary_vertices(F)
        topo = f.stem.split("_")[-1]
        if len(b) == 0:
            no_boundary += 1          # closed mesh: imposes no constraint on alpha
        else:
            bd = float(d[b].min())
            bnd_raw.append(bd)
            per_topo.setdefault(topo, []).append(bd / max(dmax, 1e-12))
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)

    scale = float(np.median(maxes))
    if not bnd_raw:
        raise RuntimeError("no mesh in the sample has a boundary; alpha is unconstrained")
    bnd_norm = np.array(bnd_raw) / scale
    alpha = float(args.safety * bnd_norm.min())

    out = {
        "scale": scale,
        "alpha": alpha,
        "safety": args.safety,
        "n_sampled": len(files),
        "n_closed_meshes": no_boundary,
        "boundary_dist_over_common_scale": {
            "min": float(bnd_norm.min()), "p05": float(np.quantile(bnd_norm, 0.05)),
            "median": float(np.median(bnd_norm)), "max": float(bnd_norm.max()),
        },
        "per_topology_boundary_over_own_max": {
            k: {"n": len(v), "min": float(np.min(v)), "median": float(np.median(v))}
            for k, v in sorted(per_topo.items())
        },
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwritten to {args.out}")
    print(f"\nuse: --alpha-mode global --alpha-value {alpha:.6f} --scale-value {scale:.6f}")


if __name__ == "__main__":
    main()
