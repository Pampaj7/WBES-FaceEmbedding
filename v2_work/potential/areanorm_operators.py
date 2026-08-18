#!/usr/bin/env python
"""Standard DiffusionNet operators computed on AREA-NORMALISED meshes.

Weyl's law gives lambda_k ~ 4*pi*k/A, so removing area shifts every eigenvalue even when the
identity is unchanged: cropping one of our faces costs ~15% of its area and noise inflates it by
128%, and the spectrum moves accordingly. Rodola et al. (Partial Functional Correspondence) make
the same point structurally -- the functional map between a shape and its part has a slanted
diagonal whose slope IS the area ratio -- so partiality does not destroy the spectrum, it
reparametrises it by a factor we can compute.

Measured over 40 identities and the first 30 modes, the mean relative spread of eigenvalues
across topologies of the SAME identity is:

    raw lambda                     0.2202
    lambda * maxabs^2 (pipeline)   0.2100     <- what the loader effectively does today
    lambda * A        (Weyl)       0.0577     <- 3.6x tighter

The pipeline's own convention is therefore no better than doing nothing. This script removes
the effect at the source by scaling each mesh to unit total area before the operators are
built, which is the same thing as reporting lambda*A but keeps every downstream tensor
self-consistent (mass, gradients and eigenvectors all belong to one geometry).

Nothing else changes: no well, no potential, no masking. The only difference from the control
operators is the scale of the mesh they were computed on.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, torch

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parents[1] / "diffusion-net/src"))
from diffusion_net.geometry import compute_operators          # noqa: E402
from potential_operators import load_mesh, save_npz           # noqa: E402


def total_area(V: np.ndarray, F: np.ndarray) -> float:
    tri = V[F]
    return float(0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--k-eig", type=int, default=128)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob("*.npz"))
    si, sn = (int(x) for x in args.shard.split("/"))
    # shard the full sorted list FIRST, then skip what exists: filtering first makes each
    # worker's stripe depend on when it started, which silently leaves files unassigned.
    todo = [p for p in files[si::sn]
            if args.overwrite or not (args.output_dir / p.name).exists()]
    print(f"{len(files)} inputs, {len(todo)} da calcolare (shard {si}/{sn})", flush=True)

    t0 = time.time(); ok = fail = 0
    for i, p in enumerate(todo):
        try:
            V, F = load_mesh(p)
            A = total_area(V, F)
            if not np.isfinite(A) or A <= 0:
                raise ValueError(f"area non valida: {A}")
            V = (V - V.mean(0)) / np.sqrt(A)          # centro + area totale = 1
            Vt = torch.tensor(V, dtype=torch.float32)
            Ft = torch.tensor(F, dtype=torch.int32)
            _, mass, L, evals, evecs, gX, gY = compute_operators(Vt, Ft, k_eig=args.k_eig)
            save_npz(args.output_dir / p.name, V, F, mass, L, evals, evecs, gX, gY)
            ok += 1
        except Exception as exc:                       # noqa: BLE001
            fail += 1
            print(f"  FAIL {p.name}: {type(exc).__name__}: {exc}", flush=True)
        if (i + 1) % 25 == 0:
            r = (i + 1) / max(time.time() - t0, 1e-9)
            print(f"  {i+1}/{len(todo)} ok={ok} fail={fail} ({r:.2f}/s)", flush=True)
    print(f"done ok={ok} fail={fail} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
