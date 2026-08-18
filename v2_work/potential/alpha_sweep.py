#!/usr/bin/env python
"""Trade-off curve for the potential well's offset, measured WITHOUT training.

The well buys boundary invariance by discarding everything outside radius alpha. Measured on
this data, the paper-faithful alpha=0.21 retains only 2-7% of the face area, so it cannot be
assumed that the invariance is worth its cost -- and a single A/B at one alpha cannot tell us,
because it confounds "the well helps" with "this particular alpha helps".

If the well works at all, it must already show up in the spectrum: no network needed. For each
alpha we build L' = L + U and measure two quantities that pull against each other:

  consistency   mean spectral distance between two TOPOLOGIES of the SAME identity.
                A working well drives this DOWN (that is the whole point).
  separability  mean spectral distance between DIFFERENT identities in the SAME topology.
                A well that has eaten the face drives this down too -- the failure mode.

Neither alone is meaningful: alpha -> 0 sends both to zero (every mesh becomes the same tiny
disc). The informative quantity is their ratio, plus the task metric itself (Spearman against
D_GT over cross-topology pairs), which is what the trained model is ultimately scored on.

    .conda_env/bin/python v2_work/potential/alpha_sweep.py \
        --input-dir datasets/REMESH/npz_data_topo_500 --n-subjects 40 --shard 0/8
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parents[1] / "diffusion-net/src"))

from diffusion_net.geometry import compute_operators  # noqa: E402
from potential_operators import geodesic_from_centre, load_mesh  # noqa: E402

# alpha=None means "no well at all" (the control); the rest span the paper's collection-wide
# value up to the per-mesh placement the first implementation effectively used.
ALPHAS = [None, 0.21, 0.30, 0.40, 0.55, 0.75]
K_EIG = 64
BETA = 100.0
C_REL = 1.0e2


def spectrum(V, F, d, L, mass, base_evals, alpha, scale):
    """Eigenvalues of L + U for one alpha (alpha=None -> plain L)."""
    mass_np = mass.numpy().astype(np.float64)
    Lc = L.coalesce()
    L_sp = sp.coo_matrix(
        (Lc.values().numpy().astype(np.float64),
         (Lc.indices()[0].numpy(), Lc.indices()[1].numpy())), shape=tuple(Lc.shape)).tocsc()
    M = sp.diags(mass_np)
    if alpha is not None:
        well = 1.0 / (1.0 + np.exp(-BETA * (d / scale - alpha)))
        c = C_REL * float(np.clip(base_evals.numpy(), 0, None).max())
        L_sp = L_sp + sp.diags(mass_np * c * well)
    ev = spla.eigsh(L_sp, k=K_EIG, M=M, sigma=-1e-8, which="LM", return_eigenvectors=False)
    return np.sort(np.clip(ev, 0, None))


def descriptor(ev):
    """Scale-free spectral descriptor: log-spectrum normalised by its own mean.

    Without the normalisation the comparison would be dominated by the well's overall energy
    shift (alpha changes lambda_1 by two orders of magnitude here), which says nothing about
    whether the SHAPE of the spectrum became topology-invariant.
    """
    x = np.log(ev[1:] + 1e-12)
    return (x - x.mean()) / (x.std() + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--n-subjects", type=int, default=40)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out-dir", type=Path, default=THIS / "sweep_parts")
    ap.add_argument("--alpha-json", type=Path, default=THIS / "alpha_global.json")
    args = ap.parse_args()

    scale = float(json.loads(args.alpha_json.read_text())["scale"])
    files = sorted(args.input_dir.glob("*.npz"))
    subj = sorted({re.match(r"(.+?)_GTready_", f.name).group(1) for f in files})[:args.n_subjects]
    keep = [f for f in files if re.match(r"(.+?)_GTready_", f.name).group(1) in subj]
    si, sn = (int(t) for t in args.shard.split("/"))
    keep = keep[si::sn]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, list[float]]] = {}
    t0 = time.time()
    for i, f in enumerate(keep):
        V, F = load_mesh(f)
        d, _ = geodesic_from_centre(V, F)
        Vt = torch.tensor(V, dtype=torch.float32)
        Ft = torch.tensor(F, dtype=torch.int32)
        _, mass, L, base_evals, _, _, _ = compute_operators(Vt, Ft, k_eig=K_EIG)
        rec = {}
        for a in ALPHAS:
            key = "plain" if a is None else f"{a:.2f}"
            try:
                rec[key] = descriptor(spectrum(V, F, d, L, mass, base_evals, a, scale)).tolist()
            except Exception as exc:  # noqa: BLE001
                print(f"  {f.name} alpha={key}: {type(exc).__name__}: {exc}", flush=True)
        out[f.stem] = rec
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(keep)}  ({(i+1)/(time.time()-t0):.2f}/s)", flush=True)

    dest = args.out_dir / f"part_{si}_{sn}.json"
    dest.write_text(json.dumps(out))
    print(f"written {dest}  ({len(out)} meshes)")


if __name__ == "__main__":
    main()
