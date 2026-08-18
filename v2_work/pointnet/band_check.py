#!/usr/bin/env python
"""Does a fixed k_eig=128 cover the same band of the spectrum across topologies?

Our encoder runs DiffusionNet with the default diffusion_method='spectral': diffusion is
exp(-lambda_i t) applied in a basis TRUNCATED at a fixed INDEX k=128. Weyl's law says
lambda_k ~ 4*pi*k/A, so at fixed index the cutoff eigenvalue moves with area. A cropped face
loses ~15% of its area, so its 128th mode sits at a higher lambda -- the same 128 modes resolve
FINER detail on the crop than on the original, and the network's learned diffusion times are
applied to a different band of frequencies.

Rodola et al. (Partial Functional Correspondence) make the structural version of this point:
the functional map between a shape and its part has a slanted diagonal whose slope is the area
ratio. Matching by INDEX is the wrong correspondence; matching by BAND is the right one.

This measures the size of the effect before anything is built to fix it.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOPOS = ["original", "crop", "noisy", "down8k", "up60k", "remesh"]
N_ID = 40


def run(d, label):
    ids = sorted({p.name.split("_")[0] for p in d.glob("*_original.npz")})[:N_ID]
    rows = {t: [] for t in TOPOS}
    for i in ids:
        vals = {}
        for t in TOPOS:
            p = d / f"{i}_GTready_{t}.npz"
            if not p.exists():
                break
            vals[t] = float(np.load(p)["evals"][127])
        if len(vals) == len(TOPOS):
            for t in TOPOS:
                rows[t].append(vals[t] / vals["original"])
    print(f"\n--- {label} ({len(rows['original'])} identita') ---")
    print(f"{'topologia':10s} {'lambda_128 / originale':>24s}")
    for t in TOPOS:
        a = np.array(rows[t])
        print(f"{t:10s} {np.median(a):16.4f} +-{a.std():.4f}")


run(ROOT / "datasets/REMESH/npz_data_topo_500_withops", "operatori attuali (coord grezze)")
run(ROOT / "v2_work/potential/bfm_areanorm", "operatori ad area unitaria")
