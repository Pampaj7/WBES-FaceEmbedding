"""Sample synthetic FLAME identities and save them as mesh .npz files.

    .conda_env/bin/python v2_work/genflame/generate_identities.py --n-identities 100

Writes `flame0000.npz` ... (keys `V` float64, `F` int32, `betas`) plus a
`manifest.json` recording the seed, the sampling parameters, the model file and
the minimum pairwise beta L2 distance over the generated set (the dedup check).

Frames
------
`--frame render` (default) stores vertices in the world frame that
`v2_work/phase0/render_mesh.py` expects, i.e. FLAME's y-up/+z-forward canonical
frame negated on y and z.  The renderer was tuned on BFM crops, which are
y-down and face -z; without the flip FLAME heads come out upside down and
back-of-head first.  diag(1,-1,-1) has determinant +1, so triangle winding and
normals stay consistent.  `--frame flame` stores the raw canonical frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

from flame_model import flame_shape_mesh, load_flame, model_path

TRUNC = 2.5  # betas are resampled until every coefficient is within +-TRUNC sigma
FLIP = np.array([1.0, -1.0, -1.0])  # FLAME canonical -> render_mesh.py world frame


def sample_betas(rng: np.random.Generator, n_identities: int, n_shape: int, sigma: float) -> np.ndarray:
    """(n_identities, n_shape) betas from N(0, sigma) truncated at +-TRUNC*sigma."""
    b = rng.normal(0.0, sigma, size=(n_identities, n_shape))
    bad = np.abs(b) > TRUNC * sigma
    while bad.any():  # rejection sampling, per-coefficient
        b[bad] = rng.normal(0.0, sigma, size=int(bad.sum()))
        bad = np.abs(b) > TRUNC * sigma
    return b


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-identities", type=int, default=100)
    p.add_argument("--n-shape", type=int, default=100)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "flame_identities")
    p.add_argument("--gender", default="neutral", choices=("neutral", "male", "female"))
    p.add_argument("--frame", default="render", choices=("render", "flame"),
                   help="'render' negates y,z for v2_work/phase0/render_mesh.py; 'flame' keeps canonical")
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    betas = sample_betas(rng, a.n_identities, a.n_shape, a.sigma)
    flip = FLIP if a.frame == "render" else np.ones(3)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    load_flame(model_path(a.gender))  # fail loudly before writing anything
    for i, b in enumerate(betas):
        V, F = flame_shape_mesh(b, a.gender)
        np.savez_compressed(a.out_dir / f"flame{i:04d}.npz", V=V * flip, F=F, betas=b)

    min_d = float(pdist(betas).min()) if len(betas) > 1 else float("nan")
    manifest = {
        "seed": a.seed,
        "sigma": a.sigma,
        "n_shape": a.n_shape,
        "n_identities": a.n_identities,
        "gender": a.gender,
        "truncation_sigma": TRUNC,
        "frame": a.frame,
        "model_file": str(model_path(a.gender)),
        "min_pairwise_beta_l2": min_d,
    }
    (a.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {a.n_identities} identities to {a.out_dir}  min pairwise beta L2 = {min_d:.3f}")


if __name__ == "__main__":
    main()
