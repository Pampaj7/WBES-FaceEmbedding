"""Sample synthetic ICT identities and save them as mesh .npz files.

    aau/run.sh v2_work/genict/generate_identities.py --n-identities 5000 --seed 1234

Twin of `v2_work/genflame/generate_identities.py` for ICT-FaceKit, which
replaces FLAME here because the FLAME 2020 pickle is under a licence the author
cannot redistribute (paper/PLAN_CVPR2027.md:50) while ICT-FaceKit is MIT.

Writes `ict0000.npz` ... (keys `V` float32 (9409,3), `F` int32 (18460,3),
`weights`), a single `identity_weights.json` with every weight vector, and a
`manifest.json` with the sampling parameters and the Z1mX distinctness check.

Weights
-------
`ict_face_model.FaceModel.randomize_identity` is `np.random.normal(size=100)`:
plain unit-sigma normals on the 100 PCA modes, with no truncation.  That is
what is used here, so the identity distribution is ICT's own.  The FLAME twin
rejection-samples at +-2.5 sigma; `--trunc` reproduces that if a run ever needs
it, but the default is 0 (off) because ICT's own sampler has no truncation and
the generated set is checked for degenerate triangles anyway.

Frame
-----
The ICT canonical frame is kept as-is.  The FLAME script negates y and z to
feed `v2_work/phase0/render_mesh.py`; that flip was a FLAME-vs-BFM convention,
and nothing in the training/eval path is orientation sensitive (the operators
are intrinsic, the GT distance is per-vertex).  Any renderer work can flip on
read.

Z1mX distinctness check
-----------------------
The reviewer objection Z1mX is "the identities may not be distinct".  Two
minima are recorded: the closest pair in *weight* space (100-D L2) and the
closest pair in *shape* space (mean per-vertex L2 between the neutral meshes,
the same quantity as the GT matrix), each with its 1st percentile and median so
the minimum can be read against the bulk of the distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ict_model import MODEL_DIR, N_SHAPE, ict_shape_mesh, load_ict  # noqa: E402
from pairdist import offdiag_stats, vertex_mean_l2_matrix  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def sample_weights(rng: np.random.Generator, n_identities: int, n_shape: int,
                   sigma: float, trunc: float) -> np.ndarray:
    """(n_identities, n_shape) weights from N(0, sigma), optionally truncated."""
    w = rng.normal(0.0, sigma, size=(n_identities, n_shape))
    if trunc > 0:
        bad = np.abs(w) > trunc * sigma
        while bad.any():  # rejection sampling, per-coefficient
            w[bad] = rng.normal(0.0, sigma, size=int(bad.sum()))
            bad = np.abs(w) > trunc * sigma
    return w


def min_triangle_area(V: np.ndarray, F: np.ndarray) -> float:
    tri = V[F]
    return float(0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).min())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-identities", type=int, default=5000)
    p.add_argument("--n-shape", type=int, default=N_SHAPE)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--trunc", type=float, default=0.0,
                   help="0 = ICT's own untruncated N(0,1); 2.5 = the FLAME twin's rule")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "datasets/ICT/identities")
    p.add_argument("--device", default="auto", help="torch device for the shape-space check")
    p.add_argument("--skip-shape-check", action="store_true")
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    weights = sample_weights(rng, a.n_identities, a.n_shape, a.sigma, a.trunc)

    model = load_ict(a.model_dir, n_shape=a.n_shape)  # fail loudly before writing anything
    a.out_dir.mkdir(parents=True, exist_ok=True)

    verts = np.empty((a.n_identities, len(model["v_template"]), 3), dtype=np.float32)
    worst_area = np.inf
    for i, w in enumerate(weights):
        V, F = ict_shape_mesh(w, model)
        verts[i] = V
        worst_area = min(worst_area, min_triangle_area(V, F))
        np.savez_compressed(a.out_dir / f"ict{i:04d}.npz", V=V.astype(np.float32),
                            F=F.astype(np.int32), weights=w)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{a.n_identities}", flush=True)

    (a.out_dir / "identity_weights.json").write_text(json.dumps(
        {f"ict{i:04d}": [float(x) for x in w] for i, w in enumerate(weights)}) + "\n")

    wd = pdist(weights)
    manifest = {
        "seed": a.seed,
        "sigma": a.sigma,
        "trunc": a.trunc,
        "n_shape": a.n_shape,
        "n_identities": a.n_identities,
        "model_dir": str(a.model_dir),
        "n_verts": int(verts.shape[1]),
        "n_faces": int(len(model["f"])),
        "frame": "ict",
        "min_triangle_area": worst_area,
        "weight_l2": {"min": float(wd.min()), "p1": float(np.percentile(wd, 1)),
                      "median": float(np.median(wd)), "max": float(wd.max())},
    }
    print("weight-space L2: " + json.dumps(manifest["weight_l2"]), flush=True)

    if not a.skip_shape_check:
        print("shape-space vertex-mean-L2 over all pairs...", flush=True)
        D = vertex_mean_l2_matrix(verts, device=a.device)
        manifest["vertex_mean_l2"] = offdiag_stats(D)
        print("shape-space vertex-mean-L2: " + json.dumps(manifest["vertex_mean_l2"]), flush=True)

    (a.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {a.n_identities} identities to {a.out_dir}")


if __name__ == "__main__":
    main()
