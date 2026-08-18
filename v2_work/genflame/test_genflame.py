"""Sanity check for the genflame scaffold.

Run (after `generate_identities.py --n-identities 10`):
    .conda_env/bin/python v2_work/genflame/test_genflame.py

Checks the truncated sampler, that the generated identities are actually
distinct meshes, and renders three of them to `sanity_renders/` for eyeballing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "v2_work/phase0"))

from generate_identities import TRUNC, sample_betas  # noqa: E402
from render_mesh import AXES, render_mesh  # noqa: E402

MESH_DIR = HERE / "flame_identities"
OUT_DIR = HERE / "sanity_renders"


def main() -> None:
    # sampler: truncation is respected and the seed is honoured
    b = sample_betas(np.random.default_rng(0), 200, 100, 1.0)
    assert np.abs(b).max() <= TRUNC, np.abs(b).max()
    assert np.abs(b).max() > 2.0, "truncation at 2.5 sigma but nothing above 2 sigma -- clipped?"
    assert np.allclose(b, sample_betas(np.random.default_rng(0), 200, 100, 1.0)), "seed not reproducible"

    paths = sorted(MESH_DIR.glob("flame*.npz"))
    assert len(paths) >= 10, f"only {len(paths)} meshes in {MESH_DIR}"
    meshes = [np.load(p) for p in paths]
    V = np.stack([m["V"] for m in meshes])
    F = meshes[0]["F"]
    assert V.dtype == np.float64 and F.dtype == np.int32, (V.dtype, F.dtype)

    # every pair of identities is a genuinely different mesh
    dists = [
        np.linalg.norm(V[i] - V[j], axis=1).mean()
        for i in range(len(V))
        for j in range(i + 1, len(V))
    ]
    print(f"pairwise mean vertex distance: min {min(dists) * 1000:.2f} mm, "
          f"median {np.median(dists) * 1000:.2f} mm, max {max(dists) * 1000:.2f} mm")
    assert min(dists) > 1e-4, f"two identities are near-identical: {min(dists):.2e}"

    OUT_DIR.mkdir(exist_ok=True)
    for p, v in list(zip(paths, V))[:3]:
        # Orientation, in the renderer's own screen frame (x right, y up, z toward
        # viewer).  A FLAME head is asymmetric along both of the axes the save-time
        # flip touches: the neck hangs ~0.19 below the centroid while the crown
        # reaches only ~0.13 above it, and the back of the skull is ~0.16 deep
        # against the ~0.08 the face projects forward.  Either flip going missing
        # inverts one of these, so this fires on an upside-down or occiput-first save.
        S = np.stack([sign * v[:, ax] for ax, sign in AXES], axis=1)
        S = S - S.mean(axis=0)
        assert S[:, 1].max() < -S[:, 1].min(), f"{p.stem}: crown/neck inverted -- upside down?"
        assert S[:, 2].max() < -S[:, 2].min(), f"{p.stem}: face/occiput inverted -- back of head to camera?"

        img = render_mesh(v, F, size=384)
        Image.fromarray(img).save(OUT_DIR / f"{p.stem}.png")
        fg = (img[:, :, 0] > 0).mean()
        assert fg > 0.15, f"{p.stem}: only {fg:.1%} foreground"
        assert img[:, :, 0][img[:, :, 0] > 0].std() > 5, f"{p.stem}: flat silhouette, no shading"
        print(f"{p.stem}: fg={fg:.1%} mean={img.mean():.1f}")

    print("OK genflame ->", OUT_DIR)


if __name__ == "__main__":
    main()
