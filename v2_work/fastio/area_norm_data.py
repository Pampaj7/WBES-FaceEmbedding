"""Area-normalised spectral quantities: make the diffusion time scale intrinsic.

## Why

`GTReadyDatasetNPZ` divides the spectral quantities by the largest computed eigenvalue:

    evals <- evals / lam_max,    L <- L / lam_max,    gradX,gradY <- .. / sqrt(lam_max)

with `lam_max = evals[k_eig-1]`. That makes the inputs dimensionless, but it ties the
scale to a *spectral* quantity, and lam_max is not a property of the surface alone: it
also moves when the surface's total area moves. Measured on one subject across the six
REMESH realisations (k_eig = 128):

    topology   n_verts   lam_128      lam_128 * Area   Area
    original     23470   5.25e-08     1.46e3           2.79e10
    noisy        23470   2.62e-08     1.67e3           6.35e10
    crop         20568   6.12e-08     1.46e3           2.38e10
    remesh       16502   5.39e-08     1.45e3           2.69e10
    down8k        8129   5.21e-08     1.45e3           2.79e10
    up60k        60432   5.32e-08     1.48e3           2.77e10

Across pure retessellation (8k to 60k vertices, a 7x resolution change) lam_128 varies by
only 1.17x — DiffusionNet's discretisation-agnosticism holds. But `noisy` more than
doubles the surface area (jitter creates micro-roughness) and halves lam_128, so the
learned diffusion times `exp(-lam * t)` operate on a 2x different time scale for what is
nominally the same face. The dimensionless product lam * Area agrees to 1.1x across *all*
six, `noisy` included.

## What this does

Renormalise by area instead, so the scale is intrinsic to the surface:

    s = A / C          (A = total area = sum of the lumped mass; C a fixed constant)
    evals <- evals * s,    L <- L * s,    gradX,gradY <- .. * sqrt(s)

Eigenvalues carry units of 1/length^2 and A of length^2, so `evals * A` is dimensionless;
gradients carry 1/length, hence the square root. `C` only sets the working magnitude and
is fixed across the dataset, so it cannot introduce a per-mesh dependence; it defaults to
the measured 1.46e3 so that normalised values land in the same range the network is used
to, which keeps the learned diffusion times comparable to a lam_max-trained run.

This is a hypothesis to be tested, not an established improvement: it predicts a gain on
realisations that change area or boundary (noisy, crop, Poisson re-reconstruction) and no
change on pure retessellation. Train both ways and compare.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "face_embedding/gt_encdec/autoencoder"))

from dataset_gtready import GTReadyDatasetNPZ, scale_sparse_tensor  # noqa: E402

DEFAULT_C = 1.46e3  # measured median of lam_{k_eig} * Area on REMESH


class AreaNormDataset(GTReadyDatasetNPZ):
    """GTReadyDatasetNPZ with the spectral scale set by surface area, not by lam_max.

    Geometry normalisation (centre, divide by max|coord|) is inherited unchanged, so the
    only difference from the v1 dataset is which quantity sets the spectral scale.
    """

    def __init__(self, npz_data_dir, area_const: float = DEFAULT_C, verbose: bool = False):
        super().__init__(npz_data_dir, verbose=verbose)
        self.area_const = float(area_const)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = super().__getitem__(idx)  # lam_max-normalised
        # undo the lam_max scaling, then apply the area scaling. lam_max is recovered from
        # the file rather than inferred, so this cannot silently drift if the base class
        # changes its convention.
        with np.load(Path(self.data_dir) / self.files[int(idx)], allow_pickle=False) as z:
            lam_max = float(np.clip(np.asarray(z["evals"], dtype=np.float64), 0.0, None).max()) + 1e-9
            area = float(np.asarray(z["mass"], dtype=np.float64).sum())

        factor = lam_max * (area / self.area_const)  # undo /lam_max, apply *A/C
        s = dict(s)
        s["evals"] = s["evals"] * factor
        s["L"] = scale_sparse_tensor(s["L"], 1.0 / factor)
        root = float(np.sqrt(factor))
        s["gradX"] = scale_sparse_tensor(s["gradX"], 1.0 / root)
        s["gradY"] = scale_sparse_tensor(s["gradY"], 1.0 / root)
        s["area"] = torch.tensor(area, dtype=torch.float32)
        return s


def install(data_dir: str, area_const: float = DEFAULT_C) -> AreaNormDataset:
    """Rebind the frozen v1 modules' `GTReadyDataset` name to the area-normalised one."""
    import robustness.data_utils as du
    import robustness.train_runner as tr

    ds = AreaNormDataset(data_dir, area_const=area_const)
    factory = lambda *_a, **_k: ds  # noqa: E731
    du.GTReadyDataset = factory
    tr.GTReadyDataset = factory
    return ds


def demo() -> None:
    """The point of the change, as an assertion: the spectral scale must stop tracking area."""
    d = REPO_ROOT / "datasets/REMESH/npz_data_topo_500_withops"
    base = GTReadyDatasetNPZ(str(d))
    area_ds = AreaNormDataset(str(d))
    want = ["id0000_GTready_original.npz", "id0000_GTready_noisy.npz",
            "id0000_GTready_down8k.npz", "id0000_GTready_up60k.npz"]
    idx = {f: base.files.index(f) for f in want if f in base.files}
    assert len(idx) == 4, f"missing meshes: {set(want) - set(idx)}"

    print(f"{'topology':10s} {'lam_max(v1 norm)':>17s} {'lam_max(area norm)':>19s}")
    v1, v2 = [], []
    for f, i in idx.items():
        a = float(base[i]["evals"].max())
        b = float(area_ds[i]["evals"].max())
        v1.append(a); v2.append(b)
        print(f"{f.split('_')[-1][:-4]:10s} {a:17.4f} {b:19.4f}")

    spread_v1 = max(v1) / min(v1)
    spread_v2 = max(v2) / min(v2)
    print(f"\nspread across topologies: v1 {spread_v1:.2f}x   area-normalised {spread_v2:.2f}x")
    # v1 normalises every mesh to exactly 1.0, which hides the disagreement rather than
    # fixing it: the **relative** spectrum is then scaled by a topology-dependent factor.
    # Under area normalisation the top eigenvalue is allowed to differ, and what must hold
    # is that it differs little.
    assert spread_v2 < 1.3, f"area normalisation did not tighten the scale: {spread_v2:.2f}x"
    print("OK: under area normalisation the spectral scale agrees across topologies")


if __name__ == "__main__":
    demo()
