"""Minimal FLAME loader + shape-only forward, pure numpy.

Why this exists: the official FLAME pickles are python-2, chumpy-laden blobs and
`chumpy` cannot even be imported under numpy 2.x.  Loading is done with a custom
`Unpickler` that maps the three dead names to live ones -- no chumpy install, no
numpy downgrade.

Only the identity part of the model is used.  With zero pose and zero expression
the LBS forward reduces to `v_template + shapedirs[..., :n] @ betas`, so joints,
pose blendshapes and skinning weights are never touched.
`# ponytail: shape-only forward. Add pose/expression (J_regressor + weights are
right there in the pickle) when the benchmark needs non-neutral heads.`

    load_flame(path)                -> dict(v_template, shapedirs, f)
    flame_shape_mesh(betas, gender) -> (V (5023,3) float64, F (9976,3) int32)

FLAME 2020 `shapedirs` is (5023, 3, 400): 300 identity dirs then 100 expression
dirs.  `N_SHAPE` below is that 300 boundary -- betas longer than it would start
bleeding into expression space, so it is asserted, not trusted.
"""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.sparse

_GENFLAME_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _GENFLAME_DIR.parents[1]
# Official FLAME 2020 (verified: bundled BFM_to_FLAME pkls are MODIFIED —
# max |v_template| diff 8.8e-3, shapedirs diff 1.6e-2 vs official; faces equal).
# sha256(official generic_model.pkl) = efcd14cc4a69f3a3d9af8ded80146b5b6b50df3bd74cf69108213b144eba725b
OFFICIAL_MODEL = _GENFLAME_DIR / "official/FLAME2020/generic_model.pkl"
MODEL_DIR = _REPO_ROOT / "BFM_to_FLAME/model/flame"  # legacy/dev-only fallback
N_SHAPE = 300  # identity dirs; shapedirs[..., 300:] is expression


class _Ch:
    """Stand-in for `chumpy.ch.Ch`: absorbs the pickled state, keeps `.x`."""

    def __setstate__(self, state):
        self.__dict__.update(state)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _Ch
        if module.startswith("scipy.sparse.") and name.endswith("_matrix"):
            return getattr(scipy.sparse, name)  # scipy >=1.14 dropped the private modules
        return super().find_class(module, name)


def _dechumpy(v):
    return np.asarray(v.x) if isinstance(v, _Ch) else np.asarray(v)


@lru_cache(maxsize=3)
def load_flame(path: str | Path) -> dict:
    """Load a FLAME .pkl, returning only what a shape-only forward needs."""
    with open(path, "rb") as fh:
        raw = _Unpickler(fh, encoding="latin1").load()

    m = {k: _dechumpy(raw[k]) for k in ("v_template", "shapedirs", "f")}
    m["v_template"] = m["v_template"].astype(np.float64)
    m["shapedirs"] = m["shapedirs"].astype(np.float64)
    m["f"] = m["f"].astype(np.int32)

    nv = len(m["v_template"])
    assert m["v_template"].shape == (nv, 3), m["v_template"].shape
    assert m["shapedirs"].shape[:2] == (nv, 3), m["shapedirs"].shape
    assert m["shapedirs"].shape[2] >= N_SHAPE, f"shapedirs has {m['shapedirs'].shape[2]} dirs"
    assert m["f"].ndim == 2 and m["f"].shape[1] == 3, m["f"].shape
    assert m["f"].max() < nv, "face index out of range"
    return m


def model_path(gender: str = "neutral") -> Path:
    if gender == "neutral" and OFFICIAL_MODEL.exists():
        return OFFICIAL_MODEL
    return MODEL_DIR / f"FLAME_{gender.upper()}.pkl"


def flame_shape_mesh(betas: np.ndarray, gender: str = "neutral") -> tuple[np.ndarray, np.ndarray]:
    """Neutral-pose, neutral-expression FLAME mesh for identity coefficients `betas`."""
    betas = np.asarray(betas, dtype=np.float64).ravel()
    if not 0 < len(betas) <= N_SHAPE:
        raise ValueError(f"need 1..{N_SHAPE} betas, got {len(betas)}")
    m = load_flame(model_path(gender))
    V = m["v_template"] + m["shapedirs"][:, :, : len(betas)] @ betas
    return V, m["f"]


def _self_check() -> None:
    m = load_flame(model_path())
    print("shapedirs", m["shapedirs"].shape, "v_template", m["v_template"].shape, "f", m["f"].shape)

    V0, F = flame_shape_mesh(np.zeros(10))
    assert np.allclose(V0, m["v_template"]), "zero betas must reproduce the template"

    b = np.zeros(10)
    b[0] = 2.0
    V1, _ = flame_shape_mesh(b)
    d = np.linalg.norm(V1 - V0, axis=1)
    assert d.mean() > 1e-4, f"beta_0=2 barely moved the mesh: {d.mean():.2e}"
    print(f"beta0=2 -> mean vertex shift {d.mean() * 1000:.2f} mm, max {d.max() * 1000:.2f} mm")

    # truncating betas == zero-padding them
    V2, _ = flame_shape_mesh(np.r_[b, np.zeros(90)])
    assert np.allclose(V1, V2), "trailing zero betas changed the mesh"

    assert F.dtype == np.int32 and F.max() == len(V0) - 1
    print("OK flame_model")


if __name__ == "__main__":
    _self_check()
