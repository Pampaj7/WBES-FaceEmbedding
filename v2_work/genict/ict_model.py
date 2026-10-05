"""Minimal ICT-FaceKit loader + shape-only forward, pure numpy.

Why this exists: the official `external/ICT-FaceKit/Scripts/ict_face_model.py`
builds the model through `openmesh`, which has no wheel for the python 3.10 of
the NGC container and does not build there either.  The model itself is a plain
linear blendshape rig stored as OBJ files, so a 60-line parser replaces the
dependency -- no openmesh install, no Blender.

Model (verbatim from `ict_face_model.FaceModel.deform_mesh` +
`face_model_io._DirectoryModelLoader._compute_shape_mode_deltas`):

    V = neutral + sum_i w_i * (identity_i - neutral)
              + sum_j e_j * (expression_j - neutral)

i.e. the shape modes are *deltas* against `generic_neutral_mesh.obj`, and the
weights multiply them directly.  `FaceModel.randomize_identity` draws
`np.random.normal(size=100)`, so the identity weights are unit-sigma: the OBJ
`identityNNN.obj` already are the +1 sigma shapes.

    load_ict(expressions=(...))        -> dict(v_template, shapedirs, exprdirs, f, ...)
    ict_shape_mesh(weights, model)     -> (V (9409,3) float64, F (18460,3) int32)

Face region
-----------
The full ICT topology is 26,719 vertices / 26,384 polygons; the REMESH-2
protocol compares *face patches*, not whole heads (BFM `original` is a
23,470-vertex face crop), so only geometry #0 of the ICT topology is kept:
vertices [0:9408] and polygons [0:9229] per `external/ICT-FaceKit/README.md`.
The crop is index-based and closed by construction -- verified here: the first
9230 polygons never reference a vertex above 9408.

Those 9230 polygons are all quads (checked; the 548 triangles of the full ICT
mesh are all outside the face region).  DiffusionNet needs triangles, so each
quad (a,b,c,d) becomes (a,b,c) + (a,c,d): 18,460 triangles, a fixed index set
identical for every identity, which is what makes `original` a *topology*.

Units are ICT's own (roughly centimetres, head height ~ 24); every downstream
rule that involves a length is scale-relative, so the unit never matters.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

_GENICT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _GENICT_DIR.parents[1]
MODEL_DIR = _REPO_ROOT / "external" / "ICT-FaceKit" / "FaceXModel"

N_TOTAL_VERTS = 26_719  # full ICT topology, README "All"
FACE_N_VERTS = 9_409    # README geometry #0 "Face", vertices [0:9408]
FACE_N_POLYS = 9_230    # README geometry #0 "Face", polygons [0:9229]
FACE_N_TRIS = 18_460    # all 9230 polygons are quads -> 2 triangles each
N_SHAPE = 100           # identity PCA modes shipped with the light model


def read_obj_vertices(path: Path) -> np.ndarray:
    """(N_TOTAL_VERTS, 3) float64 vertex positions of an ICT OBJ."""
    verts = []
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("v "):
                x, y, z = line.split()[1:4]
                verts.append((float(x), float(y), float(z)))
    V = np.asarray(verts, dtype=np.float64)
    if V.shape != (N_TOTAL_VERTS, 3):
        raise ValueError(f"{path.name}: expected {N_TOTAL_VERTS} vertices, got {V.shape}")
    return V


def read_obj_face_triangles(path: Path) -> np.ndarray:
    """(FACE_N_TRIS, 3) int32 triangles of the face region, in ICT vertex indices.

    Reads only the first FACE_N_POLYS polygons and splits every quad along its
    (0,2) diagonal, which is the winding-preserving split.
    """
    tris = []
    with open(path, "r") as fh:
        n_poly = 0
        for line in fh:
            if not line.startswith("f "):
                continue
            if n_poly >= FACE_N_POLYS:
                break
            idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
            if len(idx) != 4:
                raise ValueError(f"{path.name}: polygon {n_poly} is not a quad ({len(idx)} corners)")
            a, b, c, d = idx
            tris.append((a, b, c))
            tris.append((a, c, d))
            n_poly += 1
    F = np.asarray(tris, dtype=np.int32)
    if F.shape != (FACE_N_TRIS, 3):
        raise ValueError(f"{path.name}: expected {FACE_N_TRIS} triangles, got {F.shape}")
    if F.max() >= FACE_N_VERTS:
        raise ValueError(f"{path.name}: face region references vertex {F.max()} >= {FACE_N_VERTS}")
    return F


@lru_cache(maxsize=2)
def _load_ict_cached(model_dir: str, n_shape: int, expressions: tuple[str, ...]) -> dict:
    md = Path(model_dir)
    neutral_path = md / "generic_neutral_mesh.obj"
    if not neutral_path.exists():
        raise FileNotFoundError(
            f"ICT model not found in {md} -- clone it with:\n"
            f"  git clone --depth 1 https://github.com/ICT-VGL/ICT-FaceKit.git external/ICT-FaceKit"
        )

    neutral = read_obj_vertices(neutral_path)[:FACE_N_VERTS]
    f = read_obj_face_triangles(neutral_path)

    # shapedirs laid out (nv, 3, k) so the forward is a single matvec, like flame_model.py
    shapedirs = np.zeros((FACE_N_VERTS, 3, n_shape), dtype=np.float64)
    id_names = []
    for i in range(n_shape):
        name = f"identity{i:03d}"
        shapedirs[:, :, i] = read_obj_vertices(md / f"{name}.obj")[:FACE_N_VERTS] - neutral
        id_names.append(name)

    exprdirs = {}
    for name in expressions:
        exprdirs[name] = read_obj_vertices(md / f"{name}.obj")[:FACE_N_VERTS] - neutral

    return {
        "v_template": neutral,
        "shapedirs": shapedirs,
        "exprdirs": exprdirs,
        "f": f,
        "identity_names": id_names,
        "model_dir": str(md),
    }


def load_ict(model_dir: Path | str = MODEL_DIR, n_shape: int = N_SHAPE,
             expressions: tuple[str, ...] = ()) -> dict:
    """Load the ICT light model restricted to the face region.

    `expressions` names the expression morph targets to read (empty by default:
    reading all 53 costs 53 OBJ parses nobody needs for neutral identities).
    """
    return _load_ict_cached(str(model_dir), int(n_shape), tuple(expressions))


def ict_shape_mesh(weights: np.ndarray, model: dict) -> tuple[np.ndarray, np.ndarray]:
    """Neutral-expression ICT face patch for identity coefficients `weights`."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    n_shape = model["shapedirs"].shape[2]
    if not 0 < len(w) <= n_shape:
        raise ValueError(f"need 1..{n_shape} weights, got {len(w)}")
    V = model["v_template"] + model["shapedirs"][:, :, : len(w)] @ w
    return V, model["f"]


def ict_expression_mesh(weights: np.ndarray, model: dict, shapes: tuple[str, ...],
                        intensity: float) -> tuple[np.ndarray, np.ndarray]:
    """`ict_shape_mesh` plus `intensity` times the sum of the named expression deltas.

    `shapes` is a tuple because ICT splits the symmetric blendshapes per side
    (`mouthSmile_L`/`mouthSmile_R`); driving both with the same weight is what
    makes a symmetric expression.
    """
    V, F = ict_shape_mesh(weights, model)
    V = V.copy()
    for name in shapes:
        if name not in model["exprdirs"]:
            raise KeyError(f"expression '{name}' not loaded (pass it to load_ict)")
        V += float(intensity) * model["exprdirs"][name]
    return V, F


def _self_check() -> None:
    m = load_ict(expressions=("jawOpen",))
    print("shapedirs", m["shapedirs"].shape, "v_template", m["v_template"].shape, "f", m["f"].shape)

    V0, F = ict_shape_mesh(np.zeros(10), m)
    assert np.allclose(V0, m["v_template"]), "zero weights must reproduce the neutral mesh"

    w = np.zeros(10)
    w[0] = 2.0
    V1, _ = ict_shape_mesh(w, m)
    d = np.linalg.norm(V1 - V0, axis=1)
    assert d.mean() > 1e-4, f"w_0=2 barely moved the mesh: {d.mean():.2e}"
    print(f"w0=2 -> mean vertex shift {d.mean() * 10:.2f} mm, max {d.max() * 10:.2f} mm")

    # truncating weights == zero-padding them
    V2, _ = ict_shape_mesh(np.r_[w, np.zeros(90)], m)
    assert np.allclose(V1, V2), "trailing zero weights changed the mesh"

    Ve, _ = ict_expression_mesh(np.zeros(10), m, ("jawOpen",), 1.0)
    de = np.linalg.norm(Ve - V0, axis=1)
    assert de.max() > 1e-3, "jawOpen did not move the mesh"
    print(f"jawOpen@1.0 -> mean vertex shift {de.mean() * 10:.2f} mm, max {de.max() * 10:.2f} mm")

    assert F.dtype == np.int32 and F.max() == FACE_N_VERTS - 1
    print("OK ict_model")


if __name__ == "__main__":
    _self_check()
