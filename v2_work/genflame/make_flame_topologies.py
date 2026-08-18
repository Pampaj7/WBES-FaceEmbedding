"""Build the 6 REMESH-2 topology variants for a directory of FLAME identities.

    .conda_env/bin/python v2_work/genflame/make_flame_topologies.py \
        --in-dir  v2_work/genflame/flame_identities_600 \
        --out-dir v2_work/genflame/flame_topo_600 --n-cores 1

Input:  `flameNNNN.npz` with keys `V` (5023,3) / `F` (9976,3) from
        `generate_identities.py` (any frame; the crop is index-based).
Output: `flameNNNN_GTready_<variant>.npz` with keys `V`/`F`, variant in
        {original, remesh, crop, noisy, down8k, up60k} -- the same six names and
        the same `<subject>_GTready_<topology>` convention as the BFM half in
        `datasets/REMESH/npz_data_topo_500`, because downstream code parses it.

The face-region crop (defines `original`)
----------------------------------------
A full FLAME head must not be paired with a BFM *face patch*: the BFM
`original` is the 23,470-vertex crop of the 53,215-vertex GT mesh selected by
`WBES/utils/ix_23470_relative_to_53215.txt`.  Instead of inventing a FLAME
crop (nose-tip radius or similar), the BFM crop region is *carried over through
the published BFM->FLAME correspondence*, which is the principled choice:

`BFM_to_FLAME/data/BFM_to_FLAME_corr.npz::BFM2009_cropped_corr` holds a sparse
matrix `mtx` (5023 x 2*53215) such that `mtx @ [V_bfm; 0] = V_flame`, i.e. every
FLAME vertex is a fixed linear (barycentric) combination of BFM 53,215-mesh
vertices -- exactly the mesh our GT crop indexes into.  A FLAME vertex is kept
iff **all** of its non-zero position weights land on vertices inside the BFM
23,470 crop set, so a kept FLAME vertex is one the BFM patch fully determines.
The support is barycentric (~3 weights per row), so the criterion is sharp: the
kept count is 3022 at a >=0.99 in-crop weight fraction and only 3037 at >=0.25.

FLAME carries two eyeball spheres inside the head; the raw kept set is three
connected components (3770 + 1088 + 1088 triangles), so only the largest
triangle component is kept.  BFM has no eyeballs, and the resulting silhouette
matches the BFM patch closely (forehead notch included).

Result: a fixed 1930-vertex / 3770-triangle index set, identical for every
identity -- it depends only on the correspondence matrix and the index file, not
on the shape coefficients.  That is what makes it a *topology*.

Variant semantics
-----------------
`crop`, `down8k`, `up60k` are the imported BFM functions, unmodified.  `remesh`
and `noisy` are reproduced here from the BFM generator that made them (git
b8adab3 `datasets/remesh.py`, since rewritten to only regenerate `crop`):
smoothing + 0.7x decimation, and Gaussian vertex noise at 0.003 x bbox diagonal.
Both rules are scale-relative, so they transfer to FLAME's metre-scale units
untouched.

The one thing that cannot transfer is the *absolute* triangle target of
`down8k`/`up60k` (16k / 120k), tuned to BFM's 46,440-triangle patch: FLAME's
patch has 3,770 triangles, so those targets would make both variants no-ops and
the FLAME half of the benchmark degenerate.  The invariant kept instead is the
resolution *ratio* to each model's own `original` (0.345x and 2.58x); the
targets are scaled by 3770/46440 before calling the imported functions.  The
variant names stay, misnomers and all, because downstream code parses them.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import expand_remesh_topologies as expand  # noqa: E402
from datasets.remesh import make_crop  # noqa: E402

DEFAULT_CORR = REPO_ROOT / "BFM_to_FLAME" / "data" / "BFM_to_FLAME_corr.npz"
DEFAULT_INDEX_FILE = REPO_ROOT / "WBES" / "utils" / "ix_23470_relative_to_53215.txt"

N_BFM_VERTS = 53_215  # the GT mesh the crop index file is relative to
BFM_ORIGINAL_TRIANGLES = 46_440  # triangles of a BFM *_GTready_original patch
BFM_DOWN_TARGET = expand.DOWN_TARGET_TRIANGLES  # snapshot: scale_triangle_targets overwrites these
BFM_UP_TARGET = expand.UP_TARGET_TRIANGLES
IN_CROP_WEIGHT_FRACTION = 0.99  # keep a FLAME vertex if this much of its support is in-crop
REMESH_SMOOTH_ITERS = 2  # b8adab3 datasets/remesh.py::make_remesh
REMESH_DECIMATION = 0.7
REMESH_MIN_TRIANGLES = 2_000
NOISE_STD_BBOX_FRACTION = 0.003  # b8adab3 datasets/remesh.py::make_noisy

VARIANTS = ("original", "remesh", "crop", "noisy", "down8k", "up60k")


@lru_cache(maxsize=2)
def in_crop_weight_fraction(corr_path: Path, index_file: Path) -> np.ndarray:
    """Per-FLAME-vertex fraction of BFM position weight landing inside the BFM crop."""
    corr = np.load(corr_path, allow_pickle=True, encoding="latin1")
    mtx = abs(corr["BFM2009_cropped_corr"].item()["mtx"].tocsr()[:, :N_BFM_VERTS])
    in_crop = np.zeros(N_BFM_VERTS)
    in_crop[np.loadtxt(index_file, dtype=np.int64)] = 1.0
    total = np.asarray(mtx.sum(axis=1)).ravel()
    return (mtx @ in_crop) / np.maximum(total, 1e-30)


def face_crop_faces(faces: np.ndarray, corr_path: Path, index_file: Path) -> np.ndarray:
    """FLAME triangles of the BFM-corresponded face region, in original vertex indices.

    See the module docstring for the criterion.  Depends only on the FLAME
    topology, the correspondence matrix and the BFM crop index file, so it is
    the same set for every identity.
    """
    keep = in_crop_weight_fraction(corr_path, index_file) >= IN_CROP_WEIGHT_FRACTION

    kept = np.asarray(faces, dtype=np.int32)[keep[faces].all(axis=1)]
    # Drop the two eyeball spheres: keep only the largest connected component.
    labels, counts, _ = expand.make_mesh(
        np.zeros((int(kept.max()) + 1, 3)), kept
    ).cluster_connected_triangles()
    return kept[np.asarray(labels) == int(np.argmax(counts))]


def crop_mesh(verts: np.ndarray, crop_faces: np.ndarray) -> o3d.geometry.TriangleMesh:
    """`original`: the FLAME mesh restricted to `crop_faces`, vertices recompacted."""
    used = np.unique(crop_faces)
    remap = np.zeros(int(crop_faces.max()) + 1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return expand.make_mesh(np.asarray(verts, dtype=np.float64)[used], remap[crop_faces])


def make_remesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Light smoothing + 0.7x quadric decimation (BFM `remesh`, git b8adab3)."""
    m = expand.copy_mesh(mesh).filter_smooth_simple(number_of_iterations=REMESH_SMOOTH_ITERS)
    n_tris = len(np.asarray(m.triangles))
    target = max(int(n_tris * REMESH_DECIMATION), REMESH_MIN_TRIANGLES)
    m = m.simplify_quadric_decimation(target_number_of_triangles=target)
    m.compute_vertex_normals()
    return m


def make_noisy(mesh: o3d.geometry.TriangleMesh, seed: int) -> o3d.geometry.TriangleMesh:
    """Gaussian vertex noise at 0.003 x bbox diagonal, same topology (BFM `noisy`).

    The BFM generator used the unseeded global RNG; `seed` makes it reproducible.
    """
    verts = np.asarray(mesh.vertices)
    scale = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    noise = np.random.default_rng(seed).normal(0.0, NOISE_STD_BBOX_FRACTION * scale, verts.shape)
    return expand.make_mesh(verts + noise, np.asarray(mesh.triangles))


def scale_triangle_targets(n_original_triangles: int) -> None:
    """Rescale the BFM down8k/up60k triangle targets to FLAME's patch resolution.

    Set on the imported module (never edited on disk) so the imported
    `make_down8k`/`make_up60k` keep the same *ratio* to `original` they have on
    BFM.  Idempotent: always derived from the snapshotted BFM constants, never
    from the current (possibly already scaled) module values.
    """
    ratio = n_original_triangles / BFM_ORIGINAL_TRIANGLES
    expand.DOWN_TARGET_TRIANGLES = max(4, round(BFM_DOWN_TARGET * ratio))
    expand.UP_TARGET_TRIANGLES = max(4, round(BFM_UP_TARGET * ratio))


def process_subject(task: tuple[str, str, str, str, bool]) -> tuple[str, str]:
    in_path_str, out_dir_str, corr_str, index_str, overwrite = task
    in_path, out_dir = Path(in_path_str), Path(out_dir_str)
    subject = in_path.stem
    out = {v: out_dir / f"{subject}_GTready_{v}.npz" for v in VARIANTS}

    if not overwrite and all(p.exists() for p in out.values()):
        return "[skip]", subject

    try:
        with np.load(in_path) as d:
            verts, faces = np.asarray(d["V"], dtype=np.float64), np.asarray(d["F"], dtype=np.int32)

        crop_faces = face_crop_faces(faces, Path(corr_str), Path(index_str))
        base = crop_mesh(verts, crop_faces)
        scale_triangle_targets(len(np.asarray(base.triangles)))

        builders = {
            "original": lambda: base,
            "remesh": lambda: make_remesh(base),
            "crop": lambda: make_crop(base),
            "noisy": lambda: make_noisy(base, seed=int(subject[-4:])),
            "down8k": lambda: expand.make_down8k(base),
            "up60k": lambda: expand.make_up60k(base),
        }
        counts = []
        for variant, build in builders.items():
            if overwrite or not out[variant].exists():
                expand.save_variant(build(), out[variant])
            with np.load(out[variant]) as d:
                counts.append(f"{variant}={len(d['V'])}/{len(d['F'])}")
        return "[ok]", f"{subject} " + " ".join(counts)
    except Exception as exc:  # one bad identity must not kill the batch
        return "[fail]", f"{subject}: {exc}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--in-dir", type=Path, required=True, help="dir of flameNNNN.npz identities")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--corr", type=Path, default=DEFAULT_CORR)
    p.add_argument("--index-file", type=Path, default=DEFAULT_INDEX_FILE)
    p.add_argument("--n-subjects", type=int, default=0, help="0 = all")
    p.add_argument("--n-cores", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    subjects = sorted(q for q in a.in_dir.glob("flame*.npz"))
    if not subjects:
        raise FileNotFoundError(f"no flame*.npz in {a.in_dir}")
    if a.n_subjects:
        subjects = subjects[: a.n_subjects]
    a.out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(str(q), str(a.out_dir), str(a.corr), str(a.index_file), a.overwrite) for q in subjects]
    print(f"{len(tasks)} identities {a.in_dir} -> {a.out_dir}  workers={max(1, a.n_cores)}", flush=True)

    if a.n_cores > 1:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        pool = mp.Pool(processes=a.n_cores)
        results = pool.imap_unordered(process_subject, tasks)
    else:
        pool, results = None, map(process_subject, tasks)

    tally = {"[ok]": 0, "[skip]": 0, "[fail]": 0}
    failures = []
    try:
        for i, (status, msg) in enumerate(results, start=1):
            tally[status] += 1
            if status == "[fail]":
                failures.append(msg)
            print(f"[{i}/{len(tasks)}] {status} {msg}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    print(f"\nDone. ok={tally['[ok]']} skip={tally['[skip]']} fail={tally['[fail]']}")
    for msg in failures[:20]:
        print(f"  - {msg}")


def demo() -> None:
    """Self-check: the crop is identity-independent and every variant is non-degenerate."""
    ids = sorted((Path(__file__).parent / "flame_identities").glob("flame*.npz"))[:2]
    assert ids, "need v2_work/genflame/flame_identities to self-check"
    sets, down_tris = [], []
    for q in ids:
        with np.load(q) as d:
            verts, faces = np.asarray(d["V"], float), np.asarray(d["F"], np.int32)
        cf = face_crop_faces(faces, DEFAULT_CORR, DEFAULT_INDEX_FILE)
        sets.append(cf)
        base = crop_mesh(verts, cf)
        scale_triangle_targets(len(np.asarray(base.triangles)))
        n0 = len(np.asarray(base.triangles))
        for name, m in [
            ("remesh", make_remesh(base)),
            ("crop", make_crop(base)),
            ("noisy", make_noisy(base, 0)),
            ("down8k", expand.make_down8k(base)),
            ("up60k", expand.make_up60k(base)),
        ]:
            n = len(np.asarray(m.triangles))
            assert n > 0, name
            assert n != n0 or name == "noisy", f"{name} is a no-op ({n} tris)"
            if name == "down8k":
                down_tris.append(n)
        assert len(np.asarray(make_noisy(base, 0).vertices)) == len(np.asarray(base.vertices))
    assert np.array_equal(sets[0], sets[1]), "crop differs between identities"
    # Catches scale_triangle_targets compounding across subjects in one process.
    assert down_tris[0] == down_tris[1], f"down8k size drifts between identities: {down_tris}"
    print("demo OK: crop is a fixed index set, all 5 derived variants change the mesh")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
