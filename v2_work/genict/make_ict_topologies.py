"""Build the 6 REMESH-2 topology variants for a directory of ICT identities.

    aau/run.sh v2_work/genict/make_ict_topologies.py \
        --in-dir  datasets/ICT/identities \
        --out-dir datasets/ICT/topo --n-cores 8

Input:  `ictNNNN.npz` with keys `V` (9409,3) / `F` (18460,3) from
        `generate_identities.py`.
Output: `ictNNNN_GTready_<variant>.npz` with keys `V`/`F`, variant in
        {original, remesh, crop, noisy, down8k, up60k} -- the same six names and
        the same `<subject>_GTready_<topology>` convention as the BFM half in
        `datasets/REMESH/npz_data_topo_500`, because downstream code parses it.

`original`
----------
Unlike the FLAME twin, no crop happens here: the face region is already the
whole input.  FLAME needed `make_flame_topologies.face_crop_faces` because a
full FLAME *head* had to be reduced to something comparable to a BFM face
patch, through the published BFM->FLAME correspondence.  ICT ships the face
region as a documented index range of its own topology (README geometry #0,
vertices [0:9408] / polygons [0:9229]), so `generate_identities.py` applies it
at sampling time and `original` is the identity mesh verbatim -- still a fixed
index set shared by every identity, which is what makes it a topology.

Variant semantics
-----------------
Same five rules as the FLAME twin, all scale-relative and therefore transferable
to ICT's centimetre-ish units untouched:
  `remesh` 2 umbrella-smoothing iterations + 0.7x quadric decimation,
  `noisy`  Gaussian vertex noise at 0.003 x bbox diagonal, topology unchanged,
  `crop`   the boundary-band trim of `datasets/remesh.py`,
  `down8k`/`up60k` quadric decimation to an absolute triangle target.

The absolute targets are the one thing that cannot transfer.  BFM's `original`
patch has 46,440 triangles and its targets are 16k / 120k (measured on the
shipped dataset: down8k = 15,999 tris / 8,129 verts, up60k = 120,139 tris /
60,432 verts).  Applied to ICT's 18,460-triangle patch, 16k would be a near
no-op and 120k an 6.5x blow-up, so -- exactly as `make_flame_topologies.py`
does for FLAME -- the invariant kept is the resolution *ratio* to each model's
own `original`: 0.345x and 2.584x, i.e. 6,360 and 47,700 triangles.  The
variant names stay, misnomers and all, because downstream code parses them.

`up60k` is a 1-to-4 midpoint subdivision (4x triangles, geometry unchanged)
followed by decimation to the target, which is how a 2.58x resampling is
reached from a 1x mesh: a pure subdivision can only land on 4x.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mesh_ops as mo  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

ICT_ORIGINAL_TRIANGLES = 18_460  # face region, all quads split in two
BFM_ORIGINAL_TRIANGLES = 46_440  # triangles of a BFM *_GTready_original patch
BFM_DOWN_TARGET_TRIANGLES = 16_000   # snapshot of expand.DOWN_TARGET_TRIANGLES
BFM_UP_TARGET_TRIANGLES = 120_000    # snapshot of expand.UP_TARGET_TRIANGLES
REMESH_SMOOTH_ITERS = 2          # b8adab3 datasets/remesh.py::make_remesh
REMESH_DECIMATION = 0.7
REMESH_MIN_TRIANGLES = 2_000
NOISE_STD_BBOX_FRACTION = 0.003  # b8adab3 datasets/remesh.py::make_noisy

VARIANTS = ("original", "remesh", "crop", "noisy", "down8k", "up60k")


def triangle_targets(n_original_triangles: int) -> tuple[int, int]:
    """down8k/up60k triangle targets at ICT's resolution, same ratio as on BFM."""
    ratio = n_original_triangles / BFM_ORIGINAL_TRIANGLES
    return (max(4, round(BFM_DOWN_TARGET_TRIANGLES * ratio)),
            max(4, round(BFM_UP_TARGET_TRIANGLES * ratio)))


def make_remesh(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Light smoothing + 0.7x quadric decimation (BFM `remesh`, git b8adab3)."""
    Vs = mo.smooth_simple(V, F, REMESH_SMOOTH_ITERS)
    target = max(int(len(F) * REMESH_DECIMATION), REMESH_MIN_TRIANGLES)
    return mo.decimate_to(Vs, F, target)


def make_noisy(V: np.ndarray, F: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian vertex noise at 0.003 x bbox diagonal, same topology (BFM `noisy`).

    The BFM generator used the unseeded global RNG; `seed` makes it reproducible.
    """
    scale = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    noise = np.random.default_rng(seed).normal(0.0, NOISE_STD_BBOX_FRACTION * scale, V.shape)
    return V + noise, F


def make_down8k(V: np.ndarray, F: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
    return mo.decimate_to(V, F, target)


def make_up60k(V: np.ndarray, F: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
    Vu, Fu = mo.subdivide_midpoint(V, F, 1)
    return mo.decimate_to(Vu, Fu, target)


def process_subject(task: tuple[str, str, bool]) -> tuple[str, str]:
    in_path_str, out_dir_str, overwrite = task
    in_path, out_dir = Path(in_path_str), Path(out_dir_str)
    subject = in_path.stem
    out = {v: out_dir / f"{subject}_GTready_{v}.npz" for v in VARIANTS}

    if not overwrite and all(p.exists() for p in out.values()):
        return "[skip]", subject

    try:
        with np.load(in_path) as d:
            V, F = mo.as_arrays(d["V"], d["F"])

        base = mo.prepare_open_surface(V, F)
        if len(base[1]) != len(F):
            return "[fail]", f"{subject}: input is not a clean open surface ({len(F)} -> {len(base[1])} tris)"
        down_target, up_target = triangle_targets(len(base[1]))

        builders = {
            "original": lambda: base,
            "remesh": lambda: make_remesh(*base),
            "crop": lambda: mo.make_crop(*base),
            "noisy": lambda: make_noisy(*base, seed=int(subject[-4:])),
            "down8k": lambda: make_down8k(*base, target=down_target),
            "up60k": lambda: make_up60k(*base, target=up_target),
        }
        counts = []
        for variant, build in builders.items():
            if overwrite or not out[variant].exists():
                mo.save_variant(*build(), path=out[variant])
            with np.load(out[variant]) as d:
                counts.append(f"{variant}={len(d['V'])}/{len(d['F'])}")
        return "[ok]", f"{subject} " + " ".join(counts)
    except Exception as exc:  # one bad identity must not kill the batch
        return "[fail]", f"{subject}: {exc}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--in-dir", type=Path, default=REPO_ROOT / "datasets/ICT/identities")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "datasets/ICT/topo")
    p.add_argument("--n-subjects", type=int, default=0, help="0 = all")
    p.add_argument("--n-cores", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    subjects = sorted(q for q in a.in_dir.glob("ict*.npz"))
    if not subjects:
        raise FileNotFoundError(f"no ict*.npz in {a.in_dir}")
    if a.n_subjects:
        subjects = subjects[: a.n_subjects]
    a.out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(str(q), str(a.out_dir), a.overwrite) for q in subjects]
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
            if status != "[ok]" or i % 100 == 0 or i <= 5:
                print(f"[{i}/{len(tasks)}] {status} {msg}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    print(f"\nDone. ok={tally['[ok]']} skip={tally['[skip]']} fail={tally['[fail]']}")
    for msg in failures[:20]:
        print(f"  - {msg}")
    if tally["[fail]"]:
        raise SystemExit(1)


def demo(argv: list[str]) -> None:
    """Self-check on 3 identities: every variant loads and has the expected size."""
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path, default=REPO_ROOT / "datasets/ICT/identities")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "datasets/ICT/topo")
    p.add_argument("--n-subjects", type=int, default=3)
    a, _ = p.parse_known_args(argv)
    in_dir, out_dir = a.in_dir, a.out_dir
    ids = sorted(in_dir.glob("ict*.npz"))[:a.n_subjects]
    assert ids, f"need identities in {in_dir}"

    down_target, up_target = triangle_targets(ICT_ORIGINAL_TRIANGLES)
    expected = {
        "original": ICT_ORIGINAL_TRIANGLES,
        "remesh": int(ICT_ORIGINAL_TRIANGLES * REMESH_DECIMATION),
        "noisy": ICT_ORIGINAL_TRIANGLES,
        "down8k": down_target,
        "up60k": up_target,
    }
    print(f"targets: down8k={down_target} up60k={up_target}")

    faces_ref, sizes = {}, {}
    for q in ids:
        subject = q.stem
        for v in VARIANTS:
            path = out_dir / f"{subject}_GTready_{v}.npz"
            with np.load(path) as d:
                V, F = d["V"], d["F"]
            assert V.dtype == np.float32 and F.dtype == np.int32, (V.dtype, F.dtype)
            assert F.max() == len(V) - 1, f"{path.name}: unreferenced vertices"
            assert np.isfinite(V).all(), f"{path.name}: non-finite vertices"
            print(f"  {path.name}: V={len(V)} F={len(F)}")
            if v in expected:
                assert abs(len(F) - expected[v]) <= 0.02 * expected[v], \
                    f"{path.name}: {len(F)} tris, expected ~{expected[v]}"
            sizes.setdefault(v, []).append((len(V), len(F)))
            if v in ("original", "noisy"):
                faces_ref.setdefault(v, F)
                assert np.array_equal(faces_ref[v], F), f"{v} is not a fixed index set"
    for v, s in sizes.items():
        if v in ("original", "noisy"):
            assert len(set(s)) == 1, f"{v} size drifts between identities: {s}"
    print("demo OK: 6 variants per identity, sizes on target, original/noisy share one index set")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo(sys.argv[1:])
    else:
        main()
