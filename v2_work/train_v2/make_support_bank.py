#!/usr/bin/env python
"""Offline bank of *support* variants per subject, with correct operators.

Why a bank, and not a train-time perturbation
---------------------------------------------
The v1 perturbations (`robustness/noise.py`) move vertices; they never change
which part of the face is present. The measured cross-model failure of the
learned metric tracks mesh *support*, not the 3DMM family (Spearman 0.057
FaceScape as-is -> 0.263 re-cropped to BFM-comparable support -> 0.478 FLAME
crop transported through BFM correspondence -> 0.751 BFM in-domain), so the
augmentation we need changes the *domain* of the mesh, not the positions on it.

That collides with the DiffusionNet operators: `mass`, `evals`, `evecs`, `L`,
`gradX`, `gradY` are precomputed per mesh and are a function of the geometry AND
its boundary. Three options, and why this file implements the second:

(a) **Masking** — keep all vertices, zero/attenuate the features of the dropped
    region. Cheap and operator-safe, but it is not a support change: `L`,
    `gradX/Y` and the eigenbasis still couple the "removed" region into every
    diffusion block, so the encoder still sees a full-support mesh with dark
    features. The spectral tower in particular reads `evals`, which would be the
    full mesh's spectrum whatever we do to the features. It trains invariance to
    *feature dropout*, which v1 already has (`--xyz_feature_dropout`), not
    invariance to support.

(b) **Offline bank with correct operators** (this file) — a handful of support
    variants per subject, each a real mesh whose operators are computed by the
    same `precompute_operators_npz.py` that produced the v1 training data. Costs
    one preprocessing pass and disk; correctness is exact by construction, and
    the operator cost stays out of the training loop (recomputing `compute_operators`
    per augmented sample is tens of seconds per mesh — impossible per step).

(c) **Subgraph extraction with operator restriction** — slice `L` and `mass` to
    the kept vertices at train time. Wrong, and quietly so: the cotangent
    Laplacian rows of the new boundary vertices are those of the *old* interior
    (their missing one-ring still contributes cotangent weights), so the
    submatrix is not the Laplacian of the submesh; the removed region keeps
    leaking through `gradX/gradY` the same way; and `evecs/evals` cannot be
    restricted at all — a submatrix of an eigenbasis is not an eigenbasis, so
    the spectral tower would read the full mesh's spectrum under a cropped
    geometry. Recomputing the eigendecomposition per sample is the very cost (b)
    is paid to avoid.

Train-time sampling is free
---------------------------
Variants are written as `idNNNN_GTready_supp<k>.npz`, i.e. extra *meshes of the
same subject* in the dataset naming convention the trainer already parses
(`intrinsic_utils.SUBJECT_RE_ANY` + `robustness/data_utils.infer_topology_label_from_name`).
Symlink them into the training view and v1's own per-subject mesh sampler
(`sample_mesh_indices`, capped by `--max_meshes_per_subject_train`) samples base
and support variants uniformly: the support augmentation *is* the sampler, and no
trainer code changes. Under `--train_level mesh_pair/mixed` they additionally
become new topology labels, so cross-topology pairs gain cross-support pairs.

What varies per variant
-----------------------
One random center vertex + a distance quantile, so both extent and boundary
shape move; 1 in 3 variants removes an interior ball instead (a scanner-dropout
hole, i.e. an interior boundary loop, which is a support change the pure crops
cannot express); then a random quadric decimation, so vertex resolution varies
too. Seeded by (subject, variant index): the bank is reproducible, and supports
differ *across* subjects as well as within one, which is the point — a variant
index is deliberately not a shared topology.

Commands
--------
    .conda_env/bin/python v2_work/train_v2/make_support_bank.py --n-subjects 20
    .conda_env/bin/python v2_work/train_v2/make_support_bank.py --demo
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "face_embedding" / "gt_encdec" / "autoencoder"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import precompute_operators_npz as precomp  # noqa: E402  (reused: operator npz format)
from datasets import expand_remesh_topologies as expand  # noqa: E402
from v2_work.genflame.make_flame_topologies import crop_mesh  # noqa: E402  (recompaction)

DEFAULT_IN_DIR = REPO_ROOT / "datasets" / "REMESH" / "npz_data_topo_500"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "support_bank" / "npz_withops"

K_EIG = 128  # same as datasets/REMESH/npz_data_topo_500_withops
CROP_QUANTILE = (0.50, 0.90)  # keep vertices inside this distance quantile
HOLE_QUANTILE = (0.10, 0.30)  # or drop the nearest ones (interior hole)
HOLE_PROB = 1.0 / 3.0
DECIMATE_RATIO = (0.50, 1.00)  # triangle ratio after cropping (>0.95 = no-op)
MIN_VERTS = 400
SUBJECT_RE = re.compile(r"(id\d+)", re.IGNORECASE)


def support_variant(V: np.ndarray, F: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """One support variant of (V, F): sub-region or holed region, random resolution."""
    rng = np.random.default_rng(seed)
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int32)

    dist = np.linalg.norm(V - V[int(rng.integers(len(V)))], axis=1)
    hole = bool(rng.random() < HOLE_PROB)
    q = rng.uniform(*(HOLE_QUANTILE if hole else CROP_QUANTILE))
    thr = float(np.quantile(dist, q))
    keep = dist > thr if hole else dist <= thr

    faces = F[keep[F].all(axis=1)]
    if len(faces) < 100:
        raise ValueError(f"degenerate support: {len(faces)} triangles (hole={hole}, q={q:.2f})")
    labels, counts, _ = expand.make_mesh(V, faces).cluster_connected_triangles()
    faces = faces[np.asarray(labels) == int(np.argmax(counts))]

    mesh = crop_mesh(V, faces)
    ratio = float(rng.uniform(*DECIMATE_RATIO))
    if ratio < 0.95:
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=max(200, int(len(faces) * ratio))
        )
        mesh.compute_vertex_normals()

    V_out, F_out = expand.mesh_to_arrays(mesh)
    if len(V_out) < MIN_VERTS:
        raise ValueError(f"support variant too small: {len(V_out)} verts")
    return V_out, F_out


def build_subject(base_npz: Path, out_dir: Path, n_variants: int, overwrite: bool) -> list[str]:
    """Write `<sid>_GTready_supp<k>.npz` (verts/faces + operators) for one subject."""
    m = SUBJECT_RE.search(base_npz.name)
    if m is None:
        raise ValueError(f"no subject id in {base_npz.name}")
    sid = m.group(1).lower()
    V, F = precomp.load_geometry_from_npz(base_npz)

    made: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(n_variants):
            out = out_dir / f"{sid}_GTready_supp{k}.npz"
            if out.exists() and not overwrite:
                made.append(f"supp{k}=skip")
                continue
            V_k, F_k = None, None
            for attempt in range(5):  # small bases (FLAME) can draw a degenerate support
                try:
                    V_k, F_k = support_variant(V, F, seed=int(sid[2:]) * 1000 + k + 100_000 * attempt)
                    break
                except ValueError as exc:
                    last_exc = exc
            if V_k is None:
                raise RuntimeError(f"{out.name}: no usable support in 5 draws ({last_exc})")
            raw = Path(tmp) / out.name
            np.savez(raw, V=V_k, F=F_k)
            status, msg = precomp.process_file((str(raw), str(out_dir), "npz", K_EIG, True))
            if status != "[ok]":
                raise RuntimeError(f"operators failed for {out.name}: {msg}")
            made.append(f"supp{k}={len(V_k)}v/{len(F_k)}f")
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR, help="dir of *_GTready_original.npz")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--n-subjects", type=int, default=20, help="0 = all")
    ap.add_argument("--n-variants", type=int, default=5)
    ap.add_argument("--shard", type=str, default="0/1", help="i/n: process subjects i::n (parallel by process)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    bases = sorted(a.in_dir.glob("*_GTready_original.npz"))
    if not bases:
        raise FileNotFoundError(f"no *_GTready_original.npz in {a.in_dir}")
    if a.n_subjects:
        bases = bases[: a.n_subjects]
    shard_i, shard_n = (int(t) for t in a.shard.split("/"))
    bases = bases[shard_i::shard_n]
    a.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(bases)} subjects x {a.n_variants} variants  {a.in_dir} -> {a.out_dir}", flush=True)
    t0 = time.time()
    for i, base in enumerate(bases, start=1):
        t1 = time.time()
        made = build_subject(base, a.out_dir, a.n_variants, a.overwrite)
        print(f"[{i}/{len(bases)}] {base.stem} {' '.join(made)} ({time.time() - t1:.1f}s)", flush=True)

    elapsed = time.time() - t0
    n_files = len(bases) * a.n_variants
    size = sum(p.stat().st_size for p in a.out_dir.glob("*_supp*.npz"))
    print(
        f"\nDone in {elapsed / 60:.1f} min | {elapsed / max(n_files, 1):.1f}s/variant | "
        f"{size / 1e9:.2f} GB on disk ({size / max(n_files, 1) / 1e6:.1f} MB/variant)"
    )


def demo() -> None:
    """Self-check on the generated bank: variants are real, distinct supports with matching ops."""
    out_dir = DEFAULT_OUT_DIR
    files = sorted(out_dir.glob("*_supp*.npz"))
    assert files, f"generate the bank first: no *_supp*.npz in {out_dir}"

    by_subject: dict[str, list[Path]] = {}
    for p in files:
        by_subject.setdefault(SUBJECT_RE.search(p.name).group(1).lower(), []).append(p)

    n_checked = 0
    for sid, paths in sorted(by_subject.items())[:3]:
        base = DEFAULT_IN_DIR / f"{sid}_GTready_original.npz"
        V_base, F_base = precomp.load_geometry_from_npz(base)
        counts, extents = [], []
        for p in sorted(paths):
            with np.load(p) as d:
                V, F, evals = d["verts"], d["faces"], d["evals"]
                L_shape, mass = d["L_shape"], d["mass"]
                gx_shape, gy_shape = d["gradX_shape"], d["gradY_shape"]
            n = len(V)
            # operators describe *this* mesh
            assert tuple(L_shape) == (n, n), f"{p.name}: L {tuple(L_shape)} != ({n},{n})"
            assert tuple(gx_shape) == tuple(gy_shape) == (n, n), f"{p.name}: grad shape"
            assert mass.shape[0] == n, f"{p.name}: mass {mass.shape} vs {n} verts"
            assert len(evals) == min(K_EIG, max(1, n - 2)), f"{p.name}: {len(evals)} evals"
            assert np.isfinite(evals).all(), f"{p.name}: non-finite evals"
            assert (evals >= -1e-8).all(), f"{p.name}: negative evals {evals.min()}"
            assert np.isfinite(V).all() and F.min() >= 0 and F.max() < n, f"{p.name}: bad mesh"
            # it is a real support change, not the base mesh
            assert n < len(V_base), f"{p.name}: {n} verts >= base {len(V_base)}"
            assert len(F) < len(F_base), f"{p.name}: no triangles removed"
            counts.append(n)
            extents.append(tuple(np.round(V.max(axis=0) - V.min(axis=0), 4)))
            n_checked += 1
        assert len(set(counts)) > 1, f"{sid}: all variants share a vertex count {counts}"
        assert len(set(extents)) > 1, f"{sid}: all variants share a bounding box"

    print(f"demo OK: {n_checked} support variants, operators match their own mesh, supports differ")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
