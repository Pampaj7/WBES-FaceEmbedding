#!/usr/bin/env python
"""Poisson-reconstruction training variants for BFM and FLAME.

Why this exists
----------------
The measured cross-topology failure of the learned metric is not a change of
3DMM family, it is a change in how the cross-topology *mate* of a mesh is
built: Spearman drops from 0.406/0.404 (remesh, the benchmark rule) to
0.109/0.057 (Poisson reconstruction) in FLAME *and* FaceScape alike, with no
model-family change involved. Nothing in the current training distribution
resembles a Poisson mate (closed boundary loops, +~21% area, normalized
Hausdorff ~0.38 from its own source vs 0.088 for `original`/`remesh`), which
is exactly why the model breaks on it. The fix under test: add Poisson
realizations to training.

The rule applied here is `datasets/FaceVerse/remesh_faceverse_from_npz.py`'s
`_remesh_geometry`, imported and called verbatim (not re-implemented) with
its exact parameters (surface_points=20000, poisson_depth=7, crop_scale=1.05,
target_faces=20000, normal_radius=0.08, normal_max_nn=30, orient_k=20) --
the *kind* of corruption that was measured, applied unchanged to BFM and
FLAME source meshes, not a domain-scaled approximation of it.

Two variants per subject, `pois0`/`pois1`: same recipe, different point-cloud
sampling seed, and `pois1` uses one Poisson depth deeper (8 vs 7) so the model
sees a family of Poisson corruptions rather than one fixed mesh.

Naming follows the existing `<sid>_GTready_<variant>.npz` convention
(`intrinsic_utils.SUBJECT_RE_ANY` + `robustness/data_utils.infer_topology_label_from_name`,
see `v2_work/train_v2/make_support_bank.py`), so the outputs are ordinary extra
meshes of the same subject and the trainer samples them with zero code change.

Commands
--------
    .conda_env/bin/python v2_work/poisson_aug/make_poisson_variants.py --domain bfm --n-subjects 5 --report
    .conda_env/bin/python v2_work/poisson_aug/make_poisson_variants.py --domain flame --shard 0/40
    .conda_env/bin/python v2_work/poisson_aug/make_poisson_variants.py --demo
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "datasets" / "FaceVerse", REPO_ROOT / "faceBench" / "latentVSpipeline"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import remesh_faceverse_from_npz as rule  # noqa: E402  (the measured Poisson recipe, reused verbatim)
from mesh_npz_utils import normalize_vertices  # noqa: E402  (same normalization support_stats.py uses)

# The extracted rule (step 1). Held fixed across both variants and both domains.
RECIPE = dict(surface_points=20000, crop_scale=1.05, target_faces=20000,
              normal_radius=0.08, normal_max_nn=30, orient_k=20)
VARIANT_DEPTH = {0: 7, 1: 8}  # pois0 = the recipe's own depth; pois1 = one depth deeper

DOMAINS = {
    "bfm": dict(
        in_dir=REPO_ROOT / "datasets" / "REMESH" / "npz_data_topo_500",
        out_dir=REPO_ROOT / "v2_work" / "poisson_aug" / "bfm_poisson",
        name_re=re.compile(r"^(id\d+)_GTready_original\.npz$"),
    ),
    "flame": dict(
        in_dir=REPO_ROOT / "v2_work" / "genflame" / "flame_topo_600",
        out_dir=REPO_ROOT / "v2_work" / "poisson_aug" / "flame_poisson",
        name_re=re.compile(r"^(flame\d+)_GTready_original\.npz$"),
    ),
}


def poisson_variant(mesh: o3d.geometry.TriangleMesh, *, variant: int, seed: int):
    o3d.utility.random.seed(int(seed))
    kwargs = dict(RECIPE, poisson_depth=VARIANT_DEPTH[variant])
    return rule._remesh_geometry(mesh, **kwargs)


def hausdorff(V_src: np.ndarray, V_var: np.ndarray) -> float:
    """Symmetric normalized Hausdorff, same recipe as v2_work/transfer/support_stats.py."""
    a, b = normalize_vertices(V_src), normalize_vertices(V_var)
    d_ab = cKDTree(b).query(a)[0]
    d_ba = cKDTree(a).query(b)[0]
    return float(max(d_ab.max(), d_ba.max()))


def build_subject(base_path: Path, sid: str, out_dir: Path, overwrite: bool, report: bool,
                  variants: tuple[int, ...] = (0, 1)):
    mesh = rule._load_mesh(base_path)
    V_src = np.asarray(mesh.vertices)
    made, report_rows = [], []
    for variant in variants:
        out = out_dir / f"{sid}_GTready_pois{variant}.npz"
        if out.exists() and not overwrite:
            made.append(f"pois{variant}=skip")
            continue
        seed = int(re.sub(r"\D", "", sid)) * 1000 + variant
        verts, faces, meta = poisson_variant(mesh, variant=variant, seed=seed)
        payload = {"V": verts, "F": faces}
        for k, v in meta.items():
            payload[f"meta_{k}"] = np.array(v)
        np.savez(out, **payload)
        made.append(f"pois{variant}={len(verts)}v/{len(faces)}f")
        if report:
            report_rows.append((sid, variant, len(verts), len(faces), hausdorff(V_src, verts)))
    return made, report_rows


def iter_bases(in_dir: Path, name_re: re.Pattern, n_subjects: int, shard: str):
    bases = sorted(p for p in in_dir.glob("*_GTready_original.npz") if name_re.match(p.name))
    if not bases:
        raise FileNotFoundError(f"no *_GTready_original.npz matching {name_re.pattern} in {in_dir}")
    if n_subjects:
        bases = bases[:n_subjects]
    i, n = (int(t) for t in shard.split("/"))
    return bases[i::n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", choices=["bfm", "flame"], required=True)
    ap.add_argument("--in-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--n-subjects", type=int, default=0, help="0 = all")
    ap.add_argument("--shard", type=str, default="0/1", help="i/n: process subjects i::n")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--report", action="store_true", help="print vertex/face counts + Hausdorff per variant")
    # Measured on one subject at 2 threads: pois0 (depth 7) takes 69.8 s, pois1 (depth 8) more
    # than 240 s. Since depth 7 is the depth the FaceScape recipe itself uses, pois1 is an extra
    # variation rather than part of the reproduction, and dropping it removes ~80% of the cost.
    ap.add_argument("--variants", type=str, default="0,1",
                    help="comma-separated variant ids to build (e.g. '0' for depth 7 only)")
    args = ap.parse_args()

    spec = DOMAINS[args.domain]
    in_dir = args.in_dir or spec["in_dir"]
    out_dir = args.out_dir or spec["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    bases = iter_bases(in_dir, spec["name_re"], args.n_subjects, args.shard)
    print(f"{len(bases)} {args.domain} subjects  {in_dir} -> {out_dir}", flush=True)

    t0 = time.time()
    all_rows = []
    for i, base in enumerate(bases, start=1):
        sid = spec["name_re"].match(base.name).group(1)
        t1 = time.time()
        made, rows = build_subject(base, sid, out_dir, args.overwrite, args.report,
                                   variants=tuple(int(v) for v in args.variants.split(",")))
        all_rows.extend(rows)
        print(f"[{i}/{len(bases)}] {sid} {' '.join(made)} ({time.time() - t1:.1f}s)", flush=True)

    if args.report and all_rows:
        print(f"\n{'subject':<10} {'variant':<8} {'verts':>7} {'faces':>7} {'hausdorff_norm':>15}")
        for sid, variant, nv, nf, hd in all_rows:
            print(f"{sid:<10} pois{variant:<4} {nv:>7} {nf:>7} {hd:>15.4f}")
        hds = [r[4] for r in all_rows]
        print(f"mean hausdorff_norm = {np.mean(hds):.4f} (target ~0.38, measured on FaceScape)")

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min ({len(bases)} subjects)")


def demo() -> None:
    """Self-check: variants are real Poisson reconstructions, not the source mesh, with
    the expected boundary-closing signature (higher Hausdorff than the recipe's own
    `remesh` variant, since Poisson closes loops the plain decimation does not)."""
    for domain, spec in DOMAINS.items():
        out_dir, in_dir = spec["out_dir"], spec["in_dir"]
        pois_files = sorted(out_dir.glob("*_GTready_pois*.npz"))
        assert pois_files, f"generate variants first: no *_pois*.npz in {out_dir} (domain={domain})"
        p0 = pois_files[0]
        sid = re.match(r"^(.+?)_GTready_pois\d+\.npz$", p0.name).group(1)
        with np.load(in_dir / f"{sid}_GTready_original.npz") as d:
            V_src = d["V"]
        with np.load(p0) as d:
            V_var, F_var = d["V"], d["F"]
        assert V_var.shape[0] != V_src.shape[0] or not np.allclose(V_var, V_src[: len(V_var)]), \
            f"{p0.name}: looks identical to source, not a real reconstruction"
        assert F_var.min() >= 0 and F_var.max() < len(V_var), f"{p0.name}: face indices out of range"
        hd = hausdorff(V_src, V_var)
        assert hd > 0.15, f"{p0.name}: hausdorff {hd:.3f} too small to be a Poisson-style mate"
        print(f"demo OK ({domain}): {p0.name} {len(V_var)}v/{len(F_var)}f, hausdorff_norm={hd:.3f}")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
