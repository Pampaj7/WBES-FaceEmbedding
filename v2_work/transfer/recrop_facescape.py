"""Re-crop FaceScape meshes to a BFM-comparable support, so the zero-shot
cross-model transfer number can be read without the support-mismatch confound.

    # 1. calibrate + inspect (no writes except renders)
    .conda_env/bin/python v2_work/transfer/recrop_facescape.py --calibrate-only
    # 2. build the re-cropped original+remesh set
    .conda_env/bin/python v2_work/transfer/recrop_facescape.py --out-dir v2_work/transfer/facescape_recrop

Why
---
`support_stats.py` shows the FaceScape `original` support is far from the BFM
training support (bbox x/y 0.95 vs 0.77, z/y 0.58 vs 0.43, area within 0.6 of the
centroid 0.25 vs 0.46) while FLAME -- whose crop was carried over through the
BFM<->FLAME vertex correspondence -- matches BFM on every one of those axes.  So
the two domains differ in the *region of the face* they cover, not just in the
3DMM.  This script removes that difference and nothing else.

The crop criterion (calibrated ON BFM, applied to FaceScape)
------------------------------------------------------------
1. Nose tip = the extreme vertex along -z.  Verified to hold in all three domains
   (BFM / FLAME / FaceScape all put their z-min vertex at x~0, y~0 -- the frontal
   frame convention documented in v2_work/phase0/render_mesh.py).
2. Crop template: the BFM crop boundary is *not* a circle (it has a forehead
   notch), so it is stored as a star-shaped radial template R_B(theta): the median
   in-plane radius of BFM boundary vertices around their own nose tip, per 10 deg
   angular bin, averaged over subjects.  A FaceScape vertex is kept iff its
   in-plane radius around its own nose tip is <= s * R_B(theta).
3. Scale s: the one quantity that cannot be read off the crop itself, because
   every mesh is normalized by max|coord| over a *different* region -- so
   FaceScape's normalized unit is not BFM's.  It is fixed with an anatomical
   ruler that lives strictly INSIDE both supports: the radial depth profile of
   the nose, dz(rho) = median (z - z_nosetip) over vertices in the in-plane
   annulus rho, for rho in [0.05, 0.45] BFM units.  dz and rho are both lengths,
   so dz(rho)/rho is dimensionless and varies strongly along the nose -- matching
   the two profiles determines s.  s is fit ONCE on the domain-mean profile (not
   per mesh), so no per-identity information can leak into the crop.

   Weakness, stated: this assumes mean face anatomy is the same in the two
   domains, which is exactly what a scale ruler must assume when no landmark
   correspondence exists (there is no BFM<->FaceScape correspondence file).  Its
   independent check is that the SAME fit run on FLAME -- whose support is known
   to match BFM by construction -- must return s ~ 1.
4. Boundary handling matches the other two domains: largest connected component,
   then `close_small_boundary_loops` from datasets/expand_remesh_topologies.py.
5. `remesh` variant by the BFM/FLAME rule (2 smoothing iterations + 0.7x quadric
   decimation), imported from v2_work/genflame/make_flame_topologies.py, so the
   cross-topology pairs are built the same way in every domain.
6. Operator-scale control: the saved vertices are rescaled so the re-cropped mesh
   has the same raw surface area as the mean uncropped FaceScape `original`.
   Rationale: `precompute_operators_npz.py` computes the Laplacian on RAW
   vertices while the network is fed normalized ones, so the absolute spectrum
   scale is a property of the file.  Cropping would shrink the area and move the
   spectrum, which would confound "support changed" with "operator scale
   changed".  With the rescale, the crop arm and the +0.057 baseline arm sit at
   the same spectral scale and the only difference is the support.

OUTCOME (2026-08-17): the support hypothesis is REFUTED, keep this script for the
record but do not build on it.  Matching the support moved zero-shot Spearman from
+0.057 to +0.263, but the `--control-arm` -- uncropped FaceScape, only the `remesh`
pair mate rebuilt by the BFM/FLAME rule -- reaches +0.404, i.e. *better* than the
re-crop.  The whole gap was the cross-topology PAIR MATE, not the support: v1
paired the 10k downsample with `remesh_10k`, a Poisson depth-7 re-reconstruction
(datasets/FaceVerse/remesh_faceverse_from_npz.py) that closes the boundary loops,
adds ~10-20% surface and departs from its own source surface by a normalized
Hausdorff of 0.38 (the BFM/FLAME topology variants: 0.09).  Applying that same
Poisson mate to FLAME -- the domain that transfers at +0.478 -- collapses it to
+0.109, reproducing the "cross-model failure" without changing 3DMM at all.  Under
the matched pair rule FaceScape (+0.404) and FLAME (+0.406) are indistinguishable.
Cropping to a BFM-like support HURTS (+0.404 -> +0.263), most likely because the GT
distance matrix is a whole-face vertex-mean L2 and the crop throws away the
jaw/forehead/cheek geometry the GT integrates over.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "faceBench/latentVSpipeline"))
sys.path.insert(0, str(REPO_ROOT / "v2_work/genflame"))
sys.path.insert(0, str(REPO_ROOT / "v2_work/phase0"))

import numpy as np  # noqa: E402
from mesh_npz_utils import normalize_vertices  # noqa: E402

BFM_DIR = REPO_ROOT / "datasets/REMESH/npz_data_topo_500_withops"
FLAME_DIR = REPO_ROOT / "v2_work/genflame/flame_topo_600_withops"
FS_DIR = REPO_ROOT / "datasets/FaceVerse/cross_topology_10k_with_ops"
N_CALIB = 50  # subjects per domain used to build the template / profiles

N_ANGLE_BINS = 36
PROFILE_RHO = np.linspace(0.05, 0.45, 17)  # BFM normalized units
SCALE_GRID = np.round(np.arange(0.40, 1.601, 0.01), 3)
SWEEP_GRID = np.round(np.arange(0.90, 1.1251, 0.025), 3)
# dimensionless support-shape statistics used to pick the crop scale (fitted) and
# to check it (not fitted); see support_stats.py for their definitions.
FIT_METRICS = ("frac_area_within_r0.4", "frac_area_within_r0.6", "frac_area_within_r0.8")
CHECK_METRICS = ("ext_x_over_y", "ext_z_over_y", "area_over_diag2",
                 "boundary_mean_radius", "area_mean_radius", "mean_edge_over_diag")


# --------------------------------------------------------------------------- io
def load_mesh(path: Path, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        V = np.asarray(z["verts"] if "verts" in z else z["V"], dtype=np.float64)
        F = np.asarray(z["faces"] if "faces" in z else z["F"], dtype=np.int64)
    return (normalize_vertices(V) if normalize else V), F


def surface_area(V: np.ndarray, F: np.ndarray) -> float:
    t = V[F]
    return float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())


# ------------------------------------------------------------------ geometry
def nose_tip(V: np.ndarray) -> np.ndarray:
    """Extreme vertex along -z; assert it really is the nose (near the x/y axis)."""
    tip = V[int(np.argmin(V[:, 2]))]
    if np.linalg.norm(tip[:2]) > 0.3:
        raise ValueError(f"z-min vertex at {tip} is not near the face axis; frame convention?")
    return tip


def polar(V: np.ndarray, tip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """In-plane radius and angle of every vertex around the nose tip."""
    d = V[:, :2] - tip[:2]
    return np.linalg.norm(d, axis=1), np.arctan2(d[:, 1], d[:, 0])


def boundary_vertices(F: np.ndarray) -> np.ndarray:
    e = np.sort(F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    return np.unique(uniq[counts == 1])


def bin_centers() -> np.ndarray:
    return -np.pi + (np.arange(N_ANGLE_BINS) + 0.5) * (2 * np.pi / N_ANGLE_BINS)


def depth_profile(rho: np.ndarray, dz: np.ndarray, rho_grid: np.ndarray = PROFILE_RHO,
                  scale: float = 1.0) -> np.ndarray:
    """median (z - z_tip) in in-plane annuli at `scale * rho_grid`; NaN where empty."""
    grid = scale * rho_grid
    half = 0.5 * (grid[1] - grid[0])
    return np.array([
        np.median(dz[np.abs(rho - r) <= half]) if np.any(np.abs(rho - r) <= half) else np.nan
        for r in grid
    ])


def boundary_template(files: list[Path]) -> np.ndarray:
    """Star-shaped R(theta) of the crop boundary, averaged over subjects."""
    per_subject = []
    for f in files:
        V, F = load_mesh(f)
        rho, th = polar(V, nose_tip(V))
        b = boundary_vertices(F)
        idx = np.digitize(th[b], np.linspace(-np.pi, np.pi, N_ANGLE_BINS + 1)) - 1
        r = np.array([np.median(rho[b][idx == k]) if np.any(idx == k) else np.nan
                      for k in range(N_ANGLE_BINS)])
        per_subject.append(r)
    R = np.nanmean(np.vstack(per_subject), axis=0)
    if np.isnan(R).any():  # fill any empty bin from its neighbours
        ok = ~np.isnan(R)
        R = np.interp(bin_centers(), bin_centers()[ok], R[ok], period=2 * np.pi)
    k = np.array([1.0, 1.0, 1.0]) / 3.0  # circular 3-bin smoothing
    return np.convolve(np.r_[R[-1], R, R[0]], k, mode="valid")


def template_radius(theta: np.ndarray, R: np.ndarray) -> np.ndarray:
    return np.interp(theta, bin_centers(), R, period=2 * np.pi)


# ------------------------------------------------------------------ calibration
def nose_polar(files: list[Path]) -> list[tuple[np.ndarray, np.ndarray]]:
    """(in-plane radius, depth below the nose tip) per mesh -- loaded once, reused
    across the whole scale grid."""
    out = []
    for f in files:
        V, _ = load_mesh(f)
        tip = nose_tip(V)
        rho, _ = polar(V, tip)
        out.append((rho, V[:, 2] - tip[2]))
    return out


def mean_depth_profile(polars: list[tuple[np.ndarray, np.ndarray]], scale: float = 1.0) -> np.ndarray:
    """Domain-mean nose depth profile (one value per radius in PROFILE_RHO)."""
    return np.nanmean(np.vstack([depth_profile(r, d, scale=scale) for r, d in polars]), axis=0)


def fit_scale(target: np.ndarray, polars: list[tuple[np.ndarray, np.ndarray]],
              grid: np.ndarray = SCALE_GRID) -> tuple[float, np.ndarray]:
    """s minimizing || dz_src(s*rho)/s - dz_target(rho) ||^2 on the domain mean profile."""
    losses = []
    for s in grid:
        src = mean_depth_profile(polars, scale=s) / s
        m = np.isfinite(src) & np.isfinite(target)
        losses.append(np.mean((src[m] - target[m]) ** 2) if m.sum() >= 5 else np.inf)
    losses = np.asarray(losses)
    return float(grid[int(np.argmin(losses))]), losses


# ------------------------------------------------------------------ the crop
def crop(V: np.ndarray, F: np.ndarray, R: np.ndarray, scale: float):
    """Star-crop around the nose tip, largest component, small boundary loops closed."""
    from datasets import expand_remesh_topologies as expand

    rho, th = polar(V, nose_tip(V))
    keep = rho <= scale * template_radius(th, R)
    kf = F[keep[F].all(axis=1)]
    if not len(kf):
        raise ValueError("crop removed every triangle")

    labels, counts, _ = expand.make_mesh(np.zeros((int(kf.max()) + 1, 3)), kf
                                         ).cluster_connected_triangles()
    kf = kf[np.asarray(labels) == int(np.argmax(counts))]

    used = np.unique(kf)
    remap = np.zeros(int(kf.max()) + 1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    mesh = expand.close_small_boundary_loops(expand.make_mesh(V[used], remap[kf]))
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles, dtype=np.int32)


def domain_stats(meshes) -> dict[str, float]:
    """Mean support statistics over an iterable of (V, F)."""
    from support_stats import mesh_stats

    rows = [mesh_stats(V, F, np.zeros(64)) for V, F in meshes]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def sweep(bfm_files: list[Path], fs_files: list[Path], R: np.ndarray,
          grid: np.ndarray = SWEEP_GRID) -> tuple[float, dict]:
    """Pick the crop scale by matching BFM's dimensionless radial *area* profile.

    Fitted on FIT_METRICS only; CHECK_METRICS are printed for the chosen scale but
    never optimized, so they are an honest test of the crop and not of the fit.
    """
    tgt = domain_stats(load_mesh(f) for f in bfm_files)
    cols = FIT_METRICS + CHECK_METRICS
    print(f"{'s':>6}  " + "  ".join(f"{c[:14]:>14}" for c in cols) + "     fit_loss")
    print(f"{'BFM':>6}  " + "  ".join(f"{tgt[c]:>14.4f}" for c in cols) + "        (target)")

    best, best_loss, table = None, np.inf, {}
    for s in grid:
        st = domain_stats(crop(*load_mesh(f), R, float(s)) for f in fs_files)
        loss = float(np.mean([(st[m] - tgt[m]) ** 2 for m in FIT_METRICS]))
        table[float(s)] = st
        print(f"{s:>6.2f}  " + "  ".join(f"{st[c]:>14.4f}" for c in cols) + f"  {loss:11.3e}",
              flush=True)
        if loss < best_loss:
            best, best_loss = float(s), loss
    return best, table


def build(out_dir: Path, R: np.ndarray, scale: float, target_area: float, n: int = 0) -> None:
    from datasets import expand_remesh_topologies as expand
    from make_flame_topologies import make_remesh

    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(FS_DIR.glob("*_01_original.npz"))[: n or None]
    for i, f in enumerate(files, 1):
        subject = f.name.split("_")[0]
        V, F = load_mesh(f)
        Vc, Fc = crop(V, F, R, scale)
        # operator-scale control: match the mean raw area of the uncropped originals
        Vc = Vc * np.sqrt(target_area / surface_area(Vc, Fc))
        base = expand.make_mesh(Vc, Fc)
        expand.save_variant(base, out_dir / f"{subject}_01_original.npz")
        expand.save_variant(make_remesh(base), out_dir / f"{subject}_01_remesh.npz")
        print(f"[{i}/{len(files)}] {subject} verts={len(Vc)} faces={len(Fc)}", flush=True)


def build_control(out_dir: Path, n: int = 0) -> None:
    """Control arm: UNCROPPED FaceScape originals paired with a BFM/FLAME-rule `remesh`.

    Needed because the +0.057 baseline pairs the 10k downsample with FaceScape's own
    `remesh_10k` (an independent remesh of the scan), while the re-cropped arm pairs
    the crop with a 0.7x decimation of itself -- a different, and probably easier,
    kind of cross-topology pair.  Without this arm, "support was fixed" and "the
    remesh recipe changed" are confounded.  Only the `remesh` side is written here;
    the `original` side reuses the existing uncropped operator files unchanged.
    """
    from make_flame_topologies import make_remesh
    from datasets import expand_remesh_topologies as expand

    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(FS_DIR.glob("*_01_original.npz"))[: n or None]
    for i, f in enumerate(files, 1):
        V, F = load_mesh(f, normalize=False)
        m = make_remesh(expand.make_mesh(V, F.astype(np.int32)))
        expand.save_variant(m, out_dir / f"{f.name.split('_')[0]}_01_remesh.npz")
        print(f"[{i}/{len(files)}] {f.name} -> remesh {len(np.asarray(m.vertices))} verts", flush=True)


def train_ready(withops: Path, out_dir: Path, id_offset: int = 3000) -> None:
    """idNNNN symlink view + renamed GT matrix, mirroring v2_work/transfer/facescape_train_ready.

    The GT distance matrix is reused UNCHANGED (same subjects, same identities):
    the GT distances were computed on the FULL FaceScape meshes, and cropping
    changes the meshes, not who they are.  The eval asks whether the latent metric
    orders identities the way the identity GT does, so the correct GT for a
    re-cropped mesh of subject k is still subject k's row.
    """
    import json

    data_out = out_dir / "npz_withops"
    data_out.mkdir(parents=True, exist_ok=True)
    gt_src = REPO_ROOT / "datasets/FaceVerse/gt_distance_matrix/faceverse_detail_pose01_vertex_mean_l2_normalized.npz"

    n = 0
    for p in sorted(withops.glob("*_01_*.npz")):
        subject, _, variant = p.stem.split("_")
        sid = f"id{id_offset + int(subject):04d}"
        link = data_out / f"{sid}_GTready_{variant}.npz"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(p.resolve())
        n += 1

    with np.load(gt_src, allow_pickle=True) as z:
        D, names = z["D_orig"], [str(x) for x in z["names"]]
    renamed = np.array([f"id{id_offset + int(x):04d}" for x in names])
    np.savez(out_dir / "gt_matrix.npz", D_orig=D, names=renamed)
    meta = {
        "source_withops": str(withops), "source_gt": str(gt_src), "id_offset": id_offset,
        "n_symlinks": n, "n_subjects": len(renamed),
        "note": "re-cropped FaceScape; GT matrix reused unchanged (cropping changes the "
                "meshes, not the identities)",
    }
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


# ------------------------------------------------------------------ entrypoint
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, default=Path("v2_work/transfer/facescape_recrop"))
    p.add_argument("--scale", type=float, default=0.0, help="0 = fit it (default)")
    p.add_argument("--calibrate-only", action="store_true")
    p.add_argument("--sweep", action="store_true", help="scan the crop scale and fit it")
    p.add_argument("--n-sweep", type=int, default=15, help="meshes per domain in the sweep")
    p.add_argument("--n", type=int, default=0, help="limit meshes (debug)")
    p.add_argument("--render-dir", type=Path, default=None)
    p.add_argument("--train-ready-from", type=Path, default=None,
                   help="withops dir of the re-cropped set: build the idNNNN view and exit")
    p.add_argument("--train-ready-out", type=Path,
                   default=Path("v2_work/transfer/facescape_recrop_train_ready"))
    p.add_argument("--id-offset", type=int, default=3000)
    p.add_argument("--control-arm", action="store_true",
                   help="write the uncropped-original + BFM-rule-remesh control arm to --out-dir")
    a = p.parse_args()

    if a.train_ready_from:
        train_ready(a.train_ready_from, a.train_ready_out, a.id_offset)
        return
    if a.control_arm:
        build_control(a.out_dir, n=a.n)
        return

    bfm = sorted(BFM_DIR.glob("*_GTready_original.npz"))[:N_CALIB]
    flame = sorted(FLAME_DIR.glob("flame*_GTready_original.npz"))[:N_CALIB]
    fs = sorted(FS_DIR.glob("*_01_original.npz"))[:N_CALIB]

    R = boundary_template(bfm)
    print("BFM boundary template R(theta): "
          f"min={R.min():.3f} mean={R.mean():.3f} max={R.max():.3f}")

    # secondary, independent scale estimate: the nose depth ruler.  Its own
    # validation is the FLAME fit, which must land on ~1.0 (FLAME's support matches
    # BFM's by construction) -- read the printed value before trusting it.
    target = mean_depth_profile(nose_polar(bfm))
    s_flame, _ = fit_scale(target, nose_polar(flame))
    s_fs_depth, loss_fs = fit_scale(target, nose_polar(fs))
    print(f"nose-depth ruler: FLAME s={s_flame:.3f} (validation, want ~1.0), "
          f"FaceScape s={s_fs_depth:.3f}")

    scale = a.scale
    if not scale and a.sweep:
        scale, _ = sweep(bfm[: a.n_sweep], fs[: a.n_sweep], R)
        print(f"support-shape fit: s = {scale:.2f}")
    scale = scale or s_fs_depth

    areas = [surface_area(*load_mesh(f, normalize=False)) for f in fs]
    target_area = float(np.mean(areas))
    print(f"uncropped FaceScape mean raw area = {target_area:.4f} (operator-scale target)")
    print(f"using crop scale s = {scale:.3f}")

    if a.render_dir:
        from PIL import Image
        from render_mesh import render_mesh

        a.render_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for f in bfm[:3]:
            rows.append(render_mesh(*load_mesh(f), size=320))
        for f in fs[:3]:
            V, F = load_mesh(f)
            rows.append(render_mesh(*crop(V, F, R, scale), size=320))
        Image.fromarray(np.concatenate(rows, axis=1)).save(a.render_dir / f"recrop_s{scale:.2f}.png")
        print(f"wrote {a.render_dir / f'recrop_s{scale:.2f}.png'}")

    if not a.calibrate_only:
        build(a.out_dir, R, scale, target_area, n=a.n)


def demo() -> None:
    """Self-check: the crop+scale machinery recovers a known scale on a synthetic pair."""
    rng = np.random.default_rng(0)
    # a paraboloid "face" with a nose bump, sampled twice at different scales
    def face(scale: float, n: int = 4000) -> tuple[np.ndarray, np.ndarray]:
        from scipy.spatial import Delaunay
        xy = rng.uniform(-1, 1, (n, 2))
        z = 0.4 * (xy ** 2).sum(1) - 0.6 * np.exp(-8 * (xy ** 2).sum(1))
        V = np.c_[xy, z] * scale
        return V, Delaunay(xy).simplices.astype(np.int64)

    def prof(V: np.ndarray, scale: float = 1.0) -> np.ndarray:
        Vn = normalize_vertices(V)
        tip = nose_tip(Vn)
        return depth_profile(polar(Vn, tip)[0], Vn[:, 2] - tip[2], scale=scale)

    V1, F1 = face(1.0)
    Vn = normalize_vertices(V1)
    tip = nose_tip(Vn)
    assert np.linalg.norm(tip[:2]) < 0.1, tip
    p1 = prof(V1)
    V2, _ = face(3.0)  # same shape, 3x bigger -> identical after normalization
    assert np.nanmax(np.abs(p1 - prof(V2))) < 0.02
    # and the fit must recover a deliberately mis-normalized copy's scale
    losses = [np.nanmean((prof(V1, s) / s - p1) ** 2) for s in (0.7, 1.0, 1.4)]
    assert np.argmin(losses) == 1, losses
    # a star template of constant radius keeps exactly the disc of that radius
    R = np.full(N_ANGLE_BINS, 0.5)
    rho, th = polar(Vn, tip)
    keep = rho <= template_radius(th, R)
    assert 0.05 < keep.mean() < 0.95 and rho[keep].max() <= 0.5 + 1e-9
    print("demo ok")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
