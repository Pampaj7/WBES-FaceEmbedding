#!/usr/bin/env python
"""DiffusionNet operators from a Laplacian with an infinite potential well.

## The problem this addresses

Laplace-Beltrami eigenfunctions are *global*: they carry the geometry and topology of the
whole domain, boundary included. Two realisations of the same face that differ in where the
boundary falls are, spectrally, two different domains — heat reflects off a boundary. Our
measurements show exactly this failure profile, and only this one:

    realisation change            spectral shape distortion   metric behaviour
    pure retessellation (8k-60k)  0.5-0.6%                    fine (rho 0.78-0.80)
    crop (boundary moves)         2.1% (up to 20% per mode)   worst topology everywhere
    Poisson mate (boundary CLOSED, +21% area)                 collapse (rho 0.109)

## The fix, and why this one

Liu, Jacobson & Crane (CGF 2017), "A Dirac Operator for Extrinsic Shape Analysis", Sec. 5,
address precisely this. Their words: rather than "cut the meshes so that they all have
identical boundary shape", substitute the standard boundary conditions with an infinite
potential well, which "provides consistent behavior across patches with different boundary
shapes or discretizations".

We already ran the alternative they reject: re-cropping FaceScape to match the training
support *lowered* the correlation (0.404 -> 0.263). So the literature's warning is not
hypothetical for us — it is a result we obtained the hard way.

Construction. Add a diagonal potential to the Laplacian,

    L' = L + U,     U_ii = a_i * c / (1 + exp(-beta * (d(p_i, q) - alpha)))

where d is geodesic distance from the patch centre q (heat method), a_i the lumped vertex
area, and c a large constant. Eigenfunctions of L' are forced to vanish before reaching the
true boundary, so the *effective* domain is canonical whatever the actual boundary does.
alpha is set per mesh as a quantile of the geodesic-distance distribution, so the well sits
at the same relative depth into the patch regardless of the patch's extent.

DiffusionNet consumes (L, mass, evals, evecs, gradX, gradY); only the first four change,
since the potential is a zeroth-order term and does not touch the gradient operators. The
network is untouched: this is a preprocessing change.

## The prediction, stated before running

If the boundary is the mechanism, the potential well must raise the Poisson-mate and
crop-involving numbers and leave pure retessellation (down8k, up60k) unchanged. If instead
everything moves, or nothing does, the mechanism is something else and we will know it from
one measurement.
"""
from __future__ import annotations

import argparse
import sys
import time
import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "diffusion-net/src"))

from diffusion_net.geometry import compute_operators  # noqa: E402

# Well depth is set RELATIVE to the spectral band actually used (lambda_k of the plain
# Laplacian), not in absolute units as in Liu et al. Absolute 1e10 on meshes whose raw
# coordinates are ~1e5 makes the generalised eigenproblem ill-conditioned in a way that
# depends on the mesh, which showed up as the well *hurting* the resolution case.
# Swept on the spectral-agreement diagnostic (one subject, K=64), disagreement vs `original`:
#   c_rel    crop (boundary moves)   down8k (same boundary, 3x coarser)
#   none     0.0214                  0.0032
#   1e2      0.0117                  0.0066     <- best absolute trade
#   1e4      0.0064                  0.0132
#   1e6      0.0051                  0.0155
# The trade-off is real and worth stating: the well buys boundary invariance and pays in
# discretisation sensitivity. 1e2 is the point where the gain on crop (-0.0097) exceeds the
# loss on down8k (+0.0034); the metric-level effect is the experiment, not this table.
DEFAULT_C_REL = 1.0e2
DEFAULT_BETA = 100.0
DEFAULT_QUANTILE = 0.75


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        V = (z["verts"] if "verts" in z else z["V"]).astype(np.float64)
        F = (z["faces"] if "faces" in z else z["F"]).astype(np.int32)
    return V, F


def geodesic_from_centre(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, int]:
    """Geodesic distance from the vertex closest to the area-weighted centroid."""
    import potpourri3d as pp3d

    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(n, axis=1)
    centroid = (tri.mean(axis=1) * areas[:, None]).sum(0) / max(areas.sum(), 1e-12)
    q = int(np.argmin(((V - centroid) ** 2).sum(1)))
    solver = pp3d.MeshHeatMethodDistanceSolver(V, F.astype(np.int32))
    return np.asarray(solver.compute_distance(q), dtype=np.float64), q


def boundary_vertices(F: np.ndarray) -> np.ndarray:
    """Vertices on edges that belong to exactly one face."""
    e = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    return np.unique(uniq[cnt == 1])


def potential_operators(
    V: np.ndarray,
    F: np.ndarray,
    k_eig: int = 128,
    c_rel: float = DEFAULT_C_REL,
    beta: float = DEFAULT_BETA,
    quantile: float = DEFAULT_QUANTILE,
    alpha_mode: str = "per_mesh",
    alpha_global: float | None = None,
    scale_global: float | None = None,
    area_normalize: bool = False,
):
    """Operators for L' = L + U. Returns the same tuple shape as compute_operators.

    `alpha_mode` selects how the well offset is placed, and this is the parameter the method
    actually turns on:

    per_mesh  each mesh normalises d by its own max and takes a quantile of its own
              distribution. This is NOT what the paper prescribes: it lets the well sit at a
              different absolute place on every mesh, so the patches no longer share a domain
              and the very invariance the well exists to provide is lost.
    global    one offset and one scale for the whole collection, chosen (see --calibrate) so
              that no boundary point of any mesh falls inside the region of interest. This is
              Liu, Jacobson & Crane's prescription: "beta is normalized such that across an
              entire collection of patches no boundary points are contained in the region of
              interest" (CGF 2017, Sec. 5.1).
    """
    if area_normalize:
        # Scale to unit total area BEFORE anything else. This is diffusion-net's own
        # normalize_positions(scale_method='area'), a step the v1 pipeline skipped for the
        # operators (it normalised only the vertices, by maxabs, at load time). It matters here
        # for a second reason: the well is placed at a fixed geodesic radius, so when areas
        # differ across topologies the SAME radius covers a different fraction of the surface
        # -- measured, 22% on `noisy` against 46% on `crop`. Without this the well is not a
        # clean test of boundary invariance, because coverage varies with topology too.
        tri = V[F]
        A = float(0.5 * np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())
        if not np.isfinite(A) or A <= 0:
            raise ValueError(f"area non valida: {A}")
        V = (V - V.mean(0)) / np.sqrt(A)

    Vt = torch.tensor(V, dtype=torch.float32)
    Ft = torch.tensor(F, dtype=torch.int32)
    frames, mass, L, base_evals, _evecs, gradX, gradY = compute_operators(Vt, Ft, k_eig=k_eig)

    d, q = geodesic_from_centre(V, F)
    if alpha_mode == "global":
        if alpha_global is None or scale_global is None:
            raise ValueError("alpha_mode='global' needs --alpha-value and --scale-value")
        d = d / max(float(scale_global), 1e-12)   # COMMON scale, identical for every mesh
    else:
        d = d / max(float(d.max()), 1e-12)        # relative depth into the patch
    mass_np = mass.numpy().astype(np.float64)
    # AREA-weighted quantile, not the plain one: a plain quantile weights each *vertex*
    # equally, so where alpha lands depends on sampling density. Measured consequence of
    # getting this wrong: with a plain quantile the well cut boundary sensitivity 2.4x on
    # `crop` but made `down8k` (same boundary, 3x coarser) 4.3x *worse* — the well was
    # sitting at a different depth on a coarser mesh. Weighting by vertex area makes alpha
    # a property of the surface rather than of its sampling.
    if alpha_mode == "global":
        alpha = float(alpha_global)
    else:
        order = np.argsort(d)
        w = mass_np[order]
        cw = np.cumsum(w) / max(w.sum(), 1e-12)
        alpha = float(d[order][int(np.searchsorted(cw, quantile))])
    # normalised sigmoid well: 0 in the interior, c at the rim
    well = 1.0 / (1.0 + np.exp(-beta * (d - alpha)))
    c = c_rel * float(np.clip(base_evals.numpy(), 0, None).max())
    U = mass_np * c * well

    Lc = L.coalesce()
    L_sp = sp.coo_matrix(
        (Lc.values().numpy().astype(np.float64),
         (Lc.indices()[0].numpy(), Lc.indices()[1].numpy())),
        shape=tuple(Lc.shape),
    ).tocsr()
    Lp = L_sp + sp.diags(U)
    M = sp.diags(mass_np)

    # generalised eigenproblem L' phi = lam M phi, smallest eigenvalues
    try:
        evals, evecs = spla.eigsh(Lp, k=k_eig, M=M, sigma=-1e-8, which="LM")
    except Exception:  # shift-invert can fail on near-singular systems
        evals, evecs = spla.eigsh(Lp + 1e-8 * M, k=k_eig, M=M, which="SM")
    order = np.argsort(evals)
    evals, evecs = evals[order], evecs[:, order]

    return (
        frames,
        mass,
        L,  # the plain Laplacian is kept: DiffusionNet's spectral path does not read it
        torch.tensor(np.clip(evals, 0.0, None), dtype=torch.float32),
        torch.tensor(evecs, dtype=torch.float32),
        gradX,
        gradY,
        {"alpha": alpha, "centre_vertex": q, "well_fraction": float((well > 0.5).mean()),
         # roi = 1 inside the region of interest, 0 where the well has suppressed the surface.
         # Saved because the well alone only makes the DIFFUSION domain canonical: with the
         # standard pooling the suppressed vertices still enter the embedding directly through
         # the xyz input and the mean/max over all vertices. A model that also restricts its
         # pooling to this mask is the only version whose embedding is actually supported on
         # the canonical domain.
         "roi": (1.0 - well).astype(np.float32)},
    )


def save_npz(out: Path, V, F, mass, L, evals, evecs, gradX, gradY, roi=None) -> None:
    def coo(t):
        c = t.coalesce()
        return c.indices().numpy(), c.values().numpy(), np.array(c.shape)

    Li, Lv, Ls = coo(L)
    Xi, Xv, Xs = coo(gradX)
    Yi, Yv, Ys = coo(gradY)
    # Write to a private temporary in the same directory, then rename. Two reasons, both of
    # them things that already bit us: workers whose shards overlap would otherwise interleave
    # writes into one file, and an interrupted write leaves a truncated .npz that every later
    # run counts as "already done" and skips forever, so the set silently never completes.
    # os.replace is atomic within a filesystem, so a name either does not exist or is complete.
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp.npz")
    np.savez(
        tmp,
        verts=V.astype(np.float32), faces=F.astype(np.int64),
        mass=mass.numpy(), evals=evals.numpy(), evecs=evecs.numpy(),
        L_indices=Li, L_values=Lv, L_shape=Ls,
        gradX_indices=Xi, gradX_values=Xv, gradX_shape=Xs,
        gradY_indices=Yi, gradY_values=Yv, gradY_shape=Ys,
        # roi_mask travels with the operators so a masked-pooling model can restrict its
        # support to the same canonical domain the well defines. Harmless for models that
        # ignore it: it is just one extra per-vertex array.
        **({} if roi is None else {"roi_mask": np.asarray(roi, dtype=np.float32)}),
    )
    os.replace(tmp, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--k-eig", type=int, default=128)
    ap.add_argument("--c-rel", type=float, default=DEFAULT_C_REL)
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA)
    ap.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    ap.add_argument("--alpha-mode", choices=["per_mesh", "global"], default="per_mesh",
                    help="global = the paper's collection-wide offset (see calibrate_alpha.py)")
    ap.add_argument("--alpha-value", type=float, default=None)
    ap.add_argument("--scale-value", type=float, default=None)
    ap.add_argument("--area-normalize", action="store_true",
                    help="scale each mesh to unit total area before building operators")
    ap.add_argument("--shard", default="0/1", help="i/n: this process takes files i::n")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob("*.npz"))
    # Shard the FULL sorted list, then drop what already exists -- not the other way round.
    # Filtering first makes each worker's stripe depend on how many outputs happened to exist
    # when *that* worker started, so shards launched seconds apart get non-disjoint stripes:
    # they duplicate some files and silently leave others unassigned. That is how a 3000-file
    # run stopped at 2771 and a 229-file top-up covered only 156, every worker exiting fail=0.
    # Sharding the full list first makes the split deterministic regardless of start time.
    si, sn = (int(x) for x in args.shard.split("/"))
    todo = [p for p in files[si::sn]
            if args.overwrite or not (args.output_dir / p.name).exists()]
    print(f"{len(files)} inputs, {len(todo)} to compute", flush=True)

    t0 = time.time()
    ok = fail = 0
    for i, p in enumerate(todo):
        try:
            V, F = load_mesh(p)
            _, mass, L, evals, evecs, gX, gY, meta = potential_operators(
                V, F, k_eig=args.k_eig, c_rel=args.c_rel, beta=args.beta, quantile=args.quantile,
                alpha_mode=args.alpha_mode, alpha_global=args.alpha_value,
                scale_global=args.scale_value, area_normalize=args.area_normalize,
            )
            save_npz(args.output_dir / p.name, V, F, mass, L, evals, evecs, gX, gY,
                     roi=meta.get("roi"))
            ok += 1
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"[fail] {p.name}: {type(exc).__name__}: {exc}", flush=True)
        if (i + 1) % 25 == 0:
            r = (i + 1) / (time.time() - t0)
            print(f"{i+1}/{len(todo)} ok={ok} fail={fail} ({r:.2f}/s, "
                  f"eta {(len(todo)-i-1)/max(r,1e-9)/60:.0f}m)", flush=True)
    print(f"done ok={ok} fail={fail} in {(time.time()-t0)/60:.1f}m")


def demo() -> None:
    """The claim, as an assertion: with the well, the spectrum must stop caring about the boundary.

    Compares the normalised spectrum of `original` against `crop` (boundary moved) and
    `down8k` (boundary intact, resolution changed), with and without the potential.
    """
    d = REPO_ROOT / "datasets/REMESH/npz_data_topo_500"
    ref = "id0000_GTready_original.npz"

    def spec(name: str, with_well: bool) -> np.ndarray:
        V, F = load_mesh(d / name)
        if with_well:
            _, _, _, ev, _, _, _, _ = potential_operators(V, F, k_eig=64)
            ev = ev.numpy().astype(np.float64)
        else:
            _, _, _, ev, _, _, _ = compute_operators(
                torch.tensor(V, dtype=torch.float32), torch.tensor(F, dtype=torch.int32), k_eig=64
            )
            ev = ev.numpy().astype(np.float64)
        ev = np.clip(ev, 0, None)
        return ev / (ev.max() + 1e-30)

    print(f"{'':10s} {'no well':>10s} {'with well':>11s}")
    out = {}
    for well in (False, True):
        r = spec(ref, well)
        for other in ("id0000_GTready_crop.npz", "id0000_GTready_down8k.npz"):
            o = spec(other, well)
            err = float(np.abs(o[1:] - r[1:]).mean() / (np.abs(r[1:]).mean() + 1e-12))
            out[(other, well)] = err
    for other in ("id0000_GTready_crop.npz", "id0000_GTready_down8k.npz"):
        tag = other.split("_")[-1][:-4]
        print(f"{tag:10s} {out[(other, False)]:10.4f} {out[(other, True)]:11.4f}")

    crop_gain = out[("id0000_GTready_crop.npz", False)] / max(out[("id0000_GTready_crop.npz", True)], 1e-12)
    print(f"\ncrop: spectral disagreement shrinks {crop_gain:.2f}x with the well")
    assert np.isfinite(list(out.values())).all(), "non-finite spectra"


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
