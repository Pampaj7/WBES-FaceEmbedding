"""Correspondence-free shape distances between triangle meshes (currents / varifolds).

Pure torch, no KeOps, CPU-friendly, chunked. Meshes need no vertex correspondence
and may differ in topology and triangle count.

Math
----
A mesh is turned into a discrete geometric measure: for each triangle i we keep the
centroid c_i in R^3, the unit normal n_i, and the area a_i.

Currents (oriented) inner product between meshes X and Y:

    <X, Y>_cur = sum_i sum_j k(c_i^X, c_j^Y) * a_i a_j * (n_i^X . n_j^Y)

Varifolds (orientation-invariant) inner product -- the normal term is squared:

    <X, Y>_var = sum_i sum_j k(c_i^X, c_j^Y) * a_i a_j * (n_i^X . n_j^Y)^2

with the Gaussian position kernel k(x, y) = exp(-||x - y||^2 / sigma^2).

Both are positive semi-definite kernels, so the induced squared distance is

    d_sigma^2 = <X, X> + <Y, Y> - 2 <X, Y>   (>= 0, clamped for float noise)

Multi-scale: squared distances are summed over the sigmas (equivalent to a single
kernel that is the sum of the Gaussians), and the returned value is the square root
of that sum, so it stays a metric on the measure space.

Normalization / scale
---------------------
Two modes, both applied per mesh before the kernel sums:

``normalize="maxabs"`` mirrors the rest of the repo (see
faceBench/latentVSpipeline/mesh_npz_utils.normalize_vertices): center on the vertex
mean, divide by the max absolute coordinate -> unit max-abs box.

``normalize="area"`` (default) centers on the *area-weighted* centroid and divides by
sqrt(total surface area), so total mass is exactly 1.

The default is "area" on purpose. The repo's "maxabs" convention is topology-dependent:
the vertex mean moves when vertex density changes and the max-abs coordinate depends on
one extreme vertex surviving decimation. On REMESH id0000, original vs down8k differ by
4.5% in scale and ~0.04 (normalized units) in centroid *for the same subject*, which is
the same order as the difference between two different faces -- under "maxabs" the
retopology noise swamps the identity signal (measured cross-subject/cross-topology ratio
1.25, with same-subject distances exceeding cross-subject ones for many pairs). Switching
to "area" shrinks same-subject cross-topology distance ~37x and restores the ordering.
Pass normalize="maxabs" when exact parity with the v1 pipeline matters more than
discriminative power.

Sigmas are in units of the resulting normalization (for "area", total area = 1, so the
face spans roughly [-0.5, 0.5]).

Which distance to use
---------------------
Varifold. Measured on 8 REMESH subjects (original vs down8k for the same subject, vs
original for different subjects, max_tris=2000, default sigmas): varifold separates with
a mean ratio of 1.67 and 7% of subject pairs closer than the worst retopology pair, while
currents scores 0.97 -- i.e. currents' oriented normal term is as sensitive to
tessellation as it is to identity, and does not rank subjects at all. Currents is kept
because it is the orientation-sensitive one (a flipped mesh is ~15x further away than a
different subject, where varifold sees exactly zero).

Decimation
----------
Meshes above ``max_tris`` triangles are uniformly subsampled (triangles, not vertices)
to bound the O(N*M) kernel sum. Areas of the surviving triangles are rescaled by
total_area / sampled_area so that the total mass of the measure is preserved and
distances stay comparable across meshes with different triangle counts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

DEFAULT_SIGMAS = (0.5, 0.2, 0.1, 0.05)
BLOCK = 2048


def _load_faces_verts(src):
    """Accept a path to an npz with V/F (or verts/faces) or a raw (V, F) tuple."""
    if isinstance(src, (str, Path)):
        with np.load(str(src), allow_pickle=True) as data:
            keys = set(data.files)
            for vk, fk in (("V", "F"), ("verts", "faces"), ("vertices", "faces")):
                if vk in keys and fk in keys:
                    return np.asarray(data[vk], np.float64), np.asarray(data[fk], np.int64)
            raise KeyError(f"No vertex/face keys in {src}; keys={sorted(keys)}")
    V, F = src
    return np.asarray(V, np.float64), np.asarray(F, np.int64)


def mesh_measure(src, max_tris: int = 4000, seed: int = 0, device="cpu",
                 normalize: str = "area") -> dict:
    """Mesh -> discrete geometric measure {centroids, normals, areas} as torch tensors.

    ``normalize``: "area" (default, topology-robust) or "maxabs" (repo v1 convention).
    See the module docstring for why the default is not "maxabs".
    """
    V, F = _load_faces_verts(src)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"Expected (n, 3) vertices, got {V.shape}")
    tri = V[F]                                             # (N, 3, 3)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    areas = 0.5 * norm
    normals = cross / np.maximum(norm, 1e-12)[:, None]     # degenerate tris -> ~0 normal
    centroids = tri.mean(axis=1)

    # A similarity (shift, scale) on the vertices acts on the measure directly:
    # centroids -> (c - shift) / s, areas -> a / s^2, normals unchanged.
    if normalize == "area":
        shift = (centroids * areas[:, None]).sum(0) / max(areas.sum(), 1e-12)
        scale = float(np.sqrt(areas.sum()))
    elif normalize == "maxabs":
        shift = V.mean(axis=0)
        scale = float(np.max(np.abs(V - shift)))
    else:
        raise ValueError(f"normalize must be 'area' or 'maxabs', got {normalize!r}")
    scale = scale if scale > 1e-6 else 1.0
    centroids = (centroids - shift) / scale
    areas = areas / (scale * scale)

    total_area = float(areas.sum())
    if len(areas) > max_tris:
        idx = np.random.default_rng(seed).choice(len(areas), max_tris, replace=False)
        centroids, normals, areas = centroids[idx], normals[idx], areas[idx]
        areas = areas * (total_area / max(float(areas.sum()), 1e-12))

    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    return {"centroids": t(centroids), "normals": t(normals), "areas": t(areas), "_cache": {}}


def _inner(mA: dict, mB: dict, sigmas: Sequence[float], kind: str, block: int = BLOCK):
    """Per-sigma kernel inner products <A, B>, computed in (block x block) tiles."""
    cA, nA, aA = mA["centroids"], mA["normals"], mA["areas"]
    cB, nB, aB = mB["centroids"], mB["normals"], mB["areas"]
    inv_s2 = torch.tensor([1.0 / (float(s) ** 2) for s in sigmas], dtype=torch.float32)
    out = torch.zeros(len(sigmas), dtype=torch.float64)

    if kind not in ("varifold", "currents"):
        raise ValueError(f"kind must be 'varifold' or 'currents', got {kind!r}")

    for i in range(0, len(aA), block):
        ci, ni, ai = cA[i:i + block], nA[i:i + block], aA[i:i + block]
        for j in range(0, len(aB), block):
            cj, nj, aj = cB[j:j + block], nB[j:j + block], aB[j:j + block]
            nd2 = torch.cdist(ci, cj)
            nd2 = nd2.mul_(nd2).neg_()                     # -||c_i - c_j||^2, (bi, bj)
            w = ni @ nj.T                                  # fresh tensor, safe in-place
            if kind == "varifold":
                w.mul_(w)
            w.mul_(ai[:, None]).mul_(aj[None, :])
            wf = w.reshape(-1)
            for k in range(len(sigmas)):
                # clamp_min_: below -87 exp() underflows out of the float32 normal range,
                # and the underflow path is ~8x slower than the normal one. The floor is
                # 1e-38, i.e. nothing, but it dominates runtime at the smallest sigmas.
                e = torch.exp((nd2 * inv_s2[k]).clamp_min_(-87.0))
                # dot() avoids materializing exp*w and keeps BLAS's blocked summation
                out[k] += torch.dot(e.reshape(-1), wf).double()
    return out


def _self_inner(m: dict, sigmas, kind: str, block: int):
    key = (kind, tuple(float(s) for s in sigmas), block)
    cache = m.setdefault("_cache", {})
    if key not in cache:
        cache[key] = _inner(m, m, sigmas, kind, block)
    return cache[key]


def _distance(mA: dict, mB: dict, sigmas, kind: str, block: int) -> float:
    xx = _self_inner(mA, sigmas, kind, block)
    yy = _self_inner(mB, sigmas, kind, block)
    xy = _inner(mA, mB, sigmas, kind, block)
    d2 = (xx + yy - 2.0 * xy).clamp_min(0.0).sum()
    return float(torch.sqrt(d2))


def varifold_distance(mA: dict, mB: dict, sigmas=DEFAULT_SIGMAS, block: int = BLOCK) -> float:
    """Orientation-invariant multi-scale varifold distance."""
    return _distance(mA, mB, sigmas, "varifold", block)


def currents_distance(mA: dict, mB: dict, sigmas=DEFAULT_SIGMAS, block: int = BLOCK) -> float:
    """Orientation-sensitive multi-scale currents distance."""
    return _distance(mA, mB, sigmas, "currents", block)


def pairwise_distances(measures: Sequence[dict], pairs: Iterable[tuple], kind: str = "varifold",
                       sigmas=DEFAULT_SIGMAS, block: int = BLOCK) -> np.ndarray:
    """Distances for an explicit list of (i, j) index pairs into ``measures``."""
    from tqdm.auto import tqdm

    pairs = list(pairs)
    out = np.empty(len(pairs), dtype=np.float64)
    for p, (i, j) in enumerate(tqdm(pairs, desc=f"{kind} pairs")):
        out[p] = _distance(measures[i], measures[j], sigmas, kind, block)
    return out
