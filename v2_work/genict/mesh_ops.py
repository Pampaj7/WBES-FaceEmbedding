"""Triangle-mesh helpers for the ICT half of REMESH-2: numpy + libigl, no open3d.

Why this exists: `make_flame_topologies.py` builds the six variants through
`datasets.expand_remesh_topologies` (open3d), and that module **is not in this
checkout** -- it is not in the working tree and not in any commit of the git
history, so the FLAME pipeline cannot be run here as-is either.  open3d is also
absent from `.venv_aau`.  The five primitives the topology builder actually
needs (smooth, quadric decimation, midpoint subdivision, connected components,
clean-up) all exist in `igl`, which the venv already carries, so the ICT half is
built on igl instead of resurrecting the missing module.

`make_crop` / `trim_far_boundary_band` below are a *verbatim* port of
`datasets/remesh.py` (the same constants, the same Dijkstra band, the same
`MIN_KEEP_RATIO` bail-out) with the open3d objects replaced by (V, F) arrays.
That file cannot simply be imported: it does `import open3d` and
`from datasets.expand_remesh_topologies import ...` at module scope.

Every function takes and returns `(V float64 (n,3), F int32 (m,3))`.
"""

from __future__ import annotations

import heapq
from pathlib import Path

import numpy as np
import igl

# datasets/remesh.py::make_crop constants, unchanged (all scale-relative)
BOUNDARY_RADIUS_PERCENTILE = 70.0
TRIM_DISTANCE_RATIO = 0.06
MIN_KEEP_RATIO = 0.85


def as_arrays(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.ascontiguousarray(V, dtype=np.float64), np.ascontiguousarray(F, dtype=np.int32)


def save_variant(V: np.ndarray, F: np.ndarray, path: Path) -> None:
    """Write one topology variant with the keys the rest of the repo reads.

    `V` is float32 (not the float64 of the BFM/FLAME npz): every consumer
    -- `precompute_operators_npz.load_geometry_from_npz`, `areanorm_operators`,
    `GTReadyDatasetNPZ` -- casts to float32 immediately, and at ICT's scale
    (~1e1) float32 still resolves 1e-6, four orders below the closest pair of
    identities.  Halves 37,500 files on CephFS.
    """
    np.savez_compressed(path, V=np.asarray(V, dtype=np.float32), F=np.asarray(F, dtype=np.int32))


def remove_unreferenced(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    used = np.unique(F)
    remap = np.zeros(len(V), dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return np.ascontiguousarray(V[used]), np.ascontiguousarray(remap[F])


def remove_degenerate(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Drop triangles with a repeated corner or zero area."""
    ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    tri = V[F[ok]]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    keep = np.flatnonzero(ok)[area > 0.0]
    return np.ascontiguousarray(F[keep])


def largest_component(F: np.ndarray) -> np.ndarray:
    """Triangles of the largest face-connected component (drops floating shards)."""
    n_comp, comp = igl.facet_components(np.asarray(F, dtype=np.int64))
    if n_comp <= 1:
        return F
    counts = np.bincount(np.asarray(comp))
    return np.ascontiguousarray(F[np.asarray(comp) == int(np.argmax(counts))])


def prepare_open_surface(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Make (V,F) a single clean open surface: no degenerates, no shards, no orphans.

    Reconstruction of the `expand_remesh_topologies` helper of the same name
    that `datasets/remesh.py::make_crop` calls before and after trimming.  Kept
    conservative on purpose: it only removes, never welds or re-orients, so on
    an already-clean patch it is the identity.
    """
    V, F = as_arrays(V, F)
    F = remove_degenerate(V, F)
    F = largest_component(F)
    return remove_unreferenced(V, F)


def vertex_adjacency(n_verts: int, F: np.ndarray) -> list[np.ndarray]:
    edges = np.vstack((F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    both = np.vstack((edges, edges[:, ::-1]))
    order = np.argsort(both[:, 0], kind="stable")
    both = both[order]
    starts = np.searchsorted(both[:, 0], np.arange(n_verts + 1))
    return [both[starts[i]:starts[i + 1], 1] for i in range(n_verts)]


def smooth_simple(V: np.ndarray, F: np.ndarray, iterations: int) -> np.ndarray:
    """Umbrella smoothing, `open3d.filter_smooth_simple`: v <- (v + sum_N v_j)/(1+|N|)."""
    adj = vertex_adjacency(len(V), F)
    idx = np.concatenate(adj) if len(V) else np.zeros(0, dtype=np.int64)
    owner = np.repeat(np.arange(len(V)), [len(a) for a in adj])
    deg = np.asarray([len(a) for a in adj], dtype=np.float64)
    out = np.array(V, dtype=np.float64)
    for _ in range(iterations):
        acc = np.zeros_like(out)
        np.add.at(acc, owner, out[idx])
        out = (out + acc) / (1.0 + deg)[:, None]
    return out


def decimate_to(V: np.ndarray, F: np.ndarray, target_triangles: int) -> tuple[np.ndarray, np.ndarray]:
    """Quadric-error edge collapse down to `target_triangles` (igl.qslim)."""
    V, F = as_arrays(V, F)
    if target_triangles >= len(F):
        return V, F
    U, G, _, _ = igl.qslim(np.asfortranarray(V), np.asfortranarray(F), max(4, int(target_triangles)))
    U = np.ascontiguousarray(U, dtype=np.float64)
    G = np.ascontiguousarray(G, dtype=np.int32)
    G = remove_degenerate(U, G)
    return remove_unreferenced(U, G)


def subdivide_midpoint(V: np.ndarray, F: np.ndarray, n: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """1-to-4 midpoint split: 4^n triangles, geometry untouched (igl.upsample)."""
    V, F = as_arrays(V, F)
    U, G = igl.upsample(V, np.asarray(F, dtype=np.int64), n)
    return np.ascontiguousarray(U, dtype=np.float64), np.ascontiguousarray(G, dtype=np.int32)


# --- datasets/remesh.py port -------------------------------------------------

def extract_unique_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    return np.unique(np.sort(edges, axis=1), axis=0)


def extract_boundary_vertices(faces: np.ndarray) -> np.ndarray:
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    sorted_edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(sorted_edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    return np.unique(boundary_edges)


def build_edge_graph(verts: np.ndarray, faces: np.ndarray) -> list[list[tuple[int, float]]]:
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(len(verts))]
    for a_raw, b_raw in extract_unique_edges(faces):
        a = int(a_raw)
        b = int(b_raw)
        weight = float(np.linalg.norm(verts[a] - verts[b]))
        adjacency[a].append((b, weight))
        adjacency[b].append((a, weight))
    return adjacency


def trim_far_boundary_band(
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    radius_percentile: float = BOUNDARY_RADIUS_PERCENTILE,
    trim_distance_ratio: float = TRIM_DISTANCE_RATIO,
) -> tuple[np.ndarray, np.ndarray]:
    """Geodesic band of `trim_distance_ratio` x bbox diag off the outer boundary."""
    boundary_vertices = extract_boundary_vertices(faces)
    if boundary_vertices.size == 0:
        return verts, faces

    center = np.median(verts, axis=0)
    boundary_radius = np.linalg.norm(verts[boundary_vertices] - center, axis=1)
    radius_threshold = float(np.percentile(boundary_radius, radius_percentile))
    seed_vertices = boundary_vertices[boundary_radius >= radius_threshold]
    if seed_vertices.size == 0:
        return verts, faces

    bbox_diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    max_distance = bbox_diag * float(trim_distance_ratio)
    if max_distance <= 0.0:
        return verts, faces

    adjacency = build_edge_graph(verts, faces)
    distances = np.full(len(verts), np.inf, dtype=np.float64)
    heap: list[tuple[float, int]] = []

    for source in seed_vertices.tolist():
        source = int(source)
        distances[source] = 0.0
        heapq.heappush(heap, (0.0, source))

    while heap:
        current_distance, node = heapq.heappop(heap)
        if current_distance != distances[node] or current_distance > max_distance:
            continue
        for neighbor, weight in adjacency[node]:
            new_distance = current_distance + weight
            if new_distance < distances[neighbor] and new_distance <= max_distance:
                distances[neighbor] = new_distance
                heapq.heappush(heap, (new_distance, neighbor))

    keep_mask = np.isinf(distances)
    if keep_mask.mean() < MIN_KEEP_RATIO:
        return verts, faces

    kept_indices = np.flatnonzero(keep_mask)
    if kept_indices.size == 0:
        return verts, faces

    face_mask = np.all(keep_mask[faces], axis=1)
    cropped_faces = faces[face_mask]
    if cropped_faces.size == 0:
        return verts, faces

    remap = -np.ones(len(verts), dtype=np.int32)
    remap[kept_indices] = np.arange(len(kept_indices), dtype=np.int32)
    cropped_verts = verts[kept_indices]
    cropped_faces = remap[cropped_faces]
    return cropped_verts, cropped_faces


def make_crop(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`crop`: boundary-only trim, `datasets/remesh.py::make_crop`."""
    Vp, Fp = prepare_open_surface(V, F)
    Vc, Fc = trim_far_boundary_band(Vp, Fp)
    return prepare_open_surface(Vc, Fc)
