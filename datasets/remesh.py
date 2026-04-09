#!/usr/bin/env python3
"""Regenerate the REMESH crop variant with a boundary-only trim.

The previous `crop` variant removed a global z-band and could cut away facial
regions such as nose and mouth. This script instead trims only a thin band near
the outermost boundary vertices, leaving the full z-range intact.
"""

from __future__ import annotations

import argparse
import heapq
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

try:
    from datasets.expand_remesh_topologies import make_mesh, mesh_to_arrays, prepare_open_surface
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from datasets.expand_remesh_topologies import make_mesh, mesh_to_arrays, prepare_open_surface


REPO_ROOT = Path("/equilibrium/lpampaloni/WBES-FaceEmbedding")
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "REMESH" / "npz_data_topo_500"
DEFAULT_N_CORES = max(1, min(8, os.cpu_count() or 4))

BOUNDARY_RADIUS_PERCENTILE = 70.0
TRIM_DISTANCE_RATIO = 0.06
MIN_KEEP_RATIO = 0.85


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace *_GTready_crop.npz with a conservative boundary-based crop."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing *_GTready_original.npz and *_GTready_crop.npz.",
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=500,
        help="Use the first N sorted subjects unless --subject-id is provided.",
    )
    parser.add_argument(
        "--subject-id",
        action="append",
        default=[],
        help="Optional explicit subject id(s), e.g. --subject-id id0000.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing crop files.",
    )
    parser.add_argument(
        "--n-cores",
        type=int,
        default=DEFAULT_N_CORES,
        help="Number of worker processes to use.",
    )
    return parser.parse_args()


def list_original_meshes(dataset_dir: Path, n_subjects: int, subject_ids: list[str]) -> list[Path]:
    all_originals = sorted(dataset_dir.glob("*_GTready_original.npz"))
    if not all_originals:
        raise FileNotFoundError(f"No *_GTready_original.npz files found in {dataset_dir}")

    if subject_ids:
        wanted = set(subject_ids)
        selected = [path for path in all_originals if path.stem.split("_GTready")[0] in wanted]
        missing = sorted(wanted - {path.stem.split("_GTready")[0] for path in selected})
        if missing:
            raise FileNotFoundError(f"Missing subjects: {missing}")
        return selected

    return all_originals[: max(0, int(n_subjects))]


def load_npz_as_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    data = np.load(path, allow_pickle=False)
    return make_mesh(
        np.asarray(data["V"], dtype=np.float64),
        np.asarray(data["F"], dtype=np.int32),
    )


def save_mesh_npz(mesh: o3d.geometry.TriangleMesh, out_path: Path) -> None:
    verts, faces = mesh_to_arrays(mesh)
    np.savez(out_path, V=verts, F=faces)


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


def make_crop(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    prepared = prepare_open_surface(mesh)
    verts, faces = mesh_to_arrays(prepared)
    cropped_verts, cropped_faces = trim_far_boundary_band(verts, faces)
    cropped_mesh = make_mesh(cropped_verts, cropped_faces)
    return prepare_open_surface(cropped_mesh)


def crop_output_path(original_path: Path) -> Path:
    subject_id = original_path.stem.split("_GTready")[0]
    return original_path.with_name(f"{subject_id}_GTready_crop.npz")


def process_subject(task: tuple[str, bool]) -> tuple[str, str]:
    original_path_str, overwrite = task
    original_path = Path(original_path_str)
    out_path = crop_output_path(original_path)

    try:
        if out_path.exists() and not overwrite:
            return "[skip]", out_path.name

        mesh = load_npz_as_mesh(original_path)
        crop_mesh = make_crop(mesh)
        save_mesh_npz(crop_mesh, out_path)
        verts, faces = mesh_to_arrays(crop_mesh)
        return "[ok]", f"{out_path.name} V={len(verts)} F={len(faces)}"
    except Exception as exc:
        return "[fail]", f"{out_path.name}: {exc}"


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)

    originals = list_original_meshes(
        dataset_dir=dataset_dir,
        n_subjects=args.n_subjects,
        subject_ids=list(args.subject_id),
    )

    print(f"Dataset dir: {dataset_dir}")
    print(f"Subjects:    {len(originals)}")
    print(f"Overwrite:   {bool(args.overwrite)}")
    print(f"Workers:     {max(1, int(args.n_cores))}")
    print(
        "Crop rule:   "
        f"boundary q>={BOUNDARY_RADIUS_PERCENTILE:.0f}%, "
        f"trim={TRIM_DISTANCE_RATIO * 100:.1f}% bbox diag"
    )

    tasks = [(str(path), bool(args.overwrite)) for path in originals]

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    saved = skipped = failed = 0
    failures: list[str] = []
    worker_count = max(1, int(args.n_cores))

    if worker_count == 1:
        results_iter = map(process_subject, tasks)
    else:
        pool = mp.Pool(processes=worker_count)
        results_iter = pool.imap_unordered(process_subject, tasks)

    try:
        for index, (status, message) in enumerate(results_iter, start=1):
            if status == "[ok]":
                saved += 1
            elif status == "[skip]":
                skipped += 1
            else:
                failed += 1
                failures.append(message)
            print(f"[{index}/{len(originals)}] {status} {message}", flush=True)
    finally:
        if worker_count != 1:
            pool.close()
            pool.join()

    print(f"\nDone. Saved={saved} | Skipped={skipped} | Failed={failed}")
    if failures:
        print("Failures:")
        for message in failures[:20]:
            print(f"  - {message}")


if __name__ == "__main__":
    main()
