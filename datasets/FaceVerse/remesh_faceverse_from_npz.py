#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import math
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = THIS_DIR / "downsampled_with_ops"
DEFAULT_OUTPUT_DIR = THIS_DIR / "remesh10k_geometry"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a topology-alternative FaceVerse variant by reconstructing each "
            "mesh from sampled surface points and simplifying it back to ~10k vertices."
        )
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern", type=str, default="*_01.npz")
    parser.add_argument("--limit", type=int, default=0, help="0 = all matching files")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--surface_points",
        type=int,
        default=20000,
        help="Points sampled from the source mesh before reconstruction",
    )
    parser.add_argument(
        "--poisson_depth",
        type=int,
        default=7,
        help="Open3D Poisson reconstruction depth",
    )
    parser.add_argument(
        "--crop_scale",
        type=float,
        default=1.05,
        help="Scale factor applied to the source mesh AABB before cropping the reconstruction",
    )
    parser.add_argument(
        "--target_faces",
        type=int,
        default=20000,
        help="Target triangle count after decimation",
    )
    parser.add_argument(
        "--normal_radius",
        type=float,
        default=0.08,
        help="Neighborhood radius for point-cloud normal estimation",
    )
    parser.add_argument(
        "--normal_max_nn",
        type=int,
        default=30,
        help="Maximum nearest neighbors for point-cloud normal estimation",
    )
    parser.add_argument(
        "--orient_k",
        type=int,
        default=20,
        help="Neighborhood size for consistent normal orientation",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_remesh_10k",
        help="Suffix appended to each output stem",
    )
    return parser.parse_args()


def _iter_input_paths(input_dir: Path, pattern: str, limit: int) -> list[Path]:
    files = [path for path in sorted(input_dir.glob("*.npz")) if fnmatch.fnmatch(path.name, pattern)]
    if limit > 0:
        files = files[:limit]
    return files


def _load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    with np.load(path, allow_pickle=False) as data:
        if "verts" in data and "faces" in data:
            verts = np.asarray(data["verts"], dtype=np.float64)
            faces = np.asarray(data["faces"], dtype=np.int32)
        elif "V" in data and "F" in data:
            verts = np.asarray(data["V"], dtype=np.float64)
            faces = np.asarray(data["F"], dtype=np.int32)
        else:
            raise KeyError(f"{path.name} does not contain verts/faces or V/F")

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.remove_non_manifold_edges()
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError(f"Empty mesh after cleanup: {path}")
    mesh.compute_vertex_normals()
    return mesh


def _cleanup_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_non_manifold_edges()
    mesh = mesh.remove_unreferenced_vertices()
    return mesh


def _remesh_geometry(
    mesh: o3d.geometry.TriangleMesh,
    *,
    surface_points: int,
    poisson_depth: int,
    crop_scale: float,
    target_faces: int,
    normal_radius: float,
    normal_max_nn: int,
    orient_k: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    src_bbox = mesh.get_axis_aligned_bounding_box()

    point_cloud = mesh.sample_points_uniformly(number_of_points=int(surface_points))
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(normal_radius),
            max_nn=int(normal_max_nn),
        )
    )
    point_cloud.orient_normals_consistent_tangent_plane(int(orient_k))

    rec_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud,
        depth=int(poisson_depth),
    )
    rec_mesh = rec_mesh.crop(src_bbox.scale(float(crop_scale), src_bbox.get_center()))
    rec_mesh = _cleanup_mesh(rec_mesh)
    if len(rec_mesh.vertices) == 0 or len(rec_mesh.triangles) == 0:
        raise RuntimeError("Poisson reconstruction produced an empty cropped mesh")

    if len(rec_mesh.triangles) > int(target_faces):
        rec_mesh = rec_mesh.simplify_quadric_decimation(target_number_of_triangles=int(target_faces))
        rec_mesh = _cleanup_mesh(rec_mesh)

    verts = np.asarray(rec_mesh.vertices, dtype=np.float32)
    faces = np.asarray(rec_mesh.triangles, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError("Invalid remeshed geometry arrays")
    if verts.shape[0] < 1000 or faces.shape[0] < 1000:
        raise RuntimeError(
            f"Remeshed geometry unexpectedly small: verts={verts.shape[0]} faces={faces.shape[0]}"
        )

    meta = {
        "surface_points": int(surface_points),
        "poisson_depth": int(poisson_depth),
        "crop_scale": float(crop_scale),
        "target_faces": int(target_faces),
        "source_vertices": int(len(mesh.vertices)),
        "source_faces": int(len(mesh.triangles)),
        "output_vertices": int(verts.shape[0]),
        "output_faces": int(faces.shape[0]),
        "density_min": float(np.min(densities)) if len(densities) else math.nan,
        "density_max": float(np.max(densities)) if len(densities) else math.nan,
    }
    return verts, faces, meta


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = _iter_input_paths(input_dir=input_dir, pattern=str(args.pattern), limit=int(args.limit))
    if not paths:
        raise RuntimeError(f"No matching FaceVerse inputs found in {input_dir} for pattern={args.pattern!r}")

    print(f"Input dir:  {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Files:      {len(paths)}")

    ok = 0
    skipped = 0
    failed: list[str] = []

    for path in tqdm(paths, desc="FaceVerse remesh", dynamic_ncols=True):
        out_path = output_dir / f"{path.stem}{args.output_suffix}.npz"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            mesh = _load_mesh(path)
            verts, faces, meta = _remesh_geometry(
                mesh,
                surface_points=int(args.surface_points),
                poisson_depth=int(args.poisson_depth),
                crop_scale=float(args.crop_scale),
                target_faces=int(args.target_faces),
                normal_radius=float(args.normal_radius),
                normal_max_nn=int(args.normal_max_nn),
                orient_k=int(args.orient_k),
            )
            payload = {
                "verts": verts,
                "faces": faces,
                "source_file": np.array(str(path.name)),
            }
            for key, value in meta.items():
                payload[f"meta_{key}"] = np.array(value)
            np.savez_compressed(out_path, **payload)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{path.name}: {exc}")

    print(f"Done. ok={ok} skipped={skipped} failed={len(failed)}")
    if failed:
        print("Failures:")
        for item in failed[:20]:
            print(f"  - {item}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")


if __name__ == "__main__":
    main()
