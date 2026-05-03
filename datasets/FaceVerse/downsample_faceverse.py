#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import fnmatch
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "numpy is required. Run this script with "
        "/home/lpampaloni/miniconda3/envs/3d/bin/python."
    ) from exc

try:
    import igl
except ImportError as exc:
    raise SystemExit(
        "igl is required. Run this script with "
        "/home/lpampaloni/miniconda3/envs/3d/bin/python."
    ) from exc


DEFAULT_INPUT_ROOT = Path(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/FaceVerse/extracted/detail"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/FaceVerse/downsampled"
)
RECOMMENDED_PYTHON = "/home/lpampaloni/miniconda3/envs/3d/bin/python"
DEFAULT_OUTPUT_PLY_FORMAT = "binary_little_endian"
DEFAULT_NORMALIZATION_MODE = "none"
MAX_AUTO_JOBS = 16
PLY_DTYPE_MAP = {
    "char": "i1",
    "uchar": "u1",
    "short": "i2",
    "ushort": "u2",
    "int": "i4",
    "uint": "u4",
    "float": "f4",
    "double": "f8",
}


@dataclass(frozen=True)
class PlyHeader:
    fmt: str
    header_lines: int
    header_bytes: int
    vertex_count: int
    face_count: int
    vertex_properties: tuple[tuple[str, str], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Downsample FaceVerse meshes with libigl decimation so the output "
            "remains a valid triangle mesh and DiffusionNet operators can be "
            "computed afterwards."
        )
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory containing the source .ply meshes.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root for the downsampled triangle meshes.",
    )
    parser.add_argument(
        "--target_points",
        type=int,
        default=10000,
        help="Approximate target vertex count. Used only if --target_faces is not set.",
    )
    parser.add_argument(
        "--target_faces",
        type=int,
        default=None,
        help="Target face count for libigl decimation. Defaults to about 2 * target_points.",
    )
    parser.add_argument(
        "--pattern",
        default="*.ply",
        help="Optional glob applied to input filenames.",
    )
    parser.add_argument(
        "--stem_suffix",
        default=None,
        help="Optional suffix filter on the stem, e.g. '_01'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs even if the target .ply already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N meshes after sorting. Useful for quick tests.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help=(
            "Number of worker threads. Use 0 for auto mode "
            f"(up to {MAX_AUTO_JOBS} workers)."
        ),
    )
    parser.add_argument(
        "--output_ply_format",
        choices=("binary_little_endian", "ascii"),
        default=DEFAULT_OUTPUT_PLY_FORMAT,
        help="Output PLY encoding for the downsampled meshes.",
    )
    parser.add_argument(
        "--normalization_mode",
        choices=(
            "none",
            "center_bbox",
            "center_centroid",
            "center_bbox_unit",
            "center_centroid_unit",
        ),
        default=DEFAULT_NORMALIZATION_MODE,
        help="Optional normalization applied after decimation.",
    )
    args = parser.parse_args()
    if args.jobs < 0:
        parser.error("--jobs must be >= 0")
    if args.target_points <= 0:
        parser.error("--target_points must be > 0")
    if args.target_faces is not None and args.target_faces <= 0:
        parser.error("--target_faces must be > 0")
    return args


def parse_ply_header(path: Path) -> PlyHeader:
    fmt = None
    vertex_count = None
    face_count = 0
    vertex_properties: list[tuple[str, str]] = []
    current_element = None
    header_lines = 0
    header_bytes = 0

    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while parsing header: {path}")
            header_lines += 1
            header_bytes += len(line)
            text = line.decode("ascii", errors="strict").strip()

            if text.startswith("format "):
                parts = text.split()
                if len(parts) < 2:
                    raise ValueError(f"Malformed format line in {path}")
                fmt = parts[1]
            elif text.startswith("element "):
                parts = text.split()
                if len(parts) != 3:
                    raise ValueError(f"Malformed element line in {path}")
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
                elif current_element == "face":
                    face_count = int(parts[2])
            elif text.startswith("property ") and current_element == "vertex":
                parts = text.split()
                if len(parts) == 3:
                    vertex_properties.append((parts[2], parts[1]))
            elif text == "end_header":
                break

    if fmt is None or vertex_count is None:
        raise ValueError(f"Missing required PLY header information in {path}")

    return PlyHeader(
        fmt=fmt,
        header_lines=header_lines,
        header_bytes=header_bytes,
        vertex_count=vertex_count,
        face_count=face_count,
        vertex_properties=tuple(vertex_properties),
    )


def load_vertex_positions(path: Path, header: PlyHeader) -> np.ndarray:
    property_names = [name for name, _ in header.vertex_properties]
    try:
        x_index = property_names.index("x")
        y_index = property_names.index("y")
        z_index = property_names.index("z")
    except ValueError as exc:
        raise ValueError(f"PLY file does not expose x/y/z vertex properties: {path}") from exc

    if header.fmt == "ascii":
        points = np.loadtxt(
            path,
            skiprows=header.header_lines,
            max_rows=header.vertex_count,
            usecols=(x_index, y_index, z_index),
            dtype=np.float32,
        )
        return np.atleast_2d(points)

    if header.fmt == "binary_little_endian":
        dtype_fields = []
        for name, ply_type in header.vertex_properties:
            if ply_type not in PLY_DTYPE_MAP:
                raise ValueError(f"Unsupported vertex property type '{ply_type}' in {path}")
            dtype_fields.append((name, "<" + PLY_DTYPE_MAP[ply_type]))
        vertex_dtype = np.dtype(dtype_fields)
        raw = np.fromfile(
            path,
            dtype=vertex_dtype,
            count=header.vertex_count,
            offset=header.header_bytes,
        )
        points = np.column_stack((raw["x"], raw["y"], raw["z"])).astype(np.float32, copy=False)
        return np.atleast_2d(points)

    raise ValueError(
        f"Unsupported PLY format '{header.fmt}' in {path}. "
        "Only ascii and binary_little_endian are supported."
    )


def normalize_points(points: np.ndarray, normalization_mode: str) -> np.ndarray:
    normalized = np.asarray(points, dtype=np.float32)
    if normalization_mode == "none":
        return normalized

    mins = normalized.min(axis=0)
    maxs = normalized.max(axis=0)
    span = maxs - mins
    bbox_center = (mins + maxs) * 0.5
    centroid = normalized.mean(axis=0)

    if normalization_mode.startswith("center_bbox"):
        normalized = normalized - bbox_center
    elif normalization_mode.startswith("center_centroid"):
        normalized = normalized - centroid
    else:
        raise ValueError(f"Unsupported normalization mode: {normalization_mode}")

    if normalization_mode.endswith("_unit"):
        scale = float(np.max(span))
        if scale > 0.0:
            normalized = normalized / scale

    return normalized.astype(np.float32, copy=False)


def discover_meshes(
    input_root: Path,
    limit: int | None,
    pattern: str,
    stem_suffix: str | None,
) -> list[Path]:
    meshes = []
    for path in sorted(input_root.rglob("*.ply")):
        if not path.is_file():
            continue
        if pattern and not fnmatch.fnmatch(path.name, pattern):
            continue
        if stem_suffix and not path.stem.endswith(stem_suffix):
            continue
        meshes.append(path)
    if limit is not None:
        return meshes[:limit]
    return meshes


def build_output_path(input_path: Path, input_root: Path, output_root: Path) -> Path:
    return output_root / input_path.relative_to(input_root)


def resolve_jobs(requested_jobs: int, mesh_count: int) -> int:
    if mesh_count <= 0:
        return 1
    if requested_jobs == 0:
        return max(1, min(mesh_count, min(os.cpu_count() or 1, MAX_AUTO_JOBS)))
    return max(1, min(requested_jobs, mesh_count))


def resolve_target_faces(target_points: int, target_faces: int | None) -> int:
    if target_faces is not None:
        return int(target_faces)
    return max(4, 2 * int(target_points))


def load_triangle_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = igl.read_triangle_mesh(str(path))
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Invalid vertex array in {path}: {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0:
        raise ValueError(f"Mesh has no valid triangular faces: {path}")
    return vertices, faces


def write_mesh_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, ply_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int32)

    if ply_format == "ascii":
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("ply\n")
            handle.write("format ascii 1.0\n")
            handle.write(f"element vertex {vertices.shape[0]}\n")
            handle.write("property float x\n")
            handle.write("property float y\n")
            handle.write("property float z\n")
            handle.write(f"element face {faces.shape[0]}\n")
            handle.write("property list uchar int vertex_indices\n")
            handle.write("end_header\n")
            np.savetxt(handle, vertices, fmt="%.8f %.8f %.8f")
            for tri in faces:
                handle.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")
        return

    if ply_format == "binary_little_endian":
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {vertices.shape[0]}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            f"element face {faces.shape[0]}\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        ).encode("ascii")
        face_dtype = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
        face_records = np.empty(faces.shape[0], dtype=face_dtype)
        face_records["count"] = 3
        face_records["indices"] = faces.astype("<i4", copy=False)
        with path.open("wb") as handle:
            handle.write(header)
            handle.write(vertices.astype("<f4", copy=False).tobytes(order="C"))
            handle.write(face_records.tobytes(order="C"))
        return

    raise ValueError(f"Unsupported output PLY format: {ply_format}")


def save_manifest(output_root: Path, input_root: Path, target_points: int, target_faces: int) -> Path:
    manifest_path = output_root / "downsample_manifest.npz"
    output_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        manifest_path,
        input_root=str(input_root),
        target_points=np.int32(target_points),
        target_faces=np.int32(target_faces),
        method="igl.decimate",
    )
    return manifest_path


def process_mesh_worker(
    input_path_str: str,
    input_root_str: str,
    output_root_str: str,
    target_faces: int,
    output_ply_format: str,
    normalization_mode: str,
) -> dict[str, object]:
    started_at = time.time()
    temp_path = None
    relative_path = str(input_path_str)

    try:
        input_path = Path(input_path_str)
        input_root = Path(input_root_str)
        output_root = Path(output_root_str)
        relative_path = str(input_path.relative_to(input_root))
        output_path = build_output_path(input_path, input_root, output_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = Path(f"{output_path}.tmp.{os.getpid()}.{time.time_ns()}")
        vertices, faces = load_triangle_mesh(input_path)
        source_vertex_count = int(vertices.shape[0])
        source_face_count = int(faces.shape[0])

        if source_face_count > target_faces:
            dec_vertices, dec_faces, _, _ = igl.decimate(
                np.asfortranarray(vertices),
                np.asfortranarray(faces),
                int(target_faces),
            )
            dec_vertices = np.asarray(dec_vertices, dtype=np.float32)
            dec_faces = np.asarray(dec_faces, dtype=np.int32)
            if dec_faces.size == 0:
                raise ValueError("igl.decimate returned an empty face array.")
        else:
            dec_vertices = vertices.astype(np.float32, copy=False)
            dec_faces = faces.astype(np.int32, copy=False)

        dec_vertices = normalize_points(dec_vertices, normalization_mode)
        write_mesh_ply(temp_path, dec_vertices, dec_faces, output_ply_format)
        os.replace(temp_path, output_path)

        return {
            "status": "processed",
            "relative_path": relative_path,
            "output_path": str(output_path),
            "vertex_count": int(dec_vertices.shape[0]),
            "face_count": int(dec_faces.shape[0]),
            "source_vertex_count": source_vertex_count,
            "source_face_count": source_face_count,
            "elapsed_s": time.time() - started_at,
        }
    except Exception as exc:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
        return {
            "status": "failed",
            "relative_path": relative_path,
            "error_message": str(exc),
            "elapsed_s": time.time() - started_at,
        }


def print_summary(
    processed_vertices: list[int],
    processed_faces: list[int],
    skipped: int,
    failed: list[tuple[str, str]],
) -> None:
    print("\nSummary")
    print(f"Processed: {len(processed_vertices)}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {len(failed)}")
    if processed_vertices:
        print(f"Min final vertices: {min(processed_vertices)}")
        print(f"Max final vertices: {max(processed_vertices)}")
        print(f"Mean final vertices: {sum(processed_vertices) / len(processed_vertices):.2f}")
    else:
        print("Min final vertices: n/a")
        print("Max final vertices: n/a")
        print("Mean final vertices: n/a")
    if processed_faces:
        print(f"Min final faces: {min(processed_faces)}")
        print(f"Max final faces: {max(processed_faces)}")
        print(f"Mean final faces: {sum(processed_faces) / len(processed_faces):.2f}")
    else:
        print("Min final faces: n/a")
        print("Max final faces: n/a")
        print("Mean final faces: n/a")

    if failed:
        print("\nFailed files")
        for relative_path, error_message in failed:
            print(f"- {relative_path}: {error_message}")


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    target_faces = resolve_target_faces(args.target_points, args.target_faces)

    if not input_root.exists():
        print(f"Input root does not exist: {input_root}", file=sys.stderr)
        return 1

    meshes = discover_meshes(
        input_root=input_root,
        limit=args.limit,
        pattern=args.pattern,
        stem_suffix=args.stem_suffix,
    )
    if not meshes:
        print(f"No matching .ply files found under {input_root}", file=sys.stderr)
        return 1

    skipped = 0
    meshes_to_process: list[Path] = []
    for mesh_path in meshes:
        output_path = build_output_path(mesh_path, input_root, output_root)
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        meshes_to_process.append(mesh_path)

    manifest_path = save_manifest(output_root, input_root, args.target_points, target_faces)

    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    print("Method: igl.decimate")
    print(f"Output PLY encoding: {args.output_ply_format}")
    print(f"Normalization mode: {args.normalization_mode}")
    print(f"Target points (approx): {args.target_points}")
    print(f"Target faces: {target_faces}")
    print(f"Meshes discovered: {len(meshes)}")
    print(f"Queued for processing: {len(meshes_to_process)}")
    print(f"Skipped because output exists: {skipped}")
    print(f"Manifest: {manifest_path}")

    if not meshes_to_process:
        print_summary([], [], skipped, [])
        return 0

    jobs = resolve_jobs(args.jobs, len(meshes_to_process))
    print(f"Worker threads: {jobs}")

    processed_vertices: list[int] = []
    processed_faces: list[int] = []
    failed: list[tuple[str, str]] = []

    with cf.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(
                process_mesh_worker,
                str(mesh_path),
                str(input_root),
                str(output_root),
                target_faces,
                args.output_ply_format,
                args.normalization_mode,
            )
            for mesh_path in meshes_to_process
        ]

        for index, future in enumerate(cf.as_completed(futures), start=1):
            result = future.result()
            if result["status"] == "processed":
                processed_vertices.append(int(result["vertex_count"]))
                processed_faces.append(int(result["face_count"]))
            else:
                failed.append((str(result["relative_path"]), str(result["error_message"])))

            if index % 25 == 0 or index == len(futures):
                print(
                    f"Progress: {index}/{len(futures)} | "
                    f"processed={len(processed_vertices)} | failed={len(failed)}"
                )

    print_summary(processed_vertices, processed_faces, skipped, failed)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
