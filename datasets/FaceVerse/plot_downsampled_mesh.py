#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "numpy is required. Run this script with "
        "/home/lpampaloni/miniconda3/envs/3d/bin/python."
    ) from exc

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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
            "Load an ASCII or binary_little_endian downsampled PLY file and "
            "save a PNG with matplotlib."
        )
    )
    parser.add_argument(
        "--input_mesh",
        type=Path,
        required=True,
        help="Input ASCII or binary_little_endian PLY file.",
    )
    parser.add_argument("--output_png", type=Path, required=True, help="Output PNG file.")
    parser.add_argument(
        "--point_size",
        type=float,
        default=1.0,
        help="Marker size for point-cloud visualization.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG resolution in dots per inch.",
    )
    parser.add_argument(
        "--max_mesh_faces",
        type=int,
        default=50000,
        help="Plot faces only when the mesh stays below this triangle count.",
    )
    return parser.parse_args()


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


def load_ascii_faces(path: Path, header: PlyHeader) -> np.ndarray | None:
    if header.face_count <= 0:
        return None

    triangles: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for _ in range(header.header_lines + header.vertex_count):
            next(handle)
        for _ in range(header.face_count):
            line = handle.readline()
            if not line:
                break
            tokens = line.strip().split()
            if not tokens:
                continue
            vertex_per_face = int(tokens[0])
            if vertex_per_face < 3 or len(tokens) < vertex_per_face + 1:
                continue
            indices = [int(token) for token in tokens[1 : vertex_per_face + 1]]
            for offset in range(1, vertex_per_face - 1):
                triangles.append((indices[0], indices[offset], indices[offset + 1]))
    if not triangles:
        return None
    return np.asarray(triangles, dtype=np.int32)


def set_axes_equal(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.5)
    if radius == 0.0:
        radius = 1.0

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def main() -> int:
    args = parse_args()
    input_mesh = args.input_mesh.resolve()
    output_png = args.output_png.resolve()

    header = parse_ply_header(input_mesh)
    vertices = load_vertex_positions(input_mesh, header)
    faces = load_ascii_faces(input_mesh, header) if header.fmt == "ascii" else None

    fig = plt.figure(figsize=(8, 8), dpi=args.dpi)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    plotted_as_mesh = faces is not None and faces.shape[0] <= args.max_mesh_faces
    if plotted_as_mesh:
        triangles = vertices[faces]
        mesh = Poly3DCollection(
            triangles,
            facecolor="#cfcfcf",
            edgecolor="none",
            linewidths=0.0,
            alpha=1.0,
        )
        ax.add_collection3d(mesh)
    else:
        ax.scatter(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            s=args.point_size,
            c="black",
            linewidths=0.0,
            depthshade=False,
        )

    set_axes_equal(ax, vertices)
    ax.view_init(elev=15, azim=35)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    plt.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    if plotted_as_mesh:
        print(f"Saved mesh render to {output_png}")
    else:
        print(f"Saved point-cloud render to {output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
