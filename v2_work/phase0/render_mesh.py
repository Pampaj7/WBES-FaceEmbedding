"""Dependency-light canonical frontal renderer for 3D face meshes.

Renders a mesh to a grayscale-in-RGB uint8 image suitable for feeding 2D
perceptual encoders (ArcFace, CLIP, DINOv2, LPIPS).  numpy + PIL only: no
OpenGL / EGL / pyrender / open3d, so it runs on a headless CPU login node.

Rasterisation
-------------
Painter's algorithm: triangles are sorted far-to-near by centroid depth and
filled with `PIL.ImageDraw.polygon` (C implemented) at SSAA x supersampling,
then Lanczos downsampled.  Shading is two-sided (see `_normals`).  For an
orthographic frontal view of a near-convex face patch this is equivalent to a
z-buffer everywhere except at self-occlusion silhouettes (nostrils, ear
creases), which is well inside the tolerance of perceptual metrics.
`# ponytail: centroid-depth painter's, not a true per-pixel z-buffer; swap in
a tiled numpy z-buffer only if silhouette artefacts ever show up in a metric.`

Orientation convention (verified on datasets/REMESH/npz_data_topo_500)
---------------------------------------------------------------------
The GT-ready meshes are BFM face crops in a right-handed world frame.
Empirically the surface bulges toward **-z**: the single most extreme vertex
in -z sits on the x/y centre line (the nose tip), while the crop border sits
near z = 0.  The face therefore *looks down -z*, i.e. the camera lives at
z = -inf looking along +z.  Rendering with up = +y came out upside down
(eyes below the mouth), so the world frame is **y-down**: up = -y_world.

For a camera with forward = +z_world and up = -y_world, the viewer's right is
cross(forward, up) = cross(+z, -y) = +x_world.  Screen space (right, up,
toward-viewer) is hence

    screen = (+x_world, -y_world, -z_world)          # = AXES below

which has determinant +1, so it stays right-handed and cross products keep
their meaning.  The stored triangle winding yields normals pointing away from
that camera, so normals are flipped to face it (see `_normals`).

API
---
    render_npz(path, size=512, scale=None, center=None, yaw=0.0) -> (size,size,3) uint8
    render_mesh(V, F, size=512, scale=None, center=None) -> (size, size, 3) uint8
    mesh_frame(V) -> (bbox centre (3,), world extent) for a shared frame
    python render_mesh.py <input.npz> <output.png> [--size 512]

`scale`/`center` override the per-mesh bbox normalisation, which otherwise
renders a tighter topology (e.g. `crop`) larger than its own `original` and
biases cross-topology perceptual distances.  `yaw` rotates about the vertical
screen axis (world y) through `center` for multi-view rendering.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

# (world axis index, sign) for screen x / screen y / depth-toward-viewer.
AXES = ((0, +1.0), (1, -1.0), (2, -1.0))
LIGHT_DIR = np.array([0.3, 0.3, 1.0]) / np.linalg.norm([0.3, 0.3, 1.0])
AMBIENT = 0.25
FACE_SPAN = 0.85  # fraction of image height covered by the mesh bbox
BACKGROUND = 0  # black
SSAA = 2  # supersampling factor; 1 disables anti-aliasing


def _to_screen(V: np.ndarray) -> np.ndarray:
    """World vertices -> screen space (x right, y up, z toward viewer)."""
    return np.stack([sign * V[:, ax] for ax, sign in AXES], axis=1)


def _normals(tri: np.ndarray) -> np.ndarray:
    """Unit per-face normals, each flipped to face the camera (two-sided shading).

    Nothing is back-face culled: decimated topologies fold their lip triangles
    inside out, and culling them punches a hole through to the background.
    """
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return np.where(n[:, 2:3] < 0, -n, n)


def mesh_frame(V: np.ndarray) -> tuple[np.ndarray, float]:
    """`(center, scale)` to pass to `render_mesh` to frame other meshes like this one.

    `center` is the world bbox centre, `scale` the world extent that FACE_SPAN of
    the image covers -- same formula `render_mesh` uses for its own bbox.
    """
    V = np.asarray(V, dtype=np.float64)
    centre = (V.min(axis=0) + V.max(axis=0)) / 2.0
    return centre, float(np.ptp(_to_screen(V)[:, :2], axis=0).max())


def _yaw_rotate(V: np.ndarray, deg: float, centre: np.ndarray) -> np.ndarray:
    """Rotate world vertices by `deg` about the vertical screen axis (world y)."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return (np.asarray(V, dtype=np.float64) - centre) @ R.T + centre


def render_mesh(V: np.ndarray, F: np.ndarray, size: int = 512,
                scale: float | None = None, center: np.ndarray | None = None) -> np.ndarray:
    """Render vertices `V` (N,3) / triangles `F` (M,3) to a (size,size,3) uint8 image.

    `center` (world (3,)) and `scale` (world units spanning FACE_SPAN of the
    image) override the mesh's own bbox normalisation; see `mesh_frame`.
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    if V.ndim != 2 or V.shape[1] != 3 or F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"expected V (N,3) and F (M,3), got {V.shape} and {F.shape}")
    if F.size and (F.min() < 0 or F.max() >= len(V)):
        raise ValueError("face indices out of range")

    S = _to_screen(V)
    tri = S[F]  # (M, 3, 3)
    n = _normals(tri)

    if not len(tri):
        raise ValueError("mesh has no triangles")
    shade = AMBIENT + (1.0 - AMBIENT) * np.clip(n @ LIGHT_DIR, 0.0, 1.0)
    gray = np.round(255.0 * shade).astype(np.uint8)

    # Orthographic projection: centre on the vertex centroid, scale so the mesh
    # bounding box spans FACE_SPAN of the image height.
    res = size * SSAA
    centre = S.mean(axis=0) if center is None else _to_screen(np.asarray(center, float)[None])[0]
    extent = np.ptp(S[:, :2], axis=0).max() if scale is None else float(scale)
    px_per_unit = FACE_SPAN * res / max(extent, 1e-12)
    px = (tri[:, :, 0] - centre[0]) * px_per_unit + res / 2.0
    py = res / 2.0 - (tri[:, :, 1] - centre[1]) * px_per_unit

    # Painter's algorithm: far (small screen z) first, near last.
    order = np.argsort(tri[:, :, 2].mean(axis=1), kind="stable")
    polys = np.stack([px, py], axis=2)[order].reshape(-1, 6).tolist()
    grays = gray[order].tolist()

    img = Image.new("L", (res, res), BACKGROUND)
    draw = ImageDraw.Draw(img)
    for poly, g in zip(polys, grays):
        draw.polygon(poly, fill=g)
    if SSAA != 1:
        img = img.resize((size, size), Image.LANCZOS)

    return np.repeat(np.asarray(img, dtype=np.uint8)[:, :, None], 3, axis=2)


def render_npz(path, size: int = 512, scale: float | None = None,
               center: np.ndarray | None = None, yaw: float = 0.0) -> np.ndarray:
    """Render a .npz mesh file holding `V`/`F` or `verts`/`faces`.

    `yaw` (degrees) rotates about the vertical screen axis through `center`
    (the mesh's own bbox centre when `center` is None) before projection.
    """
    with np.load(path) as d:
        keys = ("V", "F") if "V" in d else ("verts", "faces")
        if keys[0] not in d or keys[1] not in d:
            raise KeyError(f"{path}: expected V/F or verts/faces, found {list(d.files)}")
        V, F = d[keys[0]], d[keys[1]]
    if yaw:
        pivot = mesh_frame(V)[0] if center is None else np.asarray(center, float)
        V = _yaw_rotate(V, yaw, pivot)
    return render_mesh(V, F, size=size, scale=scale, center=center)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("input", help="input .npz mesh")
    p.add_argument("output", help="output .png")
    p.add_argument("--size", type=int, default=512)
    a = p.parse_args()
    Image.fromarray(render_npz(a.input, size=a.size)).save(a.output)
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
