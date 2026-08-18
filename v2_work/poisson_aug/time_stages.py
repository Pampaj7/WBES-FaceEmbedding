#!/usr/bin/env python
"""Where do the ~1500 s per subject actually go?

Generating the Poisson variants for 1100 subjects costs ~367 single-worker hours at the
measured rate, so it matters whether that time sits in a single-threaded stage or a
parallel one: if it is single-threaded, thread caps cost nothing and the fix is *more
workers*; if it is parallel, more workers would just recreate the load-average problem
that DTU support complained about.

Times one subject stage by stage. Changes no parameter of the recipe -- the corruption it
produces was validated against FaceScape (Hausdorff 0.291/0.322 vs the 0.38 target) and
must stay bit-for-bit the same recipe.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path("/dtu/p1/leopam/WBES-FaceEmbedding")
SRC = ROOT / "datasets/REMESH/npz_data_topo_500"

RECIPE = dict(surface_points=20000, normal_radius=0.08, normal_max_nn=30,
              orient_k=20, poisson_depth=7, crop_scale=1.05, target_faces=20000)


def main() -> None:
    f = sorted(SRC.glob("*_original.npz"))[0]
    d = np.load(f)
    V, F = np.asarray(d["V"], dtype=np.float64), np.asarray(d["F"], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),
                                     o3d.utility.Vector3iVector(F))
    mesh.compute_vertex_normals()
    print(f"{f.name}: {len(V)} verts, {len(F)} faces", flush=True)

    t = time.time()
    pcd = mesh.sample_points_uniformly(number_of_points=RECIPE["surface_points"])
    print(f"  sample_points_uniformly       {time.time()-t:8.1f}s", flush=True)

    t = time.time()
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=RECIPE["normal_radius"], max_nn=RECIPE["normal_max_nn"]))
    print(f"  estimate_normals              {time.time()-t:8.1f}s", flush=True)

    t = time.time()
    pcd.orient_normals_consistent_tangent_plane(RECIPE["orient_k"])
    print(f"  orient_normals_consistent     {time.time()-t:8.1f}s   <-- suspected", flush=True)

    t = time.time()
    rec, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=RECIPE["poisson_depth"])
    print(f"  create_from_point_cloud_poisson {time.time()-t:6.1f}s  ({len(rec.triangles)} tris)",
          flush=True)

    t = time.time()
    bb = mesh.get_axis_aligned_bounding_box()
    rec = rec.crop(bb.scale(RECIPE["crop_scale"], bb.get_center()))
    rec = rec.simplify_quadric_decimation(target_number_of_triangles=RECIPE["target_faces"])
    print(f"  crop + simplify_quadric       {time.time()-t:8.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
