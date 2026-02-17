#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import open3d as o3d
from joblib import Parallel, delayed
import multiprocessing

# ============================================================
# CONFIG
# ============================================================

DATA_CANON = Path(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/REMESH/data_CANONICAL"
)

GRID_CSV = Path("grid_identity_results.csv")
OUT_CSV  = Path("chamfer_identity_results.csv")

USE_ICP   = True        # ICP only for inter-subject
ICP_ITERS = 20
N_POINTS  = 8000        # 3000 for fast debug, 8000–10000 final

N_JOBS = max(1, multiprocessing.cpu_count() - 1)

# ============================================================
# GLOBAL CACHES (per process)
# ============================================================

_mesh_cache = {}
_pcd_cache  = {}

# ============================================================
# UTILS
# ============================================================

def load_mesh(subject_id, variant):
    npz = np.load(DATA_CANON / f"{subject_id}_{variant}.npz")
    V = npz["V"]
    F = npz["F"]

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(V),
        triangles=o3d.utility.Vector3iVector(F),
    )
    mesh.compute_vertex_normals()
    return mesh


def get_mesh(subject_id, variant):
    key = (subject_id, variant)
    if key not in _mesh_cache:
        _mesh_cache[key] = load_mesh(subject_id, variant)
    return _mesh_cache[key]


def get_pcd(mesh, key):
    if key not in _pcd_cache:
        _pcd_cache[key] = mesh.sample_points_uniformly(N_POINTS)
    return _pcd_cache[key]


def chamfer_distance(mesh_A, mesh_B, cache_key_A=None, cache_key_B=None):
    """
    Symmetric Chamfer distance via point sampling.
    Uses cached point clouds when possible.
    """
    if cache_key_A is not None:
        pcd_A = get_pcd(mesh_A, cache_key_A)
    else:
        pcd_A = mesh_A.sample_points_uniformly(N_POINTS)

    if cache_key_B is not None:
        pcd_B = get_pcd(mesh_B, cache_key_B)
    else:
        pcd_B = mesh_B.sample_points_uniformly(N_POINTS)

    d_AB = np.asarray(pcd_A.compute_point_cloud_distance(pcd_B))
    d_BA = np.asarray(pcd_B.compute_point_cloud_distance(pcd_A))

    return float(d_AB.mean() + d_BA.mean())


def rigid_icp(source, target):
    """
    Rigid ICP alignment (no scaling).
    """
    pcd_src = source.sample_points_uniformly(N_POINTS)
    pcd_tgt = target.sample_points_uniformly(N_POINTS)

    reg = o3d.pipelines.registration.registration_icp(
        pcd_src,
        pcd_tgt,
        max_correspondence_distance=0.05,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=ICP_ITERS
        ),
    )

    source_aligned = source.transform(reg.transformation)
    return source_aligned


# ============================================================
# WORKER
# ============================================================

def process_row(row):
    sid_A = row.subject_A
    sid_B = row.subject_B
    va    = row.variant_A
    vb    = row.variant_B

    mesh_A = get_mesh(sid_A, va)
    mesh_B = get_mesh(sid_B, vb)

    # ICP ONLY for inter-subject
    if USE_ICP and sid_A != sid_B:
        mesh_A = rigid_icp(mesh_A, mesh_B)
        d = chamfer_distance(mesh_A, mesh_B)
    else:
        key_A = (sid_A, va)
        key_B = (sid_B, vb)
        d = chamfer_distance(mesh_A, mesh_B, key_A, key_B)

    return {
        "subject_A": sid_A,
        "subject_B": sid_B,
        "variant_A": va,
        "variant_B": vb,
        "distance": d,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("🚀 Running parallel Chamfer evaluation (aligned to Grid)")
    print(f"🧠 Using {N_JOBS} CPU workers")

    df_grid = pd.read_csv(GRID_CSV)

    rows = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(process_row)(row)
        for _, row in tqdm(df_grid.iterrows(), total=len(df_grid))
    )

    df_ch = pd.DataFrame(rows)
    df_ch.to_csv(OUT_CSV, index=False)

    print(f"\n✅ Saved Chamfer results to {OUT_CSV.resolve()}")
    print("Rows:", len(df_ch))


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()
