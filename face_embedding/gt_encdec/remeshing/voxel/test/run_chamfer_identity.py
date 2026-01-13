#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
import random
from tqdm import tqdm

import open3d as o3d

# ============================================================
# CONFIG
# ============================================================

DATA_CANON = Path(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/REMESH/data_CANONICAL"
)

OUT_CSV = Path("chamfer_identity_results.csv")

N_SUBJECTS = 50          # subset size
N_INTER_PER_SUBJ = 5     # B random per A

USE_ICP = True           # rigid ICP (reviewer-friendly)
ICP_ITERS = 20

random.seed(42)
np.random.seed(42)

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


def chamfer_distance(mesh_A, mesh_B, n_points=10000):
    """
    Symmetric Chamfer distance using point sampling.
    """
    pcd_A = mesh_A.sample_points_uniformly(n_points)
    pcd_B = mesh_B.sample_points_uniformly(n_points)

    d_AB = np.asarray(pcd_A.compute_point_cloud_distance(pcd_B))
    d_BA = np.asarray(pcd_B.compute_point_cloud_distance(pcd_A))

    return float(d_AB.mean() + d_BA.mean())


def rigid_icp(source, target):
    """
    Rigid ICP alignment (no scaling).
    """
    pcd_src = source.sample_points_uniformly(10000)
    pcd_tgt = target.sample_points_uniformly(10000)

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
# MAIN
# ============================================================

def main():
    subjects = sorted(
        p.stem.replace("_original", "")
        for p in DATA_CANON.glob("*_original.npz")
    )

    subjects = subjects[:N_SUBJECTS]
    print(f"📦 Using {len(subjects)} subjects")

    rows = []

    for sid_A in tqdm(subjects, desc="Subjects A"):
        mesh_A_orig = load_mesh(sid_A, "original")
        mesh_A_rem  = load_mesh(sid_A, "remesh")
        mesh_A_crop = load_mesh(sid_A, "crop")

        # ----------------------------
        # INTRA-subject
        # ----------------------------
        for var, mesh_A_var in [
            ("remesh", mesh_A_rem),
            ("crop",   mesh_A_crop),
        ]:
            mA = mesh_A_var
            mB = mesh_A_orig

            if USE_ICP:
                mA = rigid_icp(mA, mB)

            d = chamfer_distance(mA, mB)

            rows.append({
                "subject_A": sid_A,
                "subject_B": sid_A,
                "variant_A": var,
                "variant_B": "original",
                "distance": d,
            })

        # ----------------------------
        # INTER-subject
        # ----------------------------
        others = [s for s in subjects if s != sid_A]
        Bs = random.sample(others, N_INTER_PER_SUBJ)

        for sid_B in Bs:
            mesh_B = load_mesh(sid_B, "original")

            mA = mesh_A_orig
            mB = mesh_B

            if USE_ICP:
                mA = rigid_icp(mA, mB)

            d = chamfer_distance(mA, mB)

            rows.append({
                "subject_A": sid_A,
                "subject_B": sid_B,
                "variant_A": "original",
                "variant_B": "original",
                "distance": d,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print(f"\n✅ Saved Chamfer results to {OUT_CSV.resolve()}")
    print("Rows:", len(df))


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()
