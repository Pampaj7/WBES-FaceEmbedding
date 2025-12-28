#!/usr/bin/env python3
import os
import numpy as np
import open3d as o3d

# PATHS TO ADAPT
INPUT_DIR = "GT_ready/npz_data_cropped"        # NPZ WITHOUT operators (keys: V, F)
OUTPUT_DIR = "REMESH/npz_data_topo_500"        # output dir for subjects + variants
N_SUBJECTS = 500                              # for now: first 3 subjects for testing

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def load_npz_as_mesh(path: str) -> o3d.geometry.TriangleMesh:
    """Load an NPZ with keys V, F into an Open3D TriangleMesh."""
    d = np.load(path)
    V = d["V"]
    F = d["F"]

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(V),
        triangles=o3d.utility.Vector3iVector(F)
    )
    mesh.compute_vertex_normals()
    return mesh

def copy_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Safe copy of an Open3D mesh (works also with CUDA builds, no .clone())."""
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    new = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(V.copy()),
        triangles=o3d.utility.Vector3iVector(F.copy())
    )
    new.compute_vertex_normals()
    return new

def save_mesh_npz(mesh: o3d.geometry.TriangleMesh, out_path: str) -> None:
    """Save an Open3D mesh as NPZ with keys V, F."""
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    np.savez(out_path, V=V, F=F)

# -------------------------------------------------------------
# Remesh
# -------------------------------------------------------------
def make_remesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Smooth + quadric decimation to change triangulation."""
    m = copy_mesh(mesh)

    # light smoothing to avoid artifacts
    m = m.filter_smooth_simple(number_of_iterations=2)

    # decimate to ~70% of original triangles (but keep a minimum)
    n_tris = np.asarray(m.triangles).shape[0]
    target = max(int(n_tris * 0.7), 2000)
    m = m.simplify_quadric_decimation(target_number_of_triangles=target)

    m.compute_vertex_normals()
    return m

# -------------------------------------------------------------
# Crop (z-dimension)
# -------------------------------------------------------------
def make_crop(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """
    Keep only a central band along z.
    If crop becomes degenerate, fall back to original mesh.
    """
    m = copy_mesh(mesh)

    V = np.asarray(m.vertices)
    z = V[:, 2]

    zmin, zmax = np.percentile(z, [10, 90])
    mask = (z >= zmin) & (z <= zmax)
    idx = np.where(mask)[0]

    # If crop is too aggressive (too few vertices), just return original
    if idx.size < 0.3 * V.shape[0]:
        m.compute_vertex_normals()
        return m

    tris = np.asarray(m.triangles)
    keep = np.all(np.isin(tris, idx), axis=1)
    tris_new = tris[keep]

    # If there are no triangles left, return original
    if tris_new.size == 0:
        m.compute_vertex_normals()
        return m

    # Remap indices to [0, N_crop)
    remap = {old: i for i, old in enumerate(idx)}
    tris_new = np.vectorize(remap.get)(tris_new)

    V2 = V[idx]
    new_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(V2),
        triangles=o3d.utility.Vector3iVector(tris_new)
    )
    new_mesh.compute_vertex_normals()
    return new_mesh

# -------------------------------------------------------------
# Noise
# -------------------------------------------------------------
def make_noisy(mesh: o3d.geometry.TriangleMesh, std: float = 0.003) -> o3d.geometry.TriangleMesh:
    """Add small Gaussian noise to vertex positions, same topology."""
    m = copy_mesh(mesh)

    V = np.asarray(m.vertices)
    bbox = m.get_axis_aligned_bounding_box()
    scale = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())

    noise = np.random.normal(0, std * scale, size=V.shape)
    V2 = V + noise

    m.vertices = o3d.utility.Vector3dVector(V2)
    m.compute_vertex_normals()
    return m

# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def main():
    all_files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".npz"))
    selected = all_files[:N_SUBJECTS]

    if not selected:
        print("No NPZ files found in INPUT_DIR.")
        return

    for i, fname in enumerate(selected, 1):
        subj_id = fname.replace(".npz", "")
        path = os.path.join(INPUT_DIR, fname)

        print(f"[{i}/{len(selected)}] Processing {subj_id}")

        # Load base mesh
        mesh = load_npz_as_mesh(path)

        # Save original
        save_mesh_npz(mesh, os.path.join(OUTPUT_DIR, f"{subj_id}_original.npz"))

        # Remesh
        mesh_r = make_remesh(mesh)
        save_mesh_npz(mesh_r, os.path.join(OUTPUT_DIR, f"{subj_id}_remesh.npz"))

        # Crop
        mesh_c = make_crop(mesh)
        save_mesh_npz(mesh_c, os.path.join(OUTPUT_DIR, f"{subj_id}_crop.npz"))

        # Noisy
        mesh_n = make_noisy(mesh)
        save_mesh_npz(mesh_n, os.path.join(OUTPUT_DIR, f"{subj_id}_noisy.npz"))

    print(f"\nDone generating variants for {len(selected)} subjects.\n")

if __name__ == "__main__":
    main()
