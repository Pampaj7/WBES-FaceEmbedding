import os
import numpy as np
import trimesh

DATA_DIR = os.path.join("datasets", "GT_ready")

def convert_obj_to_npz(obj_path):
    """Crea un file .npz accanto al .obj con vertici e facce."""
    npz_path = obj_path.replace(".obj", ".npz")
    if os.path.exists(npz_path):
        print(f"⏩ Already exists: {os.path.basename(npz_path)}")
        return

    try:
        mesh = trimesh.load(obj_path, process=False)
        np.savez_compressed(
            npz_path,
            verts=mesh.vertices.astype(np.float32),
            faces=mesh.faces.astype(np.int32)
        )
        print(f"✅ Converted: {os.path.basename(obj_path)} → {os.path.basename(npz_path)}")
    except Exception as e:
        print(f"❌ Error converting {obj_path}: {e}")

def main():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".obj")]
    print(f"Found {len(files)} OBJ files in {DATA_DIR}")

    for fname in files:
        convert_obj_to_npz(os.path.join(DATA_DIR, fname))

if __name__ == "__main__":
    main()
