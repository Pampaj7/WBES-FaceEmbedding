import os
import gc
import numpy as np
import pandas as pd
import trimesh
import trimesh.registration as reg
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt
import psutil
import sys

# === CONFIG ===
RAW_DIR = "../../../render3d_Leonardo/data_creator/synthetic_meshes"
READY_DIR = "../../../datasets/GT_ready/"
RESULTS_DIR = "./results_GT_ready/"
BATCH_SIZE = 500
os.makedirs(READY_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def print_memory(tag=""):
    rss = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    print(f"💾 RAM used {rss:.2f} GB {tag}")

# ================================================================
# 1️⃣ IMPOSTA RIFERIMENTO
# ================================================================
files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(".obj")])
ref_mesh = trimesh.load(os.path.join(RAW_DIR, files[0]), process=False)
ref_verts, ref_faces = ref_mesh.vertices.copy(), ref_mesh.faces.copy()
print(f"Reference mesh: {len(ref_verts)} verts, {len(ref_faces)} faces")

del ref_mesh
gc.collect()


# ================================================================
# 2️⃣ ALLINEAMENTO RIGIDO + CENTRATURA (A BLOCCHI)
# ================================================================
offset_sum = np.zeros(3)
count_total = 0

for start in range(0, len(files), BATCH_SIZE):
    batch = files[start:start + BATCH_SIZE]
    print(f"\n🧩 Processing batch {start}-{start + len(batch) - 1}")
    print_memory("(before batch)")

    for fname in tqdm(batch, desc="Aligning batch"):
        path = os.path.join(RAW_DIR, fname)
        mesh = trimesh.load(path, process=False)

        if count_total == 0 and start == 0:
            aligned = ref_verts.copy()
        else:
            _, transformed, _ = reg.procrustes(mesh.vertices, ref_verts, scale=True)
            aligned = transformed.astype(np.float32)

        centroid = aligned.mean(axis=0)
        offset_sum += centroid
        count_total += 1

        np.savez(os.path.join(
            READY_DIR, f"{os.path.splitext(fname)[0]}_temp.npz"),
            verts=aligned, faces=ref_faces)

        # libera memoria in sicurezza (anche se transformed non esiste)
        try:
            del transformed
        except NameError:
            pass
        del mesh, aligned, centroid
        gc.collect()


    print_memory("(after batch)")
    print("🧹 Cleaning up after batch...")


    # clean per batch
    del batch
    gc.collect()

    # “hard reset” della memoria frammentata (Python 3.8+)
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass

    print_memory("(post trim)")
    print(f"✅ Batch {start}-{start + BATCH_SIZE - 1} done")

del ref_verts, ref_faces
gc.collect()
print_memory("(after all batches)")

# ================================================================
# 3️⃣ CALCOLO OFFSET GLOBALE
# ================================================================
offset = offset_sum / count_total
print(f"Mean centroid offset: {offset}")

# ================================================================
# 4️⃣ CENTRATURA E SALVATAGGIO DEFINITIVO
# ================================================================
temp_files = sorted([f for f in os.listdir(READY_DIR) if f.endswith("_temp.npz")])

for f in tqdm(temp_files, desc="Centering & exporting"):
    data = np.load(os.path.join(READY_DIR, f))
    verts = data["verts"] - offset
    faces = data["faces"]
    data.close()

    out_name = f.replace("_temp.npz", "_GTready.obj")
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(os.path.join(READY_DIR, out_name))
    os.remove(os.path.join(READY_DIR, f))

    del verts, faces, mesh
    gc.collect()
    try:
        libc.malloc_trim(0)
    except Exception:
        pass

print_memory("(after final export)")
print("✅ All meshes exported")

# ================================================================
# 5️⃣ VERIFICA GEOMETRICA (solo subset)
# ================================================================
centroids, scales, pca_axes = [], [], []
ready_files = sorted([f for f in os.listdir(READY_DIR) if f.endswith(".obj")])

for path in tqdm(ready_files[:500], desc="Checking geometry (subset)"):
    m = trimesh.load(os.path.join(READY_DIR, path), process=False)
    V = m.vertices
    c = V.mean(axis=0)
    centroids.append(c)
    bbox_min, bbox_max = V.min(axis=0), V.max(axis=0)
    scales.append(np.linalg.norm(bbox_max - bbox_min))
    pca = PCA(n_components=3)
    pca.fit(V - c)
    pca_axes.append(pca.components_)

    del m, V, c, bbox_min, bbox_max, pca
    gc.collect()
    try:
        libc.malloc_trim(0)
    except Exception:
        pass

centroids, scales, pca_axes = np.stack(centroids), np.array(scales), np.stack(pca_axes)
ref_c, ref_s, ref_a = centroids[0], scales[0], pca_axes[0]

def angle_between(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1)))

orient_diff = [angle_between(ref_a[0], pca_axes[i][0]) for i in range(len(pca_axes))]

df = pd.DataFrame({
    "file": ready_files[:len(centroids)],
    "centroid_shift": np.linalg.norm(centroids - ref_c, axis=1),
    "scale_ratio": scales / ref_s,
    "orientation_diff_deg": orient_diff
})
df.to_csv(os.path.join(RESULTS_DIR, "GT_ready_alignment_summary.csv"), index=False)

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.hist(np.linalg.norm(centroids - ref_c, axis=1), bins=30, color='skyblue')
plt.title("Translation check")

plt.subplot(1, 3, 2)
plt.hist(scales / ref_s, bins=30, color='orange')
plt.title("Scale check")

plt.subplot(1, 3, 3)
plt
