import os
import numpy as np
import json

# === CONFIG ===
subject_id = "id0033"
mesh_dir = f"/home/pampalonil/data/3DDFAV3_neutral"
gt_path = f"/home/pampalonil/data/GT/GT_BFM/{subject_id}.id.txt"
landmark_json = "/home/pampalonil/3Dfacebenchmark_presubmission/info/BFM-p23470.json"

# === Load landmark indices ===
with open(landmark_json) as f:
    landmark_indices = json.load(f)["lmk_indices"]

# === Load meshes ===
mesh_paths = sorted([os.path.join(mesh_dir, f)
                     for f in os.listdir(mesh_dir)
                     if f.startswith(subject_id) and f.endswith(".txt")])

meshes = [np.loadtxt(p) for p in mesh_paths]
meshes = [m for m in meshes if m.shape == meshes[0].shape]

print(f"✅ Loaded {len(meshes)} meshes for {subject_id}")

# === Load ground-truth and landmarks ===
gt = np.loadtxt(gt_path) / 1e6
gt_lmks = gt[landmark_indices]

# === Allineamento ===
def landmark_align(source, target, source_lmks, target_lmks):
    from scipy.linalg import svd

    muX, muY = source_lmks.mean(0), target_lmks.mean(0)
    Xc, Yc = source_lmks - muX, target_lmks - muY

    U, _, Vt = svd(Xc.T @ Yc)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:  # reflection fix
        Vt[-1] *= -1
        R = Vt.T @ U.T

    b = np.trace((Xc @ R) @ Yc.T) / np.trace(Xc @ Xc.T)
    t = muY - b * muX @ R

    return b * (source @ R) + t

def l2_error(A, B):
    return np.linalg.norm(A - B, axis=1).mean()

# === Calcola errori individuali (allineati) ===
errors = []
for mesh in meshes:
    aligned = landmark_align(mesh, gt, mesh[landmark_indices], gt_lmks)
    err = l2_error(aligned, gt)
    errors.append(err)

mean_error = np.mean(errors)

# === Mesh media allineata ===
mean_mesh = np.mean(np.stack(meshes), axis=0)
aligned_mean = landmark_align(mean_mesh, gt, mean_mesh[landmark_indices], gt_lmks)
mean_mesh_error = l2_error(aligned_mean, gt)

print(f"\n📊 Risultati (con landmark alignment) per {subject_id}:")
print(f"Errore medio sulle singole mesh (F=1): {mean_error:.6f}")
print(f"Errore sulla mesh media:              {mean_mesh_error:.6f}")
print(f"Delta (miglioramento):               {mean_error - mean_mesh_error:.6f}")
