import os
import numpy as np
import trimesh.registration as reg
from tqdm import tqdm

DATA_DIR = "../../../datasets/GT/GT_BFM/"
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".txt")])
B = np.loadtxt(os.path.join(DATA_DIR, files[0]))  # reference


def rotation_angle_from_matrix(M):
    # M è 3x3
    tr = np.trace(M)
    ang = np.degrees(np.arccos(np.clip((tr - 1) / 2.0, -1.0, 1.0)))
    return ang


residuals = []
rots_deg = []
scales = []
trans_mm = []

for f in tqdm(files[1:10], desc="Measuring proper residual"):
    A = np.loadtxt(os.path.join(DATA_DIR, f))
    M, A_aligned, cost = reg.procrustes(A, B, scale=True)  # trimesh
    # NB: A_aligned è già s R A + t

    # 1) residuo medio (ciò che conta)
    resid = np.mean(np.linalg.norm(A_aligned - B, axis=1))
    residuals.append(resid)

    # 2) decostruisco M in s,R,t per curiosità diagnostica
    #    M è 4x4: [ sR  t ; 0 0 0 1 ]
    sR = M[:3, :3]
    t = M[:3, 3]
    s = np.cbrt(np.linalg.det(sR))              # fattore di scala globale
    R = sR / s                                   # rotazione pura
    rot_deg = rotation_angle_from_matrix(R)
    rots_deg.append(rot_deg)
    scales.append(s)
    trans_mm.append(np.linalg.norm(t))

print(
    f"Residual mean (mm): {np.mean(residuals):.6f}  |  min {np.min(residuals):.6f}  max {np.max(residuals):.6f}")
print(f"Rotation (deg):     mean {np.mean(rots_deg):.4f}")
print(f"Scale factor s:     mean {np.mean(scales):.6f}")
print(f"Translation norm:   mean {np.mean(trans_mm):.2f} mm")
