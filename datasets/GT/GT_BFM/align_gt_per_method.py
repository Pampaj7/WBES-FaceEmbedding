import numpy as np
from pathlib import Path
from scipy.spatial import procrustes

def load_mesh(path):
    return np.loadtxt(path)

def save_mesh(mesh, path):
    np.savetxt(path, mesh, fmt="%.6f")

def compute_rigid_alignment(gt, recon):
    _, aligned_gt, _ = procrustes(recon, gt)
    return aligned_gt

def estimate_rigid_transform(A, B):
    A_mean, B_mean = A.mean(0), B.mean(0)
    A_centered, B_centered = A - A_mean, B - B_mean
    H = A_centered.T @ B_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    s = np.trace(B_centered.T @ A_centered @ R) / np.trace(A_centered.T @ A_centered)
    t = B_mean - s * A_mean @ R
    return R, s, t

def apply_transform(mesh, R, s, t):
    return (mesh @ R) * s + t

# === CONFIG ===
base_dir = Path("/home/pampalonil/data/")
gt_dir = base_dir / "GT" / "GT_BFM"
method_dirs = {
    "3DDFAv2": base_dir / "3DDFAV2_23470_neutral",
    "3DDFAv3": base_dir / "3DDFAV3_neutral",
    "Deep3DFace": base_dir / "Deep3DFace_23470_neutral",
    "SynergyNet": base_dir / "SynergyNet_neutral",
    "INORig": base_dir / "INORig_23470_neutral",
    "3DI": base_dir / "3DI_neutral",
}

# === LOOP PER METODO ===
for method_name, method_path in method_dirs.items():
    print(f"\n🔧 Aligning GTs for method: {method_name}")
    out_dir = gt_dir / f"GT_BFM_aligned_{method_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cerca una mesh ricostruita per soggetto noto
    sid = "id0001"
    recon_candidates = sorted(method_path.glob(f"{sid}*.txt"))
    if not recon_candidates:
        print(f"❌ No recon file for {sid} in {method_path}")
        continue

    recon = load_mesh(recon_candidates[0])
    gt_path = gt_dir / f"{sid}.id.txt"
    if not gt_path.exists():
        print(f"❌ GT file missing for {sid}")
        continue
    gt = load_mesh(gt_path)

    # Allineamento e trasformazione rigida
    aligned_gt = compute_rigid_alignment(gt, recon)
    R, s, t = estimate_rigid_transform(gt, aligned_gt)

    # Applica a tutte le GT
    for gt_file in sorted(gt_dir.glob("*.id.txt")):
        mesh = load_mesh(gt_file)
        aligned = apply_transform(mesh, R, s, t)
        save_mesh(aligned, out_dir / gt_file.name)

    print(f"✅ Saved: {out_dir}")

print("\n✅ All GTs aligned per-method.")
