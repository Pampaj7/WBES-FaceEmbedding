import os
import numpy as np
from tqdm import tqdm

# === CONFIG ===
input_root = "/data/face/synth_neutral/reconstructions/Deep3DFace"       # CAMBIA QUI
output_root = "../Deep3DFace_23470"  # CAMBIA QUI
os.makedirs(output_root, exist_ok=True)

# === MAPPA diretta da 53490 a 23470
ix23470 = np.loadtxt("/home/pampalonil/3Dfacebenchmark_presubmission/facebenchmark/utils/idxs/ix_23470_relative_to_53490.txt").astype(int)

def convert_53490_to_23470(mesh_53490: np.ndarray) -> np.ndarray:
    return mesh_53490[ix23470]

print(f"===> Scanning: {input_root}")

all_files = sorted([
    f for f in os.listdir(input_root)
    if f.endswith(".txt") and f.startswith("id")
])

for fname in tqdm(all_files, desc="Converting", ncols=100):
    in_path = os.path.join(input_root, fname)
    try:
        verts = np.loadtxt(in_path, dtype=np.float32)
    except Exception as e:
        print(f"[!] Error loading {fname}: {e}")
        continue

    if verts.shape[0] != 53490:
        print(f"[!] Skipping {fname}: {verts.shape[0]} verts, expected 53490")
        continue

    reduced = convert_53490_to_23470(verts)
    out_path = os.path.join(output_root, fname)  # <-- usa stesso nome file
    np.savetxt(out_path, reduced, fmt="%.6f")
    tqdm.write(f"[✓] Saved: {out_path}")

print("\n✅ Conversion complete — filenames preserved.")
