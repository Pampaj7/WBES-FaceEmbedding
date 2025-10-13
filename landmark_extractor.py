import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# === CONFIG ===
data_root = Path(".")
bfm_json_path = "utils/BFM-p23470.json"
flame_json_path = "utils/FLAME-face.json"
faceverse_npy_path = "/home/pampalonil/data/utils/faceverse_lmk_indices_51_cropped.npy"  # dove hai salvato i 51 indici

# === METODI CLASSIFICATI PER TOPOLOGIA ===
bfm_dirs = {
    "3DDFAV2_23470_neutral",
    "3DDFAV3_neutral",
    "3DI_neutral",
    "Deep3DFace_23470_neutral",
    "INORig_23470_neutral",
    "SynergyNet_neutral"
}
flame_dirs = {
    "Smirk_cropped_neutral",
}
faceverse_dirs = {
    "Faceverse_cropped_neutral",
}

# === LOAD INDICES ===
with open(bfm_json_path) as f:
    bfm_lmk_indices = json.load(f)["lmk_indices"][:51]

with open(flame_json_path) as f:
    flame_data = json.load(f)
    if "lmk_indices" in flame_data:
        flame_lmk_indices = flame_data["lmk_indices"][:51]
    elif "landmark_ids" in flame_data:
        flame_lmk_indices = flame_data["landmark_ids"][:51]
    else:
        raise KeyError("❌ No landmark index key found in FLAME JSON.")

faceverse_lmk_indices = np.load(faceverse_npy_path).tolist()

# === PROCESS EACH DIRECTORY ===
all_methods = bfm_dirs | flame_dirs | faceverse_dirs

for method_name in sorted(all_methods):
    method_dir = data_root / method_name
    if not method_dir.exists():
        print(f"⚠️ Skipping missing: {method_dir}")
        continue

    if method_name in bfm_dirs:
        lmk_indices = bfm_lmk_indices
    elif method_name in flame_dirs:
        lmk_indices = flame_lmk_indices
    elif method_name in faceverse_dirs:
        lmk_indices = faceverse_lmk_indices
    else:
        raise ValueError(f"Metodo {method_name} non classificato.")

    for txt_file in tqdm(sorted(method_dir.glob("*.txt")), desc=f"Processing {method_name}"):
        try:
            mesh = np.loadtxt(txt_file)

            if mesh.shape[0] <= max(lmk_indices):
                print(f"❌ {txt_file} too short")
                continue

            lmk_mesh = mesh[lmk_indices]
            out_path = txt_file.with_name(txt_file.stem + "_lmk.npy")
            np.save(out_path, lmk_mesh)

        except Exception as e:
            print(f"❌ Errore su {txt_file}: {e}")
