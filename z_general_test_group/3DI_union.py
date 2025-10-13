import os
import shutil

# === CONFIG ===
base_path = "/data/face/synth_neutral/reconstructions"
folders = {
    "3DI_010_rec1": 1,
    "3DI_010_rec2": 2,
    "3DI_010_rec3": 3,
}
output_dir = "/home/pampalonil/data/3DI_neutral"
os.makedirs(output_dir, exist_ok=True)

for folder, idx in folders.items():
    path = os.path.join(base_path, folder)
    for fname in os.listdir(path):
        if not fname.startswith("id") or not fname.endswith(".txt"):
            continue
        subj = fname.replace(".txt", "")
        newname = f"{subj}_{idx}.txt"
        src = os.path.join(path, fname)
        dst = os.path.join(output_dir, newname)
        shutil.copyfile(src, dst)

print(f"✅ Merge completed: all files copied to {output_dir}")
