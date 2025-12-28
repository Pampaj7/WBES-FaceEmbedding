import os
import numpy as np
from tqdm import tqdm

# === CONFIG ===
base = "/data/face/synth_neutral/reconstructions"
methods = [
    "3DDFAv2",
    "Deep3DFace",
    "INORig",
    "3DI_010_rec1",
    "3DI_010_rec2",
    "3DI_010_rec3",
]

nan_files = []

for method in methods:
    folder = os.path.join(base, method)
    if not os.path.isdir(folder):
        print(f"[SKIP] {method} → cartella non trovata")
        continue

    print(f"\n[CHECK] {method}")
    files = sorted([
        f for f in os.listdir(folder)
        if f.startswith("id") and f.endswith(".txt")
    ])

    for fname in tqdm(files, desc=f"{method:>15}", ncols=100):
        fpath = os.path.join(folder, fname)
        try:
            arr = np.loadtxt(fpath)
            if np.isnan(arr).any():
                tqdm.write(f"[NAN] {method}/{fname}")
                nan_files.append((method, fname))
        except Exception as e:
            tqdm.write(f"[ERROR] {method}/{fname}: {e}")
            nan_files.append((method, fname))

# === Salva output
outpath = "files_with_nan.txt"
with open(outpath, "w") as f:
    for method, fname in nan_files:
        f.write(f"{method}/{fname}\n")

print(f"\n✅ Controllo completato. File con NaN salvati in: {outpath}")
