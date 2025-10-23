import os
import sys
import torch
import igl
import pickle
import multiprocessing as mp
from tqdm import tqdm

# === Add DiffusionNet path ===
if "/equilibrium/lpampaloni/diffusion-net/src" not in sys.path:
    sys.path.append("/equilibrium/lpampaloni/diffusion-net/src")

from diffusion_net.geometry import compute_operators

# === CONFIG ===
DATA_DIR = "../../../datasets/GT_ready/"
OUT_DIR = os.path.join(DATA_DIR, "operators")
K_EIG = 128
os.makedirs(OUT_DIR, exist_ok=True)

# === FUNZIONE SINGOLA ===


def process_mesh(f):
    mesh_path = os.path.join(DATA_DIR, f)
    out_path = os.path.join(OUT_DIR, f.replace(".obj", "_ops.pkl"))

    if os.path.exists(out_path):
        return "[skip]", f

    try:
        V, F = igl.read_triangle_mesh(mesh_path)
        if V.size == 0 or F.size == 0:
            return "[error]", f

        ops = compute_operators(
            torch.tensor(V, dtype=torch.float32),
            torch.tensor(F, dtype=torch.long),
            k_eig=K_EIG
        )

        data = {
            "mass": ops[1],
            "L": ops[2],
            "evals": ops[3],
            "evecs": ops[4],
            "gradX": ops[5],
            "gradY": ops[6],
        }

        with open(out_path, "wb") as fp:
            pickle.dump(data, fp)

        return "[ok]", f

    except Exception as e:
        return "[fail]", f"{f}: {str(e)}"

if __name__ == "__main__":
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".obj")])
    
    # ✅ Limita ai primi 1000 file
    files = files[:5000]

    n_cores = max(1, mp.cpu_count() - 1)
    print(f"🚀 Using {n_cores} CPU cores | k_eig={K_EIG}")
    print(f"📂 Input: {len(files)} meshes in {DATA_DIR}")

    ok = skip = fail = 0
    results = []

    with mp.Pool(n_cores) as pool:
        with tqdm(total=len(files), dynamic_ncols=True, desc="Computing operators") as pbar:
            for status, msg in pool.imap_unordered(process_mesh, files):
                results.append(f"{status} {msg}")
                if status == "[ok]":
                    ok += 1
                elif status == "[skip]":
                    skip += 1
                else:
                    fail += 1
                pbar.set_postfix(ok=ok, skip=skip, fail=fail)
                pbar.update(1)

    # salva log riassuntivo
    log_path = os.path.join(OUT_DIR, "precompute_log.txt")
    with open(log_path, "w") as fp:
        fp.write("\n".join(results))

    print("\n✅ Done! Operators saved in:", OUT_DIR)
    print(f"🧾 Log saved to: {log_path}")
    print(f"📊 Summary → OK: {ok} | Skipped: {skip} | Failed: {fail}")
