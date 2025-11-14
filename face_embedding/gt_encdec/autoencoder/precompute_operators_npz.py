import os
import sys
import torch
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

# === PATH alla libreria DiffusionNet ===
if "/equilibrium/lpampaloni/diffusion-net/src" not in sys.path:
    sys.path.append("/equilibrium/lpampaloni/diffusion-net/src")
from diffusion_net.geometry import compute_operators

# === CONFIG ===
INPUT_DIR = "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/GT_ready/npz_data_cropped"
OUTPUT_DIR = "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/GT_ready/npz_data_cropped_23470_with_ops"
K_EIG = 128
os.makedirs(OUTPUT_DIR, exist_ok=True)

def sparse_to_coo_dict(sparse_tensor, base_name):
    coalesced = sparse_tensor.coalesce()
    indices = coalesced.indices().detach().cpu().numpy()
    values = coalesced.values().detach().cpu().numpy()
    shape = coalesced.shape
    return {
        f"{base_name}_indices": indices,
        f"{base_name}_values": values,
        f"{base_name}_shape": np.array(shape)
    }

def process_npz(filename):
    in_path = os.path.join(INPUT_DIR, filename)
    out_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(out_path):
        return "[skip]", filename

    try:
        data = np.load(in_path)
        V_np, F_np = data["V"], data["F"]

        # → torch
        V = torch.tensor(V_np, dtype=torch.float32)
        F = torch.tensor(F_np, dtype=torch.long)

        ops = compute_operators(V, F, k_eig=K_EIG)

        # crea nuovo dizionario
        new_data = {
            "verts": V_np,
            "faces": F_np,
            "mass": ops[1].cpu().numpy(),
            "evals": ops[3].cpu().numpy(),
            "evecs": ops[4].cpu().numpy()
        }
        new_data.update(sparse_to_coo_dict(ops[2], "L"))
        new_data.update(sparse_to_coo_dict(ops[5], "gradX"))
        new_data.update(sparse_to_coo_dict(ops[6], "gradY"))

        np.savez_compressed(out_path, **new_data)
        return "[ok]", filename

    except Exception as e:
        return "[fail]", f"{filename}: {e}"

if __name__ == "__main__":
    files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".npz"))
    files = files[::-1]
    n_cores = 4  # puoi mettere mp.cpu_count() per usare tutti

    print(f"🚀 Uso {n_cores} core")
    print(f"📂 Input: {INPUT_DIR}")
    print(f"💾 Output: {OUTPUT_DIR}")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    ok = skip = fail = 0
    results = []

    with mp.Pool(n_cores) as pool:
        with tqdm(total=len(files), dynamic_ncols=True, desc="Computing operators") as pbar:
            for status, msg in pool.imap_unordered(process_npz, files):
                results.append(f"{status} {msg}")
                if status == "[ok]": ok += 1
                elif status == "[skip]": skip += 1
                else: fail += 1
                pbar.set_postfix(ok=ok, skip=skip, fail=fail)
                pbar.update(1)

    print(f"\n✅ Operatori salvati: {ok} | Saltati: {skip} | Falliti: {fail}")
