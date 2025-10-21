import os
import torch
import igl
import pickle
from diffusion_net.geometry import compute_operators
import sys

# === Add DiffusionNet path ===
if "/home/pampaj/diffusion-net/src" not in sys.path:
    sys.path.append("/home/pampaj/diffusion-net/src")


def precompute_operators(data_dir, out_dir, k_eig=128):
    """
    Pre-calcola e salva su disco gli operatori DiffusionNet per ciascuna mesh OBJ.

    Args:
        data_dir (str): directory contenente le mesh .obj
        out_dir (str): directory dove salvare i file .pkl
        k_eig (int): numero di autovettori/autovalori da calcolare
    """
    os.makedirs(out_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".obj")])

    for f in files:
        mesh_path = os.path.join(data_dir, f)
        V, F = igl.read_triangle_mesh(mesh_path)

        # compute_operators restituisce (frames, mass, L, evals, evecs, gradX, gradY)
        ops = compute_operators(
            torch.tensor(V, dtype=torch.float32),
            torch.tensor(F, dtype=torch.long),
            k_eig=k_eig
        )

        # Estrai componenti nel formato standard
        data = {
            'frames': ops[0],
            'mass': ops[1],
            'L': ops[2],
            'evals': ops[3],
            'evecs': ops[4],
            'gradX': ops[5],
            'gradY': ops[6],
        }

        save_path = os.path.join(out_dir, f.replace(".obj", "_ops.pkl"))
        with open(save_path, "wb") as fp:
            pickle.dump(data, fp)

        print(f"[✓] Saved operators for {f} → {save_path}")
