import os
import sys
import torch
import igl
import pickle
from torch.utils.data import Dataset

# === Add DiffusionNet path ===
if "/equilibrium/lpampaloni/diffusion-net/src" not in sys.path:
    sys.path.append("/equilibrium/lpampaloni/diffusion-net/src")


class GTReadyDataset(Dataset):
    """
    Dataset per GT_ready: carica mesh e operatori DiffusionNet precomputati (.pkl).
    Gestisce automaticamente file corrotti, normalizza la mesh e gli operatori
    per stabilità numerica, e mantiene le matrici sparse (L, gradX, gradY) su CPU.
    """
    
    def __init__(self, data_dir, ops_dir=None):
        self.data_dir = data_dir
        self.ops_dir = ops_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".obj")])


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        mesh_path = os.path.join(self.data_dir, fname)

        # === Path operatori ===
        if self.ops_dir:
            ops_path = os.path.join(self.ops_dir, fname.replace(".obj", "_ops.pkl"))
        else:
            ops_path = mesh_path.replace(".obj", "_ops.pkl")

        if not os.path.exists(ops_path):
            print(f"[WARN] Missing operator file: {ops_path}")
            return None

        # === Carica mesh ===
        try:
            V, F = igl.read_triangle_mesh(mesh_path)
        except Exception as e:
            print(f"[WARN] Failed to read mesh {mesh_path}: {e}")
            return None

        if V.ndim == 1:
            V = V.reshape(-1, 3)
        elif V.shape[0] == 3 and V.shape[1] != 3:
            V = V.T

        # === Normalizza mesh (centra e scala in [-1, 1]) ===
        V = V - V.mean(axis=0, keepdims=True)
        scale = abs(V).max()
        if scale > 0:
            V = V / scale

        # === Carica operatori DiffusionNet ===
        try:
            with open(ops_path, "rb") as fp:
                ops_data = pickle.load(fp)
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"[WARN] Corrupted operator file skipped: {ops_path} ({e})")
            return None
        except Exception as e:
            print(f"[WARN] Failed to load {ops_path}: {e}")
            return None

        required_keys = ["mass", "L", "evals", "evecs"]
        if not all(k in ops_data for k in required_keys):
            print(f"[WARN] Missing keys in operator file: {ops_path}")
            return None

        mass = ops_data["mass"]
        L = ops_data["L"]
        evals = ops_data["evals"]
        evecs = ops_data["evecs"]
        gradX = ops_data.get("gradX", None)
        gradY = ops_data.get("gradY", None)

        # === Conversione sicura a tensori ===
        def to_tensor_safe(x, dtype=torch.float32):
            if x is None:
                return None
            if hasattr(x, "is_sparse") and x.is_sparse:
                return x.coalesce().to(dtype)
            return torch.as_tensor(x, dtype=dtype)

        try:
            mass = to_tensor_safe(mass).flatten()
            if mass.dim() == 2:
                mass = torch.diagonal(mass)
            evals = to_tensor_safe(evals)
            evecs = to_tensor_safe(evecs)
            L = to_tensor_safe(L)
            gradX = to_tensor_safe(gradX)
            gradY = to_tensor_safe(gradY)
        except Exception as e:
            print(f"[WARN] Tensor conversion failed for {fname}: {e}")
            return None

        # === Normalizzazione numerica per stabilità ===
        mass = torch.nan_to_num(mass, nan=1e-6, posinf=1e6, neginf=1e-6)
        mass = mass / (mass.mean() + 1e-6)

        if L.is_sparse:
            L_vals = L.coalesce().values()
            L_max = torch.max(L_vals.abs()) + 1e-6
            L = torch.sparse_coo_tensor(L.indices(), L_vals / L_max, L.shape)
        else:
            L = L / (torch.max(L.abs()) + 1e-6)

        evals = torch.nan_to_num(evals, nan=0.0)
        evals = evals / (evals.max() + 1e-9)

        evecs = torch.nan_to_num(evecs, nan=0.0)
        evecs = evecs / (torch.max(evecs.abs()) + 1e-9)

        # === Controllo NaN/Inf finale ===
        tensors_to_check = {
            "V": torch.tensor(V),
            "mass": mass,
            "evals": evals,
            "evecs": evecs,
        }
        for name, t in tensors_to_check.items():
            if t is not None and (torch.isnan(t).any() or torch.isinf(t).any()):
                print(f"[WARN] NaN/Inf in {name} for {fname}, skipping.")
                return None

        # → GPU solo dati compatti
        V_torch = torch.tensor(V, dtype=torch.float32).contiguous()  # (N, 3)
        F_torch = torch.tensor(F, dtype=torch.long)

        return {
            "verts": V_torch,
            "faces": F_torch,
            "mass": mass,
            "L": L,
            "evals": evals,
            "evecs": evecs,
            "gradX": gradX,
            "gradY": gradY,
            "name": fname,
        }
