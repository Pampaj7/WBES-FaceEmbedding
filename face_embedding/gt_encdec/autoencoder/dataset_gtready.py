import os
import sys
import torch
import igl
import pickle
from torch.utils.data import Dataset

# === Add DiffusionNet to sys.path ===
if "/home/pampaj/diffusion-net/src" not in sys.path:
    sys.path.append("/home/pampaj/diffusion-net/src")


class GTReadyDataset(Dataset):
    """
    Dataset per GT_ready: carica mesh e operatori DiffusionNet precomputati (.pkl).
    Se ops_dir è fornito, legge solo da quella directory (senza mai ricalcolare).
    Restituisce vertici, facce e operatori geometrici (mass, L, evals, evecs, gradX, gradY se presenti).
    """

    def __init__(self, data_dir, ops_dir=None, device="cuda"):
        self.files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith(".obj")])
        self.data_dir = data_dir
        self.ops_dir = ops_dir
        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        mesh_path = os.path.join(self.data_dir, fname)

        # Se esiste una cartella separata per gli operatori, leggi da lì
        if self.ops_dir:
            ops_path = os.path.join(
                self.ops_dir, fname.replace(".obj", "_ops.pkl"))
        else:
            ops_path = mesh_path.replace(".obj", "_ops.pkl")

        # === Carica mesh ===
        V, F = igl.read_triangle_mesh(mesh_path)
        if V.ndim == 1:
            V = V.reshape(-1, 3)
        elif V.shape[0] == 3 and V.shape[1] != 3:
            V = V.T

        V_torch = torch.tensor(V, dtype=torch.float32)
        F_torch = torch.tensor(F, dtype=torch.long)

        # === Carica operatori precomputati ===
        if not os.path.exists(ops_path):
            raise FileNotFoundError(f"Operator file not found: {ops_path}")

        with open(ops_path, "rb") as fp:
            ops_data = pickle.load(fp)

        mass = ops_data["mass"]
        L = ops_data["L"]
        evals = ops_data["evals"]
        evecs = ops_data["evecs"]
        gradX = ops_data.get("gradX", None)
        gradY = ops_data.get("gradY", None)

        # === Normalizza mass come vettore denso 1D ===
        if hasattr(mass, "is_sparse") and mass.is_sparse:
            mass = mass.to_dense()
        if mass.dim() == 2:
            mass = torch.diagonal(mass)
        mass = mass.flatten().float()

        # === Gestione sparse matrices ===
        if hasattr(L, "is_sparse") and L.is_sparse:
            L = L.coalesce()
        if gradX is not None and hasattr(gradX, "is_sparse") and gradX.is_sparse:
            gradX = gradX.coalesce()
        if gradY is not None and hasattr(gradY, "is_sparse") and gradY.is_sparse:
            gradY = gradY.coalesce()

        # === Sposta tutto sul device ===
        device = torch.device(
            self.device if torch.cuda.is_available() else "cpu")
        V_torch = V_torch.T.contiguous().float().to(device)  # (3, N)
        F_torch = F_torch.to(device)
        mass = mass.to(device)
        L = L.to(device)
        evals = evals.float().to(device)
        evecs = evecs.float().to(device)
        if gradX is not None:
            gradX = gradX.to(device)
        if gradY is not None:
            gradY = gradY.to(device)

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
