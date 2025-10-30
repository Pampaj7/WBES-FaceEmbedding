# dataset_gtready_fixed.py
import os
import sys
import torch
import igl
import pickle
import numpy as np
from torch.utils.data import Dataset

# === Add DiffusionNet path (kept your multiple possible paths) ===
for p in (
    "/equilibrium/lpampaloni/diffusion-net/src",
    "/home/pampaj/diffusion-net/src",
    "/seidenas/users/lpampaloni/diffusion-net/src",
):
    if p not in sys.path:
        sys.path.append(p)

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def safe_print(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)

def to_tensor_safe(x, dtype=torch.float32):
    """
    Robust conversion to torch tensor:
    - If x is already a torch Tensor -> cast dtype
    - If x is a scipy / numpy COO-style tuple/list (indices, values, shape) -> build sparse tensor
    - If x is a numpy array -> torch.from_numpy
    - Handles sparse torch tensors by coalescing and casting
    Returns None on unrecoverable error.
    """
    if x is None:
        return None
    # Already a torch tensor
    if isinstance(x, torch.Tensor):
        try:
            return x.to(dtype)
        except Exception:
            try:
                return x.float()
            except Exception:
                return None

    # Detect (indices, values, shape) pattern from saved PKL/NPZ
    if isinstance(x, (tuple, list)) and len(x) == 3:
        inds, vals, shape = x
        try:
            # numpy -> torch
            inds = np.asarray(inds)
            vals = np.asarray(vals)
            shape = tuple(int(s) for s in shape)
            # indices shape normalization: expect (ndim, nnz)
            if inds.ndim == 2 and inds.shape[0] != shape[0] and inds.shape[1] == 2:
                # maybe transposed: turn into (2, nnz)
                inds = inds.T
            inds_t = torch.from_numpy(inds).long()
            vals_t = torch.from_numpy(vals).to(dtype)
            return torch.sparse_coo_tensor(inds_t, vals_t, torch.Size(shape)).coalesce()
        except Exception:
            return None

    # numpy array dense
    if isinstance(x, np.ndarray):
        try:
            return torch.from_numpy(x).to(dtype)
        except Exception:
            try:
                return torch.tensor(x, dtype=dtype)
            except Exception:
                return None

    # try a final fallback: attempt to build tensor from object
    try:
        return torch.as_tensor(x, dtype=dtype)
    except Exception:
        return None

def safe_isfinite_tensor(t):
    """Return False if t is None or contains non-finite entries. Works for sparse/dense."""
    if t is None:
        return True
    try:
        if isinstance(t, torch.Tensor):
            if t.is_sparse:
                vals = t.coalesce().values()
                return torch.isfinite(vals).all().item()
            else:
                return torch.isfinite(t).all().item()
        else:
            return True
    except Exception:
        return False

def scale_sparse_tensor(t, scale):
    """Scale dense or sparse tensor values by scalar scale (safe)."""
    if t is None:
        return None
    try:
        scale_val = scale.item() if isinstance(scale, torch.Tensor) and scale.numel() == 1 else float(scale)
    except Exception:
        scale_val = float(scale)
    scale_val = max(scale_val, 1e-9)
    if isinstance(t, torch.Tensor) and not t.is_sparse:
        return torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6) / scale_val
    try:
        t_c = t.coalesce()
        vals = torch.nan_to_num(t_c.values(), nan=0.0, posinf=1e6, neginf=-1e6)
        return torch.sparse_coo_tensor(t_c.indices(), vals / scale_val, t_c.shape).coalesce()
    except Exception:
        return None

# -------------------------------------------------------------------------
# CLASS 1: GTReadyDataset (Legacy - Loads from .obj and .pkl)
# -------------------------------------------------------------------------
class GTReadyDataset(Dataset):
    """
    Legacy Dataset for GT_ready: loads mesh (.obj) and precomputed DiffusionNet
    operators (.pkl). Handles numerical and geometric normalization.

    Params:
      data_dir: directory with .obj
      ops_dir: directory with corresponding _ops.pkl files (optional)
      verbose: bool for logging
    """
    def __init__(self, data_dir, ops_dir=None, verbose=False):
        self.data_dir = data_dir
        self.ops_dir = ops_dir
        self.verbose = verbose
        try:
            self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".obj")])
            if not self.files:
                safe_print(self.verbose, f"[WARN] No .obj files found in {data_dir}")
        except Exception as e:
            safe_print(self.verbose, f"[ERROR] Failed to list files in {data_dir}: {e}")
            self.files = []

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.files):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.files)} items")

        fname = self.files[idx]
        npz_path = os.path.join(self.data_dir, fname)

        try:
            # --- 1️⃣ Caricamento sicuro dello zip ---
            try:
                data = np.load(npz_path, allow_pickle=False)
            except Exception as e:
                print(f"[WARN] File NPZ corrotto o non leggibile: {fname} → {e}")
                return None

            # --- 2️⃣ Lettura campi obbligatori ---
            V_np = data.get("verts", None)
            F_np = data.get("faces", None)
            if V_np is None or F_np is None:
                print(f"[WARN] Mancano verts/faces in {fname}, skip.")
                return None

            mass_np = data.get("mass", None)
            evals_np = data.get("evals", None)
            evecs_np = data.get("evecs", None)
            if mass_np is None or evals_np is None or evecs_np is None:
                print(f"[WARN] Mancano mass/evals/evecs in {fname}, skip.")
                return None

            # --- 3️⃣ Conversione e normalizzazione geometrica ---
            V = V_np.astype(np.float64)
            V = V - V.mean(axis=0, keepdims=True)
            scale = np.max(np.abs(V))
            if scale > 1e-6:
                V = V / scale
            else:
                V = V * 0.0

            V_torch = torch.tensor(V, dtype=torch.float32).contiguous()
            F_torch = torch.tensor(F_np, dtype=torch.long)

            mass = torch.from_numpy(np.asarray(mass_np)).float().flatten()
            if mass.dim() == 2:
                mass = torch.diagonal(mass)
            evals = torch.from_numpy(np.asarray(evals_np)).float()
            evecs = torch.from_numpy(np.asarray(evecs_np)).float()

            # --- 4️⃣ Costruzione operatori sparsi ---
            L = self.coo_dict_to_sparse_tensor(data, "L")
            gradX = self.coo_dict_to_sparse_tensor(data, "gradX")
            gradY = self.coo_dict_to_sparse_tensor(data, "gradY")
            if L is None or gradX is None or gradY is None:
                print(f"[WARN] Operatori sparsi mancanti o invalidi in {fname}, skip.")
                return None

        except Exception as e:
            print(f"[WARN] Errore inatteso in {fname}: {e}")
            return None
        finally:
            try:
                if hasattr(data, "close"):
                    data.close()
            except Exception:
                pass

        # --- 5️⃣ Normalizzazione numerica ---
        try:
            mass = torch.nan_to_num(mass, nan=1e-6, posinf=1e6, neginf=-1e6)
            evals = torch.nan_to_num(evals, nan=0.0, posinf=1e6, neginf=-1e6)
            evecs = torch.nan_to_num(evecs, nan=0.0, posinf=1e6, neginf=-1e6)
            evals = torch.clamp(evals, min=0.0)
            lambda_max = float(evals.max().item()) + 1e-9

            evals = evals / lambda_max
            L = scale_sparse_tensor(L, lambda_max)
            gradX = scale_sparse_tensor(gradX, np.sqrt(lambda_max))
            gradY = scale_sparse_tensor(gradY, np.sqrt(lambda_max))
        except Exception as e:
            print(f"[WARN] Fallita normalizzazione numerica in {fname}: {e}")
            return None

        # --- 6️⃣ Validazione finale ---
        tensors_to_check = {"V": V_torch, "mass": mass, "evals": evals, "evecs": evecs}
        for k, t in tensors_to_check.items():
            if t is not None and not safe_isfinite_tensor(t):
                print(f"[WARN] Non-finite values in {k} for {fname}, skip.")
                return None
        if not all([
            safe_isfinite_tensor(L),
            safe_isfinite_tensor(gradX),
            safe_isfinite_tensor(gradY),
        ]):
            print(f"[WARN] Non-finite sparse ops in {fname}, skip.")
            return None

        # --- ✅ Tutto ok ---
        return {
            "verts": V_torch, "faces": F_torch, "mass": mass, "L": L,
            "evals": evals, "evecs": evecs, "gradX": gradX, "gradY": gradY,
            "name": fname.replace(".npz", ""),
        }


# -------------------------------------------------------------------------
# CLASS 2: GTReadyDatasetNPZ (New - Loads from .npz with robustness fixes)
# -------------------------------------------------------------------------
class GTReadyDatasetNPZ(Dataset):
    """
    Loads precomputed .npz files with keys like:
      - verts, faces, mass, evals, evecs
      - L_indices, L_values, L_shape, gradX_indices, ...
    """
    def __init__(self, npz_data_dir, verbose=False):
        self.data_dir = npz_data_dir
        self.verbose = verbose
        try:
            self.files = sorted([f for f in os.listdir(self.data_dir) if f.endswith(".npz")])
            if not self.files:
                raise FileNotFoundError(f"No .npz files found in {self.data_dir}")
        except Exception as e:
            safe_print(self.verbose, f"[ERROR] Error listing NPZ directory {self.data_dir}: {e}")
            self.files = []

    def __len__(self):
        return len(self.files)

    def coo_dict_to_sparse_tensor(self, data, base_name, dtype=torch.float32):
        indices_key = f"{base_name}_indices"
        values_key = f"{base_name}_values"
        shape_key = f"{base_name}_shape"
        if not all(k in data for k in (indices_key, values_key, shape_key)):
            return None
        try:
            inds = np.asarray(data[indices_key])
            vals = np.asarray(data[values_key])
            shape = tuple(int(s) for s in data[shape_key])
            # Normalize indices -> expected (ndim, nnz)
            if inds.ndim == 2 and inds.shape[0] != len(shape) and inds.shape[1] == len(shape):
                inds = inds.T
            inds_t = torch.from_numpy(inds).long()
            vals_t = torch.from_numpy(vals).to(dtype)
            return torch.sparse_coo_tensor(inds_t, vals_t, torch.Size(shape)).coalesce()
        except Exception:
            return None

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.files):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.files)} items")
        fname = self.files[idx]
        npz_path = os.path.join(self.data_dir, fname)

        try:
            data = np.load(npz_path, allow_pickle=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load npz {npz_path}: {e}")

        try:
            V_np = data.get("verts", None)
            F_np = data.get("faces", None)
            if V_np is None or F_np is None:
                raise RuntimeError(f"Missing verts/faces in {npz_path}")

            # geometric normalization
            V = V_np.astype(np.float64)
            V = V - V.mean(axis=0, keepdims=True)
            scale = np.max(np.abs(V))
            if scale > 1e-6:
                V = V / scale
            else:
                V = V * 0.0
            V_torch = torch.tensor(V, dtype=torch.float32).contiguous()
            F_torch = torch.tensor(F_np, dtype=torch.long)

            mass_np = data.get("mass", None)
            evals_np = data.get("evals", None)
            evecs_np = data.get("evecs", None)
            if mass_np is None or evals_np is None or evecs_np is None:
                raise RuntimeError(f"Missing mass/evals/evecs in {npz_path}")

            mass = torch.from_numpy(np.asarray(mass_np)).float().flatten()
            if mass.dim() == 2:
                mass = torch.diagonal(mass)

            evals = torch.from_numpy(np.asarray(evals_np)).float()
            evecs = torch.from_numpy(np.asarray(evecs_np)).float()

            L = self.coo_dict_to_sparse_tensor(data, "L")
            if L is None:
                raise RuntimeError(f"Missing or invalid L in {npz_path}")
            gradX = self.coo_dict_to_sparse_tensor(data, "gradX")
            if gradX is None:
                raise RuntimeError(f"Missing or invalid gradX in {npz_path}")
            gradY = self.coo_dict_to_sparse_tensor(data, "gradY")
            if gradY is None:
                raise RuntimeError(f"Missing or invalid gradY in {npz_path}")

        finally:
            # close if numpy memmap-like object needs it
            try:
                if hasattr(data, "close"):
                    data.close()
            except Exception:
                pass

        # Numerical normalization
        mass = torch.nan_to_num(mass, nan=1e-6, posinf=1e6, neginf=-1e6)
        evals = torch.nan_to_num(evals, nan=0.0, posinf=1e6, neginf=-1e6)
        evecs = torch.nan_to_num(evecs, nan=0.0, posinf=1e6, neginf=-1e6)
        evals = torch.clamp(evals, min=0.0)
        lambda_max = float(evals.max().item()) + 1e-9

        evals = evals / lambda_max

        L_scaled = scale_sparse_tensor(L, lambda_max)
        if L_scaled is None:
            raise RuntimeError(f"Failed scaling L for {fname}")
        L = L_scaled

        lambda_max_clamped = max(lambda_max, 1e-9)
        lambda_max_sqrt = np.sqrt(lambda_max_clamped)

        gradX_scaled = scale_sparse_tensor(gradX, lambda_max_sqrt)
        if gradX_scaled is None:
            raise RuntimeError(f"Failed scaling gradX for {fname}")
        gradX = gradX_scaled

        gradY_scaled = scale_sparse_tensor(gradY, lambda_max_sqrt)
        if gradY_scaled is None:
            raise RuntimeError(f"Failed scaling gradY for {fname}")
        gradY = gradY_scaled

        # final checks
        tensors_to_check = {"V": V_torch, "mass": mass, "evals": evals, "evecs": evecs}
        for k, t in tensors_to_check.items():
            if t is not None and not safe_isfinite_tensor(t):
                raise RuntimeError(f"Non-finite values detected in {k} for {fname}")

        if not safe_isfinite_tensor(L) or not safe_isfinite_tensor(gradX) or not safe_isfinite_tensor(gradY):
            raise RuntimeError(f"Non-finite values in sparse ops for {fname}")

        return {
            "verts": V_torch, "faces": F_torch, "mass": mass, "L": L,
            "evals": evals, "evecs": evecs, "gradX": gradX, "gradY": gradY,
            "name": fname.replace(".npz", ""),
        }
