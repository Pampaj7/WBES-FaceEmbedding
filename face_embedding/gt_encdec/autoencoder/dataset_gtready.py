import os
import sys
import torch
import igl
import pickle
from torch.utils.data import Dataset

# === Add DiffusionNet path ===
if "/equilibrium/lpampaloni/diffusion-net/src" not in sys.path:
    sys.path.append("/equilibrium/lpampaloni/diffusion-net/src")
# Aggiungi anche il path dell'utente corrente se diverso
if "/home/pampaj/diffusion-net/src" not in sys.path:
    sys.path.append("/home/pampaj/diffusion-net/src")


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

    def __getitem__(self, idx): # this method is critic, here can relay a bug
        fname = self.files[idx]
        mesh_path = os.path.join(self.data_dir, fname)

        # === Path operatori ===
        if self.ops_dir:
            ops_path = os.path.join(self.ops_dir, fname.replace(".obj", "_ops.pkl"))
        else:
            ops_path = mesh_path.replace(".obj", "_ops.pkl")

        if not os.path.exists(ops_path):
            # print(f"Warn: Skip {fname}, ops file not found")
            return None

        # === Carica mesh ===
        try:
            V, F = igl.read_triangle_mesh(mesh_path)
        except Exception as e:
            # print(f"Warn: Skip {fname}, mesh read error: {e}")
            return None

        if V.ndim == 1:
            V = V.reshape(-1, 3)
        elif V.shape[0] == 3 and V.shape[1] != 3:
            V = V.T
            
        if V.size == 0 or F.size == 0:
            return None

        # === Normalizza mesh (centra e scala in [-1, 1]) ===
        V_mean = V.mean(axis=0, keepdims=True)
        V = V - V_mean
        scale = abs(V).max()
        if scale > 1e-6: # Evita divisione per zero se la mesh è degenere
            V = V / scale
        else:
            V = V * 0.0 # Imposta a zero se la scala è zero

        # === Carica operatori DiffusionNet ===
        try:
            with open(ops_path, "rb") as fp:
                ops_data = pickle.load(fp)
        except (EOFError, pickle.UnpicklingError) as e:
            # print(f"Warn: Skip {fname}, ops load error: {e}")
            return None
        except Exception as e:
            # print(f"Warn: Skip {fname}, generic load error: {e}")
            return None

        required_keys = ["mass", "L", "evals", "evecs", "gradX", "gradY"]
        if not all(k in ops_data for k in required_keys):
            # print(f"Warn: Skip {fname}, missing keys in ops file")
            return None

        mass = ops_data["mass"]
        L = ops_data["L"]
        evals = ops_data["evals"]
        evecs = ops_data["evecs"]
        gradX = ops_data.get("gradX", None)
        gradY = ops_data.get("gradY", None)
        
        if gradX is None or gradY is None:
            # print(f"Warn: Skip {fname}, missing gradX/gradY")
            return None

        # === Conversione sicura a tensori ===
        def to_tensor_safe(x, dtype=torch.float32):
            if x is None:
                return None
            if hasattr(x, "is_sparse") and x.is_sparse:
                # Assicura che i tensori sparsi siano coalesced
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
            # print(f"Warn: Skip {fname}, tensor conversion error: {e}")
            return None

        # ===================================================================
        # === numerical normalization ===
        # ===================================================================
        
        # 1. Pulisci NaNs e Inf dagli input
        mass = torch.nan_to_num(mass, nan=1e-6, posinf=1e6, neginf=1e-6)
        evals = torch.nan_to_num(evals, nan=0.0)
        evecs = torch.nan_to_num(evecs, nan=0.0)
        
        # 2. Trova il fattore di scala spettrale (autovalore massimo)
        lambda_max = evals.max() + 1e-9
        
        # 3. Scala SIA L CHE evals dello STESSO fattore
        evals = evals / lambda_max

        # 🌟 === CORREZIONE === 🌟
        # Funzione helper modificata per restituire SEMPRE un tensore coalesced
        def scale_sparse_tensor(t, scale):
            if not t.is_sparse:
                t = torch.nan_to_num(t, nan=0.0)
                return t / scale
            
            t_coalesced = t.coalesce() # Coalesce una volta per leggere in sicurezza
            vals = t_coalesced.values()
            vals = torch.nan_to_num(vals, nan=0.0)
            
            # Crea il nuovo tensore
            new_t = torch.sparse_coo_tensor(t_coalesced.indices(), vals / scale, t_coalesced.shape)
            
            # Ritorna la versione coalesced del NUOVO tensore
            return new_t.coalesce()
        # 🌟 =======================

        L = scale_sparse_tensor(L, lambda_max)
        
        # 4. Scala gradX e gradY di sqrt(lambda_max)
        lambda_max_sqrt = torch.sqrt(lambda_max)
        gradX = scale_sparse_tensor(gradX, lambda_max_sqrt)
        gradY = scale_sparse_tensor(gradY, lambda_max_sqrt)

        # ===================================================================

        # === Controllo NaN/Inf finale ===
        # Questo blocco ora funzionerà perché L, gradX, gradY sono coalesced
        tensors_to_check = {
            "V": torch.tensor(V), "mass": mass, "evals": evals, "evecs": evecs
        }
        for name, t in tensors_to_check.items():
            if t is not None and (torch.isnan(t).any() or torch.isinf(t).any()):
                # print(f"Warn: Skip {fname}, NaN/Inf detected in {name}")
                return None
        
        # Controlla anche i tensori sparsi
        if (torch.isnan(L.values()).any() or torch.isinf(L.values()).any() or
            torch.isnan(gradX.values()).any() or torch.isinf(gradX.values()).any() or
            torch.isnan(gradY.values()).any() or torch.isinf(gradY.values()).any()):
            # print(f"Warn: Skip {fname}, NaN/Inf detected in sparse operators")
            return None


        V_torch = torch.tensor(V, dtype=torch.float32).contiguous()
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