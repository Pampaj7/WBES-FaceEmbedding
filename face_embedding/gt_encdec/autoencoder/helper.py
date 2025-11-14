import os
import numpy as np
from datetime import datetime
import scipy.stats as st
import zipfile

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch.nn.functional as F

from torch.cuda.amp import autocast, GradScaler

from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset
from diffusion_autoencoder import DiffusionAutoencoder
from geometric_loss import GeometricLoss
from latent_loss import varcov_loss, smooth_loss, stress_loss_with_scale, hard_negative_mining_loss, distortion_regularizer, \
    percentile_matching_loss, multiscale_distance_loss, curriculum_distance_mask

# ============================================================
# Utilities (clean)
# ============================================================
def collate_skip(batch):
    return [s for s in batch if s is not None]


def sample_distance_submatrix(D_full: np.ndarray, rowcol_idx: np.ndarray) -> torch.Tensor:
    return torch.tensor(D_full[np.ix_(rowcol_idx, rowcol_idx)], dtype=torch.float32)


def build_name_index_map(names_from_npz):
    mapping = {}
    for i, nm in enumerate(names_from_npz):
        if isinstance(nm, bytes):
            try:
                nm = nm.decode('utf-8')
            except Exception:
                nm = str(nm)
        nm = str(nm)
        base = nm[:-4] if nm.endswith(".npz") else nm
        mapping[base] = i
    return mapping


# ============================================================
# STRICT LATENT CHECK (CORRETTA, DETERMINISTICA)
# ============================================================
def latent_identity_check(model, dataset, fixed_names, fixed_idx, D_ref, device):
    """
    Strict evaluation:
    - stesso subset fisso ogni volta
    - stesso ordine dei soggetti
    - pairwise L2 tra vettori latenti
    - Pearson e Spearman affidabili
    """
    if fixed_names is None or D_ref is None or len(fixed_idx) < 2:
        return None

    model.eval()
    Z_list = []

    with torch.no_grad():
        for nm in fixed_names:
            s = dataset.get_by_name(nm)
            if s is None:
                continue

            V = s["verts"].to(device)
            mass = s["mass"].to(device)
            L = s["L"].to(device)
            evals = s["evals"].to(device)
            evecs = s["evecs"].to(device)
            faces = s["faces"].to(device)
            gX = s["gradX"].to(device)
            gY = s["gradY"].to(device)

            out = model(V, mass, L, evals, evecs, faces, gX, gY)
            Zg = out[-1]
            if Zg.dim() == 1:
                Zg = Zg.unsqueeze(0)

            Z_list.append(Zg.cpu().numpy())

    if len(Z_list) < 2:
        return None

    Z = np.vstack(Z_list)  # shape (N, latent_dim)

    diff = Z[:, None, :] - Z[None, :, :]
    D_lat = np.sqrt((diff**2).sum(-1) + 1e-8)

    mask = np.triu_indices_from(D_lat, k=1)
    x = D_ref[mask]
    y = D_lat[mask]

    if x.std() < 1e-12 or y.std() < 1e-12:
        return None

    pear = st.pearsonr(x, y)[0]
    spear = st.spearmanr(x, y)[0]
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    slope = float(np.polyfit(x, y, 1)[0])

    print(f"   🔎 Latent Identity STRICT → "
          f"ρ_P={pear:.3f}, ρ_S={spear:.3f}, R²={r2:.3f}, slope={slope:.3f}")

    return {
        "pearson": pear,
        "spearman": spear,
        "r2": r2,
        "slope": slope,
    }


# ============================================================
# Patch per supportare get_by_name nel dataset
# ============================================================
def patch_dataset_with_get_by_name(dataset):
    mapping = {}
    for i, f in enumerate(dataset.files):
        base = f[:-4] if f.endswith(".npz") else f
        mapping[base] = i

    def get_by_name(name):
        base = name[:-4] if name.endswith(".npz") else name
        if base not in mapping:
            return None
        return dataset[mapping[base]]

    dataset.get_by_name = get_by_name
    return dataset

