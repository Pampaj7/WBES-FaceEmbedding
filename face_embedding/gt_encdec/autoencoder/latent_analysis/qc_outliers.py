#!/usr/bin/env python3
import os, torch, numpy as np, scipy.stats as st
from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset
from diffusion_autoencoder import DiffusionAutoencoder

# CONFIG (adatta i path se serve)
DATA_DIR = "../../../../datasets/GT_ready/npz_data/"
DIST_PATH = "../results_diffusionAE_latentaware/dist_matrices_fields/distance_matrices_fields.npz"
CKPT = "../results_diffusionAE_latentaware/latentaware_epoch50.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES = 200  # come in train (min(n_samples, len(dataset)))

# --- load model (same arch) ---
model = DiffusionAutoencoder(latent_dim=256, width=128, n_blocks=4).to(DEVICE)
sd = torch.load(CKPT, map_location=DEVICE)
model.load_state_dict(sd)
model.eval()

# --- dataset (lo stesso del train) ---
dataset = GTReadyDataset(DATA_DIR)

# --- load D_orig e map nome->idx (usato in train) ---
pack = np.load(DIST_PATH)
D_orig = pack["D_orig"]
names_pack = [str(x) for x in pack["names"]]
name_to_idx = { (nm[:-4] if nm.endswith(".npz") else nm): i
                for i,nm in enumerate(names_pack) }

# --- sample indices deterministically for reproducibility ---
rng = np.random.default_rng(42)
indices = rng.choice(len(dataset), min(N_SAMPLES, len(dataset)), replace=False)

# --- collect Zg and their matching name indices (exact same logic as train) ---
Z_list = []
idx_list = []
with torch.no_grad():
    for i in indices:
        s = dataset[i]
        V = s["verts"].to(DEVICE)
        mass, evals, evecs = s["mass"].to(DEVICE), s["evals"].to(DEVICE), s["evecs"].to(DEVICE)
        faces, L = s["faces"].to(DEVICE), s["L"].to(DEVICE)
        gX, gY = s["gradX"].to(DEVICE), s["gradY"].to(DEVICE)
        out = model(V, mass, L, evals, evecs, faces, gX, gY)
        # as in your training check: Zg = out[-1]
        if isinstance(out, (tuple, list)):
            Zg = out[-1]
        else:
            continue
        if Zg.dim() == 1:
            Zg = Zg.unsqueeze(0)
        Z_list.append(Zg.cpu().numpy())
        base = s["name"][:-4] if s["name"].endswith(".npz") else s["name"]
        if base in name_to_idx:
            idx_list.append(name_to_idx[base])

Z_all = np.vstack(Z_list)  # (M, D)
# --- pairwise euclidean distances on raw Zg (exactly like train latent_identity_check) ---
diff = Z_all[:, None, :] - Z_all[None, :, :]
D_lat = np.sqrt((diff ** 2).sum(-1))
# min-max normalize like in train
D_lat = (D_lat - D_lat.min()) / (D_lat.max() - D_lat.min() + 1e-12)

# ground truth submatrix
idx_arr = np.array(idx_list)
D_gt = D_orig[np.ix_(idx_arr, idx_arr)]
D_gt = (D_gt - D_gt.min()) / (D_gt.max() - D_gt.min() + 1e-12)

mask = np.triu_indices_from(D_gt, k=1)
x = D_gt[mask].ravel()
y = D_lat[mask].ravel()

pear = st.pearsonr(x, y)[0]
spear = st.spearmanr(x, y)[0]
r2 = np.corrcoef(x, y)[0,1]**2

print(f"Recomputed: Pearson={pear:.4f}, Spearman={spear:.4f}, R2={r2:.4f}")
