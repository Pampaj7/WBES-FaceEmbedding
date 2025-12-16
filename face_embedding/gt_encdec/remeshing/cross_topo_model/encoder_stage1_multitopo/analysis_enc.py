#!/usr/bin/env python3
import os
import re
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# -----------------------------
# paths (ADATTA SE SERVE)
# -----------------------------
BASE = "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/gt_encdec/autoencoder"
DATA_DIR = (
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/REMESH/"
    "npz_data_topo_500_withops"
)
DIST_PATH = (
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/"
    "gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/"
    "normalized_matrix_distances.npz"
)
CKPT = "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/gt_encdec/remeshing/cross_topo_model/encoder_stage1_multitopo/encoder_stage1_epoch35.pth"  # relativo o assoluto
OUT_DIR = "encoder_stage1_multitopo_eval"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# imports progetto
# -----------------------------
import sys
sys.path.append(BASE)

from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset
from diffusion_autoencoder import DiffusionEncoderOnly

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Varianti attese: idXXXX_..._(original|remesh|crop|noisy).npz
VARIANT_RE = re.compile(r"^(id\d+)_.*_(original|remesh|crop|noisy)\.npz$")


def build_subject_map(dataset):
    subj_to_idxs = {}
    for idx, fname in enumerate(dataset.files):
        m = VARIANT_RE.match(fname)
        subj = m.group(1) if m else fname.split("_")[0]
        subj_to_idxs.setdefault(subj, []).append(idx)
    return subj_to_idxs


def pearsonr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x*x).sum()) * np.sqrt((y*y).sum()) + 1e-12)
    return float((x*y).sum() / denom)


def spearmanr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    return pearsonr(rx, ry)


def fit_slope_r2(x, y):
    # y ≈ a*x + b
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = a*x + b
    ss_res = ((y - yhat)**2).sum()
    ss_tot = ((y - y.mean())**2).sum() + 1e-12
    r2 = 1.0 - ss_res/ss_tot
    return float(a), float(b), float(r2)


@torch.no_grad()
def encode_one(model, sample):
    V = sample["verts"].to(DEVICE)
    mass = sample["mass"].to(DEVICE)
    L = sample["L"].to(DEVICE)
    evals = sample["evals"].to(DEVICE)
    evecs = sample["evecs"].to(DEVICE)
    faces = sample["faces"].to(DEVICE)
    gradX = sample["gradX"].to(DEVICE)
    gradY = sample["gradY"].to(DEVICE)

    # ritorna solo globale
    Zg = model(V, mass, L, evals, evecs, faces, gradX, gradY,
               return_per_vertex=False, add_noise=False)
    return Zg.squeeze(0).float().cpu().numpy()  # (D,)


def main():
    print(f"🔎 Evaluating encoder checkpoint: {CKPT}")
    ckpt_path = CKPT if os.path.isabs(CKPT) else str(Path(BASE) / CKPT)
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    # Dataset
    dataset = GTReadyDataset(DATA_DIR)
    subj_map = build_subject_map(dataset)
    subjects = sorted(subj_map.keys())

    print(f"Subjects: {len(subjects)} | Meshes: {len(dataset.files)}")

    # GT distances
    D_pack = np.load(DIST_PATH, allow_pickle=True)
    D_orig = D_pack["D_orig"].astype(np.float64)
    D_orig /= np.max(D_orig[D_orig > 0])

    names = [str(n) for n in D_pack["names"]]
    name_to_idx = {}
    for i, n in enumerate(names):
        m = re.search(r"(id\d{4})", n)
        if m:
            name_to_idx[m.group(1)] = i
    assert len(name_to_idx) > 0, "Failed to parse subject ids from D_pack['names']"

    # Model
    model = DiffusionEncoderOnly(latent_dim=256, width=128, n_blocks=4, dropout=0.1).to(DEVICE)
    sd = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(sd, strict=True)
    model.eval()

    # ------------------------------------------------------------
    # 1) Compute embeddings for ALL meshes, then subject means
    # ------------------------------------------------------------
    subj_embeds = {}        # subj -> list[(D,)]
    subj_mean = {}          # subj -> (D,)
    intra_mse = {}          # subj -> float

    for subj in tqdm(subjects, desc="Encoding"):
        idxs = subj_map[subj]
        Zs = []
        for idx in idxs:
            sample = dataset[idx]
            Zs.append(encode_one(model, sample))
        Zs = np.stack(Zs, axis=0)          # (K,D)
        zm = Zs.mean(axis=0)               # (D,)
        subj_embeds[subj] = Zs
        subj_mean[subj] = zm
        # intra: mean squared deviation from mean (averaged over K and D)
        intra_mse[subj] = float(((Zs - zm[None, :])**2).mean())

    intra_vals = np.array(list(intra_mse.values()), dtype=np.float64)
    print("\n=== INTRA-SUBJECT (variants consistency) ===")
    print(f"Intra MSE: mean={intra_vals.mean():.6f} | median={np.median(intra_vals):.6f} | "
          f"p90={np.quantile(intra_vals, 0.9):.6f} | max={intra_vals.max():.6f}")

    # ------------------------------------------------------------
    # 2) Inter-subject distances vs GT distances
    # ------------------------------------------------------------
    kept_subjects = [s for s in subjects if s in name_to_idx]
    if len(kept_subjects) < 3:
        raise RuntimeError("Not enough subjects with GT distance entries.")

    Zmat = np.stack([subj_mean[s] for s in kept_subjects], axis=0)  # (S,D)
    # latent pairwise euclidean
    # efficient cdist
    G = (Zmat * Zmat).sum(axis=1, keepdims=True)
    D_lat = np.sqrt(np.maximum(G + G.T - 2.0*(Zmat @ Zmat.T), 0.0))

    idx = np.array([name_to_idx[s] for s in kept_subjects], dtype=int)
    D_gt = D_orig[np.ix_(idx, idx)]

    # take upper triangle (excluding diag)
    iu = np.triu_indices(D_gt.shape[0], k=1)
    gt_flat = D_gt[iu]
    lat_flat = D_lat[iu]

    pear = pearsonr(gt_flat, lat_flat)
    spear = spearmanr(gt_flat, lat_flat)
    slope, intercept, r2 = fit_slope_r2(gt_flat, lat_flat)

    print("\n=== INTER-SUBJECT STRUCTURE (latent vs GT distances) ===")
    print(f"Pairs: {len(gt_flat)} | Subjects used: {len(kept_subjects)}")
    print(f"Pearson:  {pear:.4f}")
    print(f"Spearman: {spear:.4f}")
    print(f"R^2:      {r2:.4f}")
    print(f"Slope:    {slope:.4f} | Intercept: {intercept:.4f}")

    # ------------------------------------------------------------
    # 3) Save artifacts for plotting later
    # ------------------------------------------------------------
    out_npz = os.path.join(OUT_DIR, "encoder_stage1_epoch35_eval.npz")
    np.savez_compressed(
        out_npz,
        kept_subjects=np.array(kept_subjects),
        Z_subject_mean=Zmat.astype(np.float32),
        D_lat=D_lat.astype(np.float32),
        D_gt=D_gt.astype(np.float32),
        intra_mse=intra_vals.astype(np.float32),
    )
    print(f"\n💾 Saved: {out_npz}")


if __name__ == "__main__":
    main()
