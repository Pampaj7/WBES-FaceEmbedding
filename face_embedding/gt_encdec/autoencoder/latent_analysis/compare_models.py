#!/usr/bin/env python3
"""
Test baseline model (geometric-only training) for latent-GT correlations.
Compares with latent-aware models to see if mode collapse was already present.
"""
import os
import numpy as np
import torch
import scipy.stats as st
import torch.nn.functional as F
from torch.utils.data import random_split

from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset
from diffusion_autoencoder import DiffusionAutoencoder

# ============================================================
# Configuration
# ============================================================

# Paths to your trained models
BASELINE_CHECKPOINT = "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/gt_encdec/autoencoder/results_diffusionAE/diffusionAE_5000_epoch45.pth"  # geo-only
LATENT_AWARE_CHECKPOINT = "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/gt_encdec/autoencoder/results_diffusionAE_latentaware_v2/latentaware_v3_best.pth"  # with latent losses

# Data
DATA_DIR = "../../../../datasets/GT_ready/npz_data/"
DIST_PATH = "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/gt_encdec/autoencoder/results_diffusionAE/dist_matrices_fields/D_orig_gt_normalized.npz"

# Model params
LATENT_DIM = 256
WIDTH = 128
N_BLOCKS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_SPLIT = 0.1
N_SAMPLES_TEST = 200  # number of meshes to test

# ============================================================
# Helper Functions
# ============================================================

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


def latent_identity_check(model, dataset, D_orig, name_to_idx, device, n_samples=200):
    """Compute Pearson/Spearman/R²/slope between D_orig and D_latent."""
    model.eval()
    
    # Sample from dataset
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    Z_list, idx_list = [], []

    print(f"   Encoding {len(indices)} meshes...")
    
    with torch.no_grad():
        for i in indices:
            s = dataset[i]
            V = s["verts"].to(device)
            mass, evals, evecs = s["mass"].to(device), s["evals"].to(device), s["evecs"].to(device)
            faces, L = s["faces"].to(device), s["L"].to(device)
            gX, gY = s["gradX"].to(device), s["gradY"].to(device)
            
            out = model(V, mass, L, evals, evecs, faces, gX, gY)
            
            if isinstance(out, (tuple, list)):
                Zg = out[-1]  # Z_global
            else:
                continue
            
            if Zg is None:
                continue
            if Zg.dim() == 1:
                Zg = Zg.unsqueeze(0)
            
            Z_list.append(Zg.cpu())
            base = s["name"][:-4] if s["name"].endswith(".npz") else s["name"]
            if base in name_to_idx:
                idx_list.append(name_to_idx[base])

    if len(Z_list) < 2 or len(idx_list) < 2:
        return None

    Z_all = torch.cat(Z_list, dim=0)
    
    # Use multi-scale distance (same as training)
    Z_norm = F.normalize(Z_all, p=2, dim=1)
    cos_sim = torch.mm(Z_norm, Z_norm.T)
    D_cos = 1.0 - cos_sim
    diff = Z_norm[:, None, :] - Z_norm[None, :, :]
    D_euc = torch.sqrt((diff ** 2).sum(-1))
    D_lat = (0.7 * D_cos + 0.3 * D_euc).cpu().numpy()
    
    # Normalize
    D_lat = (D_lat - D_lat.min()) / (D_lat.max() - D_lat.min() + 1e-8)

    # Get GT distances for same meshes
    idx_array = np.array(idx_list)
    D_gt = D_orig[np.ix_(idx_array, idx_array)]
    D_gt = (D_gt - D_gt.min()) / (D_gt.max() - D_gt.min() + 1e-8)

    # Compute correlations on upper triangle
    mask = np.triu_indices_from(D_gt, k=1)
    x, y = D_gt[mask], D_lat[mask]

    if x.std() < 1e-12 or y.std() < 1e-12:
        return None

    pear = st.pearsonr(x, y)[0]
    spear = st.spearmanr(x, y)[0]
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    slope = float(np.polyfit(x, y, 1)[0])

    return {
        "pearson": pear, 
        "spearman": spear, 
        "r2": r2, 
        "slope": slope,
        "n_pairs": len(x)
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("BASELINE vs LATENT-AWARE COMPARISON")
    print("=" * 70)
    
    # Load dataset
    print(f"\n📂 Loading dataset from {DATA_DIR}...")
    dataset = GTReadyDataset(DATA_DIR)
    dataset.files = dataset.files[:1000]  # same subset as training
    
    n_val = int(len(dataset) * VAL_SPLIT)
    n_train = len(dataset) - n_val
    _, val_set = random_split(dataset, [n_train, n_val])
    print(f"   Using {len(val_set)} validation samples")
    
    # Load GT distance matrix
    print(f"\n📂 Loading GT distance matrix from {DIST_PATH}...")
    D_pack = np.load(DIST_PATH, allow_pickle=True)
    D_orig = D_pack["D_orig"].astype(np.float64)
    norm_factor = np.max(D_orig[D_orig > 0]) if np.any(D_orig > 0) else 1.0
    D_orig = D_orig / norm_factor
    name_to_idx = build_name_index_map(D_pack["names"])
    print(f"   Distance matrix shape: {D_orig.shape}")
    
    # ========================================
    # TEST BASELINE MODEL
    # ========================================
    print(f"\n{'='*70}")
    print("🔵 TESTING BASELINE MODEL (Geometric-only training)")
    print(f"{'='*70}")
    
    if not os.path.exists(BASELINE_CHECKPOINT):
        print(f"❌ Baseline checkpoint not found: {BASELINE_CHECKPOINT}")
        print("   Skipping baseline test.")
        stats_baseline = None
    else:
        model_baseline = DiffusionAutoencoder(
            latent_dim=LATENT_DIM, 
            width=WIDTH, 
            n_blocks=N_BLOCKS
        ).to(DEVICE)
        
        print(f"   Loading checkpoint: {BASELINE_CHECKPOINT}")
        model_baseline.load_state_dict(torch.load(BASELINE_CHECKPOINT, map_location=DEVICE))
        print(f"   ✅ Checkpoint loaded")
        
        stats_baseline = latent_identity_check(
            model_baseline, val_set, D_orig, name_to_idx, DEVICE, n_samples=N_SAMPLES_TEST
        )
        
        if stats_baseline:
            print(f"\n📊 BASELINE RESULTS:")
            print(f"   Pearson:  {stats_baseline['pearson']:.4f}")
            print(f"   Spearman: {stats_baseline['spearman']:.4f}")
            print(f"   R²:       {stats_baseline['r2']:.4f}")
            print(f"   Slope:    {stats_baseline['slope']:.4f}")
            print(f"   N pairs:  {stats_baseline['n_pairs']}")
        else:
            print("   ⚠️ Failed to compute stats for baseline")
    
    # ========================================
    # TEST LATENT-AWARE MODEL (if available)
    # ========================================
    print(f"\n{'='*70}")
    print("🟢 TESTING LATENT-AWARE MODEL")
    print(f"{'='*70}")
    
    if not os.path.exists(LATENT_AWARE_CHECKPOINT):
        print(f"⚠️ Latent-aware checkpoint not found: {LATENT_AWARE_CHECKPOINT}")
        print("   Skipping latent-aware test.")
        stats_latent = None
    else:
        model_latent = DiffusionAutoencoder(
            latent_dim=LATENT_DIM, 
            width=WIDTH, 
            n_blocks=N_BLOCKS
        ).to(DEVICE)
        
        print(f"   Loading checkpoint: {LATENT_AWARE_CHECKPOINT}")
        checkpoint = torch.load(LATENT_AWARE_CHECKPOINT, map_location=DEVICE)
        
        # Handle both formats (with or without 'model_state_dict' key)
        if 'model_state_dict' in checkpoint:
            model_latent.load_state_dict(checkpoint['model_state_dict'])
        else:
            model_latent.load_state_dict(checkpoint)
        print(f"   ✅ Checkpoint loaded")
        
        stats_latent = latent_identity_check(
            model_latent, val_set, D_orig, name_to_idx, DEVICE, n_samples=N_SAMPLES_TEST
        )
        
        if stats_latent:
            print(f"\n📊 LATENT-AWARE RESULTS:")
            print(f"   Pearson:  {stats_latent['pearson']:.4f}")
            print(f"   Spearman: {stats_latent['spearman']:.4f}")
            print(f"   R²:       {stats_latent['r2']:.4f}")
            print(f"   Slope:    {stats_latent['slope']:.4f}")
            print(f"   N pairs:  {stats_latent['n_pairs']}")
        else:
            print("   ⚠️ Failed to compute stats for latent-aware model")
    
    # ========================================
    # COMPARISON
    # ========================================
    print(f"\n{'='*70}")
    print("📊 COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    if stats_baseline and stats_latent:
        print(f"\n{'Metric':<12} | {'Baseline':<12} | {'Latent-Aware':<12} | {'Δ':<12}")
        print("-" * 60)
        print(f"{'Pearson':<12} | {stats_baseline['pearson']:>12.4f} | {stats_latent['pearson']:>12.4f} | {stats_latent['pearson']-stats_baseline['pearson']:>+12.4f}")
        print(f"{'Spearman':<12} | {stats_baseline['spearman']:>12.4f} | {stats_latent['spearman']:>12.4f} | {stats_latent['spearman']-stats_baseline['spearman']:>+12.4f}")
        print(f"{'R²':<12} | {stats_baseline['r2']:>12.4f} | {stats_latent['r2']:>12.4f} | {stats_latent['r2']-stats_baseline['r2']:>+12.4f}")
        print(f"{'Slope':<12} | {stats_baseline['slope']:>12.4f} | {stats_latent['slope']:>12.4f} | {stats_latent['slope']-stats_baseline['slope']:>+12.4f}")
        
        print(f"\n🎯 INTERPRETATION:")
        if stats_latent['pearson'] > stats_baseline['pearson'] + 0.05:
            print("   ✅ Latent-aware training SIGNIFICANTLY IMPROVED embeddings")
        elif stats_latent['pearson'] > stats_baseline['pearson']:
            print("   ✅ Latent-aware training improved embeddings")
        elif stats_latent['pearson'] > stats_baseline['pearson'] - 0.05:
            print("   ≈  Latent-aware training maintained similar embedding quality")
        else:
            print("   ❌ Latent-aware training degraded embeddings")
        
        if abs(stats_latent['slope'] - 1.0) < abs(stats_baseline['slope'] - 1.0):
            print("   ✅ Latent-aware has better scale preservation (slope closer to 1.0)")
        
        if stats_baseline['pearson'] < 0.75:
            print(f"\n   ⚠️  BASELINE had poor embeddings (ρ={stats_baseline['pearson']:.3f})")
            print("   → Mode collapse was already present in geometric-only training")
            print("   → Latent losses expose the problem, not cause it")
    
    elif stats_baseline:
        print(f"\n✅ Baseline tested successfully")
        print(f"   This is your reference point for latent space quality")
    
    print(f"\n{'='*70}")
    print("✅ Test complete")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()