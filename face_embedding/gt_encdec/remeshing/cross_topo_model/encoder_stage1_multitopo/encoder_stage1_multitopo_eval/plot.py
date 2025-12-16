#!/usr/bin/env python3
"""
Plots for Stage-1 Encoder (multi-topology)
------------------------------------------
Input: .npz produced by analysis_enc.py

Generates:
1) GT vs Latent distance scatter
2) GT vs Latent with y=x and fitted line
3) Intra-subject MSE distribution
4) Intra vs Inter latent distance separation
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
NPZ_PATH = "encoder_stage1_epoch35_eval.npz"
OUT_DIR = "."
os.makedirs(OUT_DIR, exist_ok=True)

SAVE = True          # salva PNG
SHOW = False         # mostra a schermo
DPI = 150

# ============================================================
# LOAD
# ============================================================
data = np.load(NPZ_PATH)

D_gt = data["D_gt"]              # (S,S)
D_lat = data["D_lat"]            # (S,S)
intra_mse = data["intra_mse"]    # (S,)

# upper triangle (pairs)
iu = np.triu_indices(D_gt.shape[0], k=1)
gt = D_gt[iu]
lat = D_lat[iu]

# ============================================================
# HELPERS
# ============================================================
def savefig(name):
    if SAVE:
        path = os.path.join(OUT_DIR, name)
        plt.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"💾 saved: {path}")
    if SHOW:
        plt.show()
    plt.close()

# ============================================================
# 1) Scatter: GT vs Latent
# ============================================================
plt.figure(figsize=(6, 6))
plt.scatter(gt, lat, s=2, alpha=0.2)
plt.xlabel("GT distance (normalized)")
plt.ylabel("Latent distance")
plt.title("Latent vs GT distances")
plt.grid(True)
savefig("01_scatter_gt_vs_latent.png")

# ============================================================
# 2) Scatter + ideal y=x + fitted line
# ============================================================
# linear fit
A = np.vstack([gt, np.ones_like(gt)]).T
slope, intercept = np.linalg.lstsq(A, lat, rcond=None)[0]

x = np.linspace(gt.min(), gt.max(), 300)

plt.figure(figsize=(6, 6))
plt.scatter(gt, lat, s=2, alpha=0.15, label="Pairs")
plt.plot(x, x, "k--", label="Ideal y=x")
plt.plot(x, slope*x + intercept, "r",
         label=f"Fit y={slope:.2f}x + {intercept:.2f}")
plt.xlabel("GT distance")
plt.ylabel("Latent distance")
plt.title("Scale distortion in latent space")
plt.legend()
plt.grid(True)
savefig("02_scatter_with_fit.png")

# ============================================================
# 3) Intra-subject MSE distribution
# ============================================================
plt.figure(figsize=(6, 4))
plt.hist(intra_mse, bins=50)
plt.xlabel("Intra-subject MSE")
plt.ylabel("Count")
plt.title("Cross-topology identity consistency")
plt.grid(True)
savefig("03_intra_subject_mse_hist.png")

# ============================================================
# 4) Intra vs Inter distance comparison
# ============================================================
plt.figure(figsize=(6, 4))
plt.hist(lat, bins=120, alpha=0.6, label="Inter-subject")
plt.hist(intra_mse, bins=50, alpha=0.6, label="Intra-subject")
plt.yscale("log")
plt.xlabel("Distance")
plt.ylabel("Count (log)")
plt.title("Intra vs Inter latent distances")
plt.legend()
plt.grid(True)
savefig("04_intra_vs_inter.png")

# ============================================================
# 5) Optional: density plot (hexbin)
# ============================================================
plt.figure(figsize=(6, 6))
hb = plt.hexbin(gt, lat, gridsize=80, bins="log")
plt.colorbar(hb, label="log10(count)")
plt.xlabel("GT distance")
plt.ylabel("Latent distance")
plt.title("Latent vs GT (density)")
plt.grid(True)
savefig("05_hexbin_density.png")

print("\n✅ All plots generated.")
