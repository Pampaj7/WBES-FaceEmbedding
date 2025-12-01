#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, linregress

# ============================================================
# CONFIG
# ============================================================
DIST_FILE = "../encoder_only/latent_distances_encoderonly.npz"   
#DIST_FILE = "../test_safe_latent/latent_distances_autoencoder.npz"  

OUT_PLOT  = "scatter_autoencoder.png"

# ============================================================
# LOAD MATRICES
# ============================================================
data = np.load(DIST_FILE)
D_orig = data["D_orig"].astype(np.float64)
D_lat  = data["D_lat"].astype(np.float64)

print("📂 Loaded:", DIST_FILE)
print("   shape =", D_orig.shape)

# ============================================================
# FLATTEN upper triangle
# ============================================================
def upper_flat(M):
    return M[np.triu_indices_from(M, k=1)]

x = upper_flat(D_orig)
y = upper_flat(D_lat)

# Normalizzazione [0,1]
x = (x - x.min()) / (x.max() - x.min() + 1e-9)
y = (y - y.min()) / (y.max() - y.min() + 1e-9)

# ============================================================
# CORRELATION & REGRESSION
# ============================================================
rho, p_rho = spearmanr(x, y)
tau, p_tau = kendalltau(x, y)

slope, intercept, r_val, _, _ = linregress(x, y)
r2 = r_val**2

print(f"\n📈 Spearman ρ = {rho:.4f}  (p={p_rho:.3e})")
print(f"📈 Kendall τ  = {tau:.4f}  (p={p_tau:.3e})")
print(f"📉 slope = {slope:.4f}, R² = {r2:.4f}")

# ============================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================
N_BOOT = 200
rng = np.random.default_rng(42)

slopes = []
for _ in range(N_BOOT):
    idx = rng.choice(len(x), size=len(x), replace=True)
    s, _, _, _, _ = linregress(x[idx], y[idx])
    slopes.append(s)

ci_low, ci_high = np.percentile(slopes, [2.5, 97.5])

# ============================================================
# PLOT
# ============================================================
fig, ax = plt.subplots(figsize=(6, 6))

hb = ax.hexbin(x, y, gridsize=80, cmap="viridis")
cbar = plt.colorbar(hb, ax=ax)
cbar.set_label("Density")

xx = np.linspace(0, 1, 100)
ax.plot(xx, slope*xx + intercept, "r--", lw=2, label=fr"$R^2$={r2:.3f}")
ax.plot(xx, xx, "k:", lw=1.2, label="y = x")

ax.fill_between(xx, ci_low*xx, ci_high*xx, color="red", alpha=0.15,
                label="95% conf. interval")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal", adjustable="box")

ax.set_xlabel("D_orig (normalized)")
ax.set_ylabel("D_latent (normalized)")
ax.set_title(f"Latent vs GT — ρ={rho:.2f}, slope={slope:.2f}")

ax.legend(loc="upper left", frameon=True)
plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=250)
plt.close(fig)

print(f"\n✅ Saved plot to {OUT_PLOT}")
