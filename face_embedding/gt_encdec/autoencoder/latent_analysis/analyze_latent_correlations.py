#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, linregress
from sklearn.metrics import pairwise_distances
try:
    from skbio.stats.distance import mantel
    HAVE_MANTEL = True
except ImportError:
    HAVE_MANTEL = False
    print("⚠️  scikit-bio non installato: salto Mantel test (pip install scikit-bio)")

# === PATH ===
DIST_FILE = "results_diffusionAE/dist_matrices_fields/distance_matrices_fields.npz"
OUT_DIR   = "results_diffusionAE/dist_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# === LOAD DATA ===
data = np.load(DIST_FILE)
D_orig = data["D_orig"].astype(np.float64)
D_lat_mean = data["D_lat_mean"].astype(np.float64)
D_lat_field = data["D_lat_field"].astype(np.float64)

# Normalizzazione [0,1] per confrontare spazi diversi
def normalize_matrix(M):
    M = M - M.min()
    M = M / (M.max() + 1e-9)
    np.fill_diagonal(M, 0.0)
    return M

D_orig = normalize_matrix(D_orig)
D_lat_mean = normalize_matrix(D_lat_mean)
D_lat_field = normalize_matrix(D_lat_field)

names = data["names"]

print(f"📂 Loaded distance matrices: {D_orig.shape[0]} samples")

# === Flatten upper triangle ===
def upper_flat(M): return M[np.triu_indices_from(M, k=1)]

x_orig = upper_flat(D_orig)
y_mean = upper_flat(D_lat_mean)
y_field = upper_flat(D_lat_field)

# === Compute correlations ===
ρ_mean, pρ_mean = spearmanr(x_orig, y_mean)
ρ_field, pρ_field = spearmanr(x_orig, y_field)
τ_mean, pτ_mean = kendalltau(x_orig, y_mean)
τ_field, pτ_field = kendalltau(x_orig, y_field)

print("\n📈 Correlations:")
print(f"  Spearman ρ (orig vs global): {ρ_mean:.3f} (p={pρ_mean:.3e})")
print(f"  Spearman ρ (orig vs field):  {ρ_field:.3f} (p={pρ_field:.3e})")
print(f"  Kendall τ (orig vs global): {τ_mean:.3f} (p={pτ_mean:.3e})")
print(f"  Kendall τ (orig vs field):  {τ_field:.3f} (p={pτ_field:.3e})")

# === Mantel test (optional) ===
if HAVE_MANTEL:
    from skbio.stats.distance import DistanceMatrix
    mρ_mean, mp_mean, _ = mantel(DistanceMatrix(D_orig), DistanceMatrix(D_lat_mean), method='spearman', permutations=999)
    mρ_field, mp_field, _ = mantel(DistanceMatrix(D_orig), DistanceMatrix(D_lat_field), method='spearman', permutations=999)
    print(f"  Mantel ρ (orig vs global): {mρ_mean:.3f} (p={mp_mean:.3e})")
    print(f"  Mantel ρ (orig vs field):  {mρ_field:.3f} (p={mp_field:.3e})")


def scatter_plot(D_orig, D_lat, title, out_path, n_bootstrap=200):
    # --- Flatten e normalizza ---
    x = D_orig[np.triu_indices_from(D_orig, 1)]
    y = D_lat[np.triu_indices_from(D_lat, 1)]
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())

    # --- Regressione lineare ---
    slope, intercept, r_value, _, _ = linregress(x, y)
    r2 = r_value**2

    # --- Bootstrapping per banda 95% ---
    boot_slopes = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(len(x), size=len(x), replace=True)
        s, _, _, _, _ = linregress(x[idx], y[idx])
        boot_slopes.append(s)
    ci_low, ci_high = np.percentile(boot_slopes, [2.5, 97.5])

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(x, y, gridsize=80, cmap="viridis")
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label("Density")

    xx = np.linspace(0, 1, 100)
    ax.plot(xx, slope*xx + intercept, "r--", lw=2, label=fr"$R^2$={r2:.3f}")
    ax.plot(xx, xx, "k:", lw=1.2, label="y = x")
    ax.fill_between(xx, ci_low*xx, ci_high*xx, color="red", alpha=0.15,
                    label="95% conf. interval")

    # --- Assi e stile ---
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("D_orig (normalized)")
    ax.set_ylabel("D_latent (normalized)")
    ax.legend(loc="upper left", frameon=True)
    ax.set_title(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close(fig)


scatter_plot(D_orig, D_lat_mean, f"Preservation (Global Latent) — ρ={ρ_mean:.2f}", "scatter_global.png")
scatter_plot(D_orig, D_lat_field, f"Preservation (Per-vertex Field) — ρ={ρ_field:.2f}", "scatter_field.png")

print(f"\n✅ Saved plots and stats in: {OUT_DIR}")
print(f"   - scatter_global.png")
print(f"   - scatter_field.png")
print("   - correlations printed above")
