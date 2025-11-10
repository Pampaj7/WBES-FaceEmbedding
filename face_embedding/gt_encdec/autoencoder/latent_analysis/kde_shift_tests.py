# kde_shift_tests.py — MMD con tqdm e parallelizzazione joblib
import os, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import gaussian_kde, ks_2samp, cramervonmises_2samp, wasserstein_distance

# --- Path robusti relativi a questo file ---
HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "dist_matrices_fields/distance_matrices_fields.npz")
OUT_DIR = os.path.join(HERE, "dist_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

def upper_norm(D):
    x = D[np.triu_indices_from(D, 1)]
    x = (x - x.min()) / (x.max() - x.min() + 1e-12)
    return x

def mmd_rbf(x, y, gamma=None, n_perm=200, seed=0, n_jobs=1):
    """
    MMD^2 con kernel RBF (1D), versione memory-safe.
    Evita di costruire matrici NxN: usa solo somme incrementali.
    """
    rng = np.random.default_rng(seed)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = np.vstack([x, y])

    # median heuristic
    if gamma is None:
        # usa un piccolo sottocampione per stimare gamma
        idx = rng.choice(len(z), size=min(2000, len(z)), replace=False)
        z_s = z[idx]
        d = np.abs(z_s - z_s.T)
        med = np.median(d[np.triu_indices_from(d, 1)])
        med = float(med) if med > 1e-12 else 1.0
        gamma = 1.0 / (2.0 * (med ** 2))

    def k(a, b):
        d2 = (a - b.T) ** 2
        return np.exp(-gamma * d2)

    def mmd2(a, b):
        """Computa MMD senza salvare tutte le matrici."""
        na, nb = len(a), len(b)
        # campiona sottoinsieme per ridurre costo quadratico
        max_samp = min(2000, na, nb)
        ia = rng.choice(na, max_samp, replace=False)
        ib = rng.choice(nb, max_samp, replace=False)
        Ka = np.mean(np.exp(-gamma * (a[ia] - a[ia].T) ** 2))
        Kb = np.mean(np.exp(-gamma * (b[ib] - b[ib].T) ** 2))
        Kab = np.mean(np.exp(-gamma * (a[ia] - b[ib].T) ** 2))
        return Ka + Kb - 2 * Kab

    base_mmd = mmd2(x, y)

    # permutation test (streaming, no big arrays)
    n = len(x)
    perm_mmds = []
    for _ in tqdm(range(n_perm), desc=f"MMD {n_perm} perm", leave=True):
        idx = rng.permutation(len(z))
        xa, ya = z[idx[:n]], z[idx[n:]]
        perm_mmds.append(mmd2(xa, ya))

    perm_mmds = np.array(perm_mmds)
    p_val = np.mean(perm_mmds >= base_mmd)
    return float(base_mmd), float(p_val)


# === Caricamento matrici ===
data = np.load(IN)
D_orig     = data["D_orig"]
D_lat_mean = data["D_lat_mean"]
D_lat_field= data["D_lat_field"]

x = upper_norm(D_orig)
y_mean  = upper_norm(D_lat_mean)
y_field = upper_norm(D_lat_field)

def compare(name, y):
    print(f"\n🔍 Analizzo {name} ...")
    ks = ks_2samp(x, y)
    cvm = cramervonmises_2samp(x, y)
    emd = wasserstein_distance(x, y)
    mmd, p_mmd = mmd_rbf(x, y, n_perm=200, n_jobs=-1)

    # KDE plot
    xs = np.linspace(0, 1, 400)
    kde_x = gaussian_kde(x); kde_y = gaussian_kde(y)
    plt.figure(figsize=(7,4))
    plt.plot(xs, kde_x(xs), label="orig (GT)", lw=2)
    plt.plot(xs, kde_y(xs), label=name, lw=2)
    plt.fill_between(xs, np.minimum(kde_x(xs), kde_y(xs)), alpha=0.15)
    plt.xlabel("Pairwise distance (normalized)"); plt.ylabel("Density")
    plt.title(f"KDE — GT vs {name}")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"kde_{name}.png"), dpi=200); plt.close()

    return {
        "name": name,
        "Delta_mu": float(y.mean() - x.mean()),
        "EMD": float(emd),
        "KS_stat": float(ks.statistic), "KS_p": float(ks.pvalue),
        "CvM_stat": float(cvm.statistic), "CvM_p": float(cvm.pvalue),
        "MMD": float(mmd), "MMD_p": float(p_mmd)
    }

rows = [
    compare("lat_field", y_field),
    compare("lat_mean",  y_mean),
]

# === Salvataggio CSV ===
hdr = ["name","Delta_mu","EMD","KS_stat","KS_p","CvM_stat","CvM_p","MMD","MMD_p"]
with open(os.path.join(OUT_DIR,"kde_shift_stats.csv"),"w") as f:
    f.write(",".join(hdr)+"\n")
    for r in rows:
        f.write(",".join(str(r[h]) for h in hdr)+"\n")

print("\n✅ KDE/test salvati in:", OUT_DIR)
for r in rows:
    print(r)
