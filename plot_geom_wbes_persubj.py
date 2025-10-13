import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
from tqdm import tqdm

# === CONFIG ===
RESULTS_DIR = "results"
SAVE_DIR = "z_correlation_wbes_geom_per_subject"
os.makedirs(SAVE_DIR, exist_ok=True)

METHODS = [
    "3DDFAV2_23470_neutral",
    "3DDFAV3_neutral",
    "Deep3DFace_23470_neutral",
    "SynergyNet_neutral",
    "INORig_23470_neutral",
    "3DI_neutral"
]

F_OVERRIDES = {
    "3DI_neutral": {1: 10},
    "INORig_23470_neutral": {1: 5, 2: 10, 3: 15}
}

F_TARGET = 1  # Cambia qui per il valore di F che vuoi visualizzare

# === Load and merge all ===
all_rows = []

for method in tqdm(METHODS, desc="Methods"):
    wbes_path = os.path.join(
        RESULTS_DIR, method, f"{method}-wbes_per_subject_v2.csv")
    e_path = os.path.join(RESULTS_DIR, method,
                          f"{method}-geom_error_per_subject.csv")

    if not os.path.exists(wbes_path) or not os.path.exists(e_path):
        print(f"[!] Skipping {method} — missing CSVs")
        continue

    df_wbes = pd.read_csv(wbes_path)
    df_error = pd.read_csv(e_path)

    df_wbes.columns = df_wbes.columns.str.lower()
    df_error.columns = df_error.columns.str.lower()

    df = pd.merge(df_wbes, df_error, on=["method", "f", "subject"])
    for _, row in df.iterrows():
        orig_f = int(row["f"])
        actual_f = F_OVERRIDES.get(method, {}).get(orig_f, orig_f)
        all_rows.append({
            "method": method,
            "F": actual_f,
            "subject": row["subject"],
            "wbes": row["wbse"],
            "error": row["error"]
        })

df_all = pd.DataFrame(all_rows)
print(f"✅ Total matched rows: {len(df_all)}")

# === Plot grid: 2x3 subplot per metodo ===
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

mean_rows = []

for i, method in enumerate(METHODS):
    ax = axes[i]
    sub = df_all[(df_all["method"] == method) & (df_all["F"] == F_TARGET)]

    # Clip per outlier
    sub = sub[(sub["wbes"] < 5) & (sub["error"] < 0.01)]

    if len(sub) < 2:
        ax.set_title(f"{method} (No data)")
        ax.axis('off')
        continue

    x = sub["wbes"]
    y = sub["error"]

    try:
        r_pearson = scipy.stats.pearsonr(x, y)[0]
    except:
        r_pearson = float("nan")

    try:
        r_spearman = scipy.stats.spearmanr(x, y)[0]
    except:
        r_spearman = float("nan")

    # Scatter con alpha
    ax.scatter(x, y, s=30, alpha=0.6, color='darkorange')

    # Linea di regressione
    try:
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m * x + b, color="gray", linestyle="--", linewidth=1)
    except:
        pass

    ax.set_title(
        f"{method} | r={r_pearson:.2f}, ρ={r_spearman:.2f} (n={len(sub)})", fontsize=10)
    ax.set_xlabel("WBES")
    ax.set_ylabel("Geom. Error")
    ax.grid(True)

    # Aggiungi punto medio alla lista (facoltativo)
    mean_rows.append({
        "method": method,
        "F": F_TARGET,
        "mean_wbes": x.mean(),
        "mean_error": y.mean()
    })

# === Salva il grid plot
plt.suptitle(f"Correlation per Subject (F = {F_TARGET})", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

out_path = os.path.join(SAVE_DIR, f"grid_wbes_vs_geom_F{F_TARGET:02d}.png")
plt.savefig(out_path)
plt.close()
print(f"✅ Saved grid plot: {out_path}")

# === Salva CSV dei punti medi per possibile plot separato
df_means = pd.DataFrame(mean_rows)
df_means.to_csv(os.path.join(
    SAVE_DIR, f"meanpoints_wbes_geom_F{F_TARGET:02d}.csv"), index=False)
print("✅ Saved mean-point table.")
