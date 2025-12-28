import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from tqdm import tqdm

# === CONFIG ===
RESULTS_DIR = "results"
SAVE_DIR = "z_correlation_within_geom_per_subject"
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

F_TARGET = 15  # Cambia qui per il valore di F che vuoi visualizzare

# === Load and merge all ===
all_rows = []

for method in tqdm(METHODS, desc="Methods"):
    w_path = os.path.join(RESULTS_DIR, method,
                          f"{method}-within_per_subject_v2.csv")
    e_path = os.path.join(RESULTS_DIR, method,
                          f"{method}-geom_error_per_subject.csv")

    if not os.path.exists(w_path) or not os.path.exists(e_path):
        print(f"[!] Skipping {method} — missing CSVs")
        continue

    df_w = pd.read_csv(w_path)
    df_e = pd.read_csv(e_path)

    df_w.columns = df_w.columns.str.lower()
    df_e.columns = df_e.columns.str.lower()

    df = pd.merge(df_w, df_e, on=["method", "f", "subject"])
    for _, row in df.iterrows():
        orig_f = int(row["f"])
        actual_f = F_OVERRIDES.get(method, {}).get(orig_f, orig_f)
        all_rows.append({
            "method": method,
            "F": actual_f,
            "subject": row["subject"],
            "within": row["within"],
            "error": row["error"]
        })

df_all = pd.DataFrame(all_rows)
print(f"✅ Total matched rows: {len(df_all)}")

# === Plot grid: 2x3 subplot per metodo ===
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, method in enumerate(METHODS):
    ax = axes[i]
    sub = df_all[(df_all["method"] == method) & (df_all["F"] == F_TARGET)]

    if len(sub) < 2:
        ax.set_title(f"{method} (No data)")
        ax.axis('off')
        continue

    try:
        r_pearson = scipy.stats.pearsonr(sub["within"], sub["error"])[0]
    except:
        r_pearson = float("nan")

    try:
        r_spearman = scipy.stats.spearmanr(sub["within"], sub["error"])[0]
    except:
        r_spearman = float("nan")

    ax.scatter(sub["within"], sub["error"], s=40, color='royalblue')
    ax.set_title(
        f"{method} | r={r_pearson:.2f}, ρ={r_spearman:.2f}", fontsize=10)
    ax.set_xlabel("Within-subject")
    ax.set_ylabel("Geom. Error")
    ax.grid(True)

plt.suptitle(f"Correlation per Subject (F = {F_TARGET})", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

out_path = os.path.join(SAVE_DIR, f"grid_within_vs_geom_F{F_TARGET:02d}.png")
plt.savefig(out_path)
plt.close()
print(f"✅ Saved grid plot: {out_path}")
