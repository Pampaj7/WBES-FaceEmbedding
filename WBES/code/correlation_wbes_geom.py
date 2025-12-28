import seaborn as sns
import os
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# === CONFIG ===
RESULTS_DIR = "results"
SAVE_DIR = "z_correlation_wbes_geom_byF"
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

# === Collect all data
all_rows = []

for method in tqdm(METHODS, desc="Loading data"):
    wbes_csv = os.path.join(RESULTS_DIR, method, f"{method}-wbes_inter_F.csv")
    error_csv = os.path.join(
        RESULTS_DIR, method, f"{method}-geom_error_vs_F.csv")

    if not os.path.exists(wbes_csv) or not os.path.exists(error_csv):
        print(f"[!] Skipping {method} — missing files")
        continue

    df_wbes = pd.read_csv(wbes_csv)
    df_err = pd.read_csv(error_csv)

    df_wbes.columns = [c.strip().lower() for c in df_wbes.columns]
    df_err.columns = [c.strip().lower() for c in df_err.columns]
    df_wbes.rename(columns={"wbse": "wbes"}, inplace=True)

    df_wbes["method"] = method
    df_err["method"] = method
    df = pd.merge(df_wbes, df_err, on=["f", "method"], how="inner")

    all_rows.append(df)

df_all = pd.concat(all_rows, ignore_index=True)
df_all["f"] = df_all["f"].astype(int)

# Applica override per il plot (non modifica le analisi)
df_all["f_plot"] = df_all.apply(
    lambda row: F_OVERRIDES.get(row["method"], {}).get(row["f"], row["f"]),
    axis=1
)

# === Per ogni F, fai correlazione e plot ===
records = []

for f_val in sorted(df_all["f_plot"].unique()):
    sub = df_all[df_all["f_plot"] == f_val]
    if len(sub) < 2:
        continue

    try:
        r_pearson = scipy.stats.pearsonr(
            sub["wbes"], sub["mean_vertex_error"])[0]
    except:
        r_pearson = float("nan")

    try:
        r_spearman = scipy.stats.spearmanr(
            sub["wbes"], sub["mean_vertex_error"])[0]
    except:
        r_spearman = float("nan")

    records.append({
        "F": f_val,
        "pearson_r": r_pearson,
        "spearman_r": r_spearman,
        "n_methods": len(sub)
    })

    # Plot
    plt.figure(figsize=(6, 5))
    plt.scatter(sub["wbes"], sub["mean_vertex_error"], s=80, color="steelblue")
    for _, row in sub.iterrows():
        plt.text(row["wbes"], row["mean_vertex_error"],
                 row["method"], fontsize=7)

    plt.title(
        f"F = {f_val} | Pearson r = {r_pearson:.3f}, Spearman r = {r_spearman:.3f}")
    plt.xlabel("WBES (inter-subject)")
    plt.ylabel("Mean Vertex L2 Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(
        SAVE_DIR, f"wbes_vs_geomerror_F{int(f_val):02d}.png"))
    plt.close()
    print(f"✅ Saved: wbes_vs_geomerror_F{int(f_val):02d}.png")

# === SAVE SUMMARY CSV ===
summary = pd.DataFrame(records)
summary.to_csv(os.path.join(SAVE_DIR, "summary.csv"), index=False)
print(f"✅ Saved correlation summary to {SAVE_DIR}/summary.csv")
