import os
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt

# === CONFIG ===
RESULTS_DIR = "results"
SAVE_DIR = "z_correlation_geom_cv"
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

for method in METHODS:
    path_cv = os.path.join(RESULTS_DIR, method, f"{method}-wbes_inter_F.csv")
    path_err = os.path.join(RESULTS_DIR, method,
                            f"{method}-geom_error_vs_F.csv")

    if not os.path.exists(path_cv) or not os.path.exists(path_err):
        print(f"[!] Skipping {method} — missing files")
        continue

    df_cv = pd.read_csv(path_cv)
    df_err = pd.read_csv(path_err)

    df_cv.columns = [c.strip().lower() for c in df_cv.columns]
    df_err.columns = [c.strip().lower() for c in df_err.columns]
    df_cv["f"] = df_cv["f"].astype(int)
    df_err["f"] = df_err["f"].astype(int)

    df_m = pd.merge(df_cv[["f", "within_cv"]], df_err[[
                    "f", "mean_vertex_error"]], on="f", how="inner")
    df_m["method"] = method

    all_rows.extend(df_m.to_dict(orient="records"))

# === Convert to DataFrame
df_all = pd.DataFrame(all_rows)

# === Apply F override globally
df_all["F_plot"] = df_all.apply(
    lambda row: F_OVERRIDES.get(row["method"], {}).get(row["f"], row["f"]),
    axis=1
)

# === Plot per F reale (dopo override)
for f_val in sorted(df_all["F_plot"].unique()):
    sub = df_all[df_all["F_plot"] == f_val]
    if len(sub) < 2:
        continue

    try:
        r_pearson = scipy.stats.pearsonr(
            sub["within_cv"], sub["mean_vertex_error"])[0]
    except:
        r_pearson = float("nan")

    try:
        r_spearman = scipy.stats.spearmanr(
            sub["within_cv"], sub["mean_vertex_error"])[0]
    except:
        r_spearman = float("nan")

    # Plot
    plt.figure(figsize=(6, 5))
    plt.scatter(sub["within_cv"], sub["mean_vertex_error"],
                s=80, color="green")

    for _, row in sub.iterrows():
        plt.text(row["within_cv"], row["mean_vertex_error"],
                 row["method"], fontsize=7)

    plt.title(
        f"F = {f_val} | Pearson r = {r_pearson:.3f}, Spearman r = {r_spearman:.3f}")
    plt.xlabel("Intra-subject Coefficient of Variation (CV)")
    plt.ylabel("Mean Vertex L2 Error")
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(SAVE_DIR, f"cv_vs_geom_F{int(f_val):02d}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Saved: {out_path}")
