import os
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt

# === CONFIG ===
GEOM_DIR = "results"
COMPLEX_CSV = "Z_cosine/results/cosine_wbesstyle_disjoint.csv"
SAVE_DIR = "z_correlation_geom_complex_perF"
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

# === LOAD DATA ===
df_complex = pd.read_csv(COMPLEX_CSV)
df_complex.columns = [c.strip().lower() for c in df_complex.columns]
df_complex = df_complex.rename(columns={"f": "F"})
df_complex["F"] = df_complex["F"].astype(int)

# === Collect all data (geom error + complex corr) ===
all_rows = []

for method in METHODS:
    geom_csv = os.path.join(GEOM_DIR, method, f"{method}-geom_error_vs_F.csv")
    if not os.path.exists(geom_csv):
        continue

    df_geom = pd.read_csv(geom_csv)
    df_geom.columns = [c.strip().lower() for c in df_geom.columns]
    df_geom["F"] = df_geom["f"].astype(int)

    df_m = pd.merge(
        df_geom[["F", "mean_vertex_error"]],
        df_complex[df_complex["method"] == method][["F", "complex_corr"]],
        on="F", how="inner"
    )

    for _, row in df_m.iterrows():
        F_orig = int(row["F"])
        F_real = F_OVERRIDES.get(method, {}).get(F_orig, F_orig)
        all_rows.append({
            "method": method,
            "F_orig": F_orig,
            "F_real": F_real,
            "error": row["mean_vertex_error"],
            "complex_corr": row["complex_corr"]
        })

df_all = pd.DataFrame(all_rows)
print(df_all["method"].value_counts())
print("F_real values present:", sorted(df_all["F_real"].unique()))
print(f"✅ Loaded {len(df_all)} total rows")

# === Plot per ogni F_real ===
summary = []

for f_real in sorted(df_all["F_real"].unique()):
    sub = df_all[df_all["F_real"] == f_real]
    if len(sub) < 2:
        print(f"[!] F_real = {f_real}: not enough methods")
        continue

    try:
        r_pearson = scipy.stats.pearsonr(sub["error"], sub["complex_corr"])[0]
    except:
        r_pearson = float("nan")
    try:
        r_spearman = scipy.stats.spearmanr(
            sub["error"], sub["complex_corr"])[0]
    except:
        r_spearman = float("nan")

    summary.append({
        "F_real": f_real,
        "n_methods": len(sub),
        "pearson_r": r_pearson,
        "spearman_r": r_spearman
    })

    # Plot
    plt.figure(figsize=(6, 5))
    plt.scatter(sub["complex_corr"], sub["error"], color='darkorange', s=80)

    for _, row in sub.iterrows():
        label = f"{row['method']} (F={row['F_orig']})"
        plt.text(row["complex_corr"], row["error"], label, fontsize=7)

    plt.title(
        f"F_real = {f_real} | Pearson r = {r_pearson:.3f}, Spearman r = {r_spearman:.3f}")
    plt.xlabel("Complex correlation")
    plt.ylabel("Mean Vertex L2 Error")
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(
        SAVE_DIR, f"geom_vs_complex_F{int(f_real):02d}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Saved: {out_path}")

# === Save summary
df_summary = pd.DataFrame(summary)
df_summary.to_csv(os.path.join(
    SAVE_DIR, "correlation_summary.csv"), index=False)
print("✅ Saved correlation summary CSV.")
