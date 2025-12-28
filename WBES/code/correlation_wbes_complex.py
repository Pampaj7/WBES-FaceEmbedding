import os
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt

# === Config ===
COSINE_CSV = "Z_cosine/results/cosine_wbesstyle_disjoint.csv"
GEOM_DIR = "results"
OUTPUT_DIR = "z_correlation_wbes_complex"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Carica cosine/complex correlation ===
df_corr = pd.read_csv(COSINE_CSV)
df_corr.columns = [c.lower().strip() for c in df_corr.columns]

# === Carica tutti i geom_error_vs_F ===
rows = []
for method in df_corr["method"].unique():
    path = os.path.join(GEOM_DIR, method, f"{method}-geom_error_vs_F.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        df["method"] = method
        rows.append(df)

df_geom = pd.concat(rows, ignore_index=True)

# === Merge su method + f ===
df_merged = df_corr.merge(df_geom, on=["method", "f"], how="inner")

# === Crea un'immagine per ciascun metodo ===
summary = []

for method in df_merged["method"].unique():
    sub = df_merged[df_merged["method"] == method]
    if len(sub) < 2:
        continue

    try:
        r_pearson = scipy.stats.pearsonr(sub["mean_vertex_error"], sub["complex_corr"])[0]
    except:
        r_pearson = float("nan")

    try:
        r_spearman = scipy.stats.spearmanr(sub["mean_vertex_error"], sub["complex_corr"])[0]
    except:
        r_spearman = float("nan")

    summary.append({
        "method": method,
        "pearson_r": r_pearson,
        "spearman_r": r_spearman,
        "n_points": len(sub)
    })

    plt.figure(figsize=(6, 5))
    plt.scatter(sub["complex_corr"], sub["mean_vertex_error"], color="steelblue", s=60)
    plt.title(f"{method}\nPearson r = {r_pearson:.3f}, Spearman r = {r_spearman:.3f}")
    plt.xlabel("Complex correlation")
    plt.ylabel("Mean vertex error")
    plt.grid(True)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{method}_geom_vs_complex.png")
    plt.savefig(out_path)
    plt.close()

# === (Opzionale) salva riepilogo CSV ===
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "correlation_summary.csv"), index=False)

print(f"✅ Saved plots and summary to: {OUTPUT_DIR}")
