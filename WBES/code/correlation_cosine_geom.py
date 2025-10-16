import os
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt

# === CONFIG ===
COSINE_CSV = "Z_cosine/results/cosine_wbesstyle_disjoint.csv"
RESULTS_DIR = "results"
SAVE_DIR = "z_correlation_geom_cosine_perF"
os.makedirs(SAVE_DIR, exist_ok=True)

METHODS = [
    "3DDFAV2_23470_neutral",
    "3DDFAV3_neutral",
    "Deep3DFace_23470_neutral",
    "SynergyNet_neutral",
    "INORig_23470_neutral",
    "3DI_neutral"
]

# Valori reali di frame mediati (da usare solo per labeling nei plot)
F_OVERRIDES = {
    "3DI_neutral": {1: 10},
    "INORig_23470_neutral": {1: 5, 2: 10, 3: 15}
}

# === Load cosine correlation
df_cosine = pd.read_csv(COSINE_CSV)
df_cosine.columns = [c.strip().lower() for c in df_cosine.columns]
df_cosine = df_cosine.rename(columns={"f": "F"})
df_cosine["F"] = df_cosine["F"].astype(int)

# === Merge with geometric error
all_rows = []

for method in METHODS:
    geom_path = os.path.join(
        RESULTS_DIR, method, f"{method}-geom_error_vs_F.csv")
    if not os.path.exists(geom_path):
        print(f"[!] Missing geom file: {method}")
        continue

    df_geom = pd.read_csv(geom_path)
    df_geom.columns = [c.strip().lower() for c in df_geom.columns]
    df_geom["F"] = df_geom["f"].astype(int)

    df_m = pd.merge(
        df_geom[["F", "mean_vertex_error"]],
        df_cosine[df_cosine["method"] == method][["F", "cosine_corr"]],
        on="F", how="inner"
    )

    for _, row in df_m.iterrows():
        all_rows.append({
            "method": method,
            "F": int(row["F"]),
            "error": row["mean_vertex_error"],
            "cosine_corr": row["cosine_corr"]
        })

df_all = pd.DataFrame(all_rows)
print(f"✅ Loaded {len(df_all)} merged rows")

# === Plot per F (usando override per labeling) ===
all_f_vals = sorted(set(
    F_OVERRIDES.get(m, {}).get(f, f)
    for m, f in zip(df_all["method"], df_all["F"])
))

for f_val_plot in sorted(set(all_f_vals)):
    sub = []

    for _, row in df_all.iterrows():
        method = row["method"]
        orig_f = row["F"]
        actual_f = F_OVERRIDES.get(method, {}).get(orig_f, orig_f)
        if actual_f == f_val_plot:
            sub.append(row)

    if len(sub) < 2:
        print(f"[!] Skipping F={f_val_plot}: not enough methods.")
        continue

    sub_df = pd.DataFrame(sub)

    try:
        r_pearson = scipy.stats.pearsonr(
            sub_df["cosine_corr"], sub_df["error"])[0]
    except Exception:
        r_pearson = float("nan")

    try:
        r_spearman = scipy.stats.spearmanr(
            sub_df["cosine_corr"], sub_df["error"])[0]
    except Exception:
        r_spearman = float("nan")

    # Plot
    plt.figure(figsize=(6, 5))
    plt.scatter(sub_df["cosine_corr"], sub_df["error"],
                color='darkgreen', s=80)

    for _, row in sub_df.iterrows():
        plt.text(row["cosine_corr"], row["error"], row["method"], fontsize=7)

    plt.title(
        f"F = {f_val_plot} | Pearson r = {r_pearson:.3f}, Spearman r = {r_spearman:.3f}")
    plt.xlabel("Cosine correlation")
    plt.ylabel("Mean Vertex L2 Error")
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(SAVE_DIR, f"cosine_vs_geom_F{f_val_plot:02d}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Saved: {out_path}")
