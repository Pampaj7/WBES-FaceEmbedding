import os
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt

# === CONFIG ===
RESULTS_DIR = "results"
SAVE_DIR = "z_correlation_within_geom_perF"
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

# === Load all data ===
all_rows = []

for method in METHODS:
    within_csv = os.path.join(
        RESULTS_DIR, method, f"{method}-wbes_inter_F.csv")
    error_csv = os.path.join(
        RESULTS_DIR, method, f"{method}-geom_error_vs_F.csv")

    if not os.path.exists(within_csv) or not os.path.exists(error_csv):
        print(f"[!] Skipping {method} — missing files")
        continue

    df_within = pd.read_csv(within_csv)
    df_err = pd.read_csv(error_csv)

    df_within.columns = [c.strip().lower() for c in df_within.columns]
    df_err.columns = [c.strip().lower() for c in df_err.columns]

    df_within["f"] = df_within["f"].astype(int)
    df_err["f"] = df_err["f"].astype(int)

    df = pd.merge(df_within, df_err, on="f", how="inner")
    for _, row in df.iterrows():
        orig_f = int(row["f"])
        actual_f = F_OVERRIDES.get(method, {}).get(orig_f, orig_f)

        wmean = row["within_mean"]
        if wmean > 1e3:  # fix scale inconsistency (e.g. SynergyNet)
            wmean /= 1e6

        all_rows.append({
            "method": method,
            "F": actual_f,
            "within_mean": wmean,
            "error": row["mean_vertex_error"]
        })

df_all = pd.DataFrame(all_rows)
print(f"✅ Loaded {len(df_all)} total rows")

# === Plot per F ===
for f_val in sorted(df_all["F"].unique()):
    sub = df_all[df_all["F"] == f_val]
    if len(sub) < 2:
        print(f"[!] F = {f_val}: not enough methods for correlation.")
        continue

    try:
        r_pearson = scipy.stats.pearsonr(sub["within_mean"], sub["error"])[0]
    except Exception as e:
        print(f"[!] Pearson error for F={f_val}: {e}")
        r_pearson = float("nan")

    try:
        r_spearman = scipy.stats.spearmanr(sub["within_mean"], sub["error"])[0]
    except Exception as e:
        print(f"[!] Spearman error for F={f_val}: {e}")
        r_spearman = float("nan")

    plt.figure(figsize=(6, 5))
    plt.scatter(sub["within_mean"], sub["error"], color='royalblue', s=80)

    for _, row in sub.iterrows():
        plt.text(row["within_mean"], row["error"], row["method"], fontsize=7)

    plt.title(
        f"F = {f_val} | Pearson r = {r_pearson:.3f}, Spearman r = {r_spearman:.3f}")
    plt.xlabel("Within-subject mean distance (normalized)")
    plt.ylabel("Mean Vertex L2 Error")
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(SAVE_DIR, f"within_vs_geom_F{int(f_val):02d}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Saved: {out_path}")
