#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

GRID_CSV = Path("grid_identity_results.csv")
CHAMFER_CSV = Path("chamfer_identity_results.csv")  # opzionale

OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

sns.set(style="whitegrid", font_scale=1.2)

# -------------------------------------------------
# LOAD
# -------------------------------------------------

df = pd.read_csv(GRID_CSV)

intra = df[df.subject_A == df.subject_B]
inter = df[df.subject_A != df.subject_B]

print("INTRA mean:", intra.distance.mean(), "std:", intra.distance.std())
print("INTER mean:", inter.distance.mean(), "std:", inter.distance.std())
print("Ratio median:", df.ratio.median())

# -------------------------------------------------
# 1️⃣ KDE: INTRA vs INTER
# -------------------------------------------------

plt.figure(figsize=(7, 5))
sns.kdeplot(intra.distance, label="Intra-subject", fill=True)
sns.kdeplot(inter.distance, label="Inter-subject", fill=True)
plt.xlabel("Distance")
plt.ylabel("Density")
plt.title("Grid-based Identity Distance")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "kde_intra_vs_inter.png", dpi=200)
plt.close()

# -------------------------------------------------
# 2️⃣ Ratio distribution
# -------------------------------------------------

plt.figure(figsize=(6, 5))
sns.histplot(df.ratio, bins=40, kde=True)
plt.axvline(1.0, color="red", linestyle="--", label="Chance")
plt.xlabel("Inter / Intra ratio")
plt.ylabel("Count")
plt.title("Ratio distribution")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "ratio_distribution.png", dpi=200)
plt.close()

# -------------------------------------------------
# 3️⃣ Cross-topology boxplot
# -------------------------------------------------

if "variant_A" in df.columns and "variant_B" in df.columns:
    df["variant_pair"] = df.variant_A + " vs " + df.variant_B

    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=df[df.subject_A != df.subject_B],
        x="variant_pair",
        y="distance"
    )
    plt.xticks(rotation=30)
    plt.ylabel("Distance")
    plt.title("Grid distance across topology pairs")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "boxplot_cross_topology.png", dpi=200)
    plt.close()

# -------------------------------------------------
# 4️⃣ Chamfer vs Grid (AGGREGATED, SAFE)
# -------------------------------------------------

if CHAMFER_CSV.exists():
    df_ch = pd.read_csv(CHAMFER_CSV)

    def summarize(df):
        intra = df[df.subject_A == df.subject_B]["distance"]
        inter = df[df.subject_A != df.subject_B]["distance"]
        return intra.mean(), inter.mean()

    g_intra, g_inter = summarize(df)
    c_intra, c_inter = summarize(df_ch)

    plt.figure(figsize=(6, 6))

    plt.scatter(c_intra, g_intra, s=120, label="Intra-subject")
    plt.scatter(c_inter, g_inter, s=120, label="Inter-subject")

    plt.plot(
        [min(c_intra, c_inter), max(c_intra, c_inter)],
        [min(g_intra, g_inter), max(g_intra, g_inter)],
        linestyle="--",
        color="gray",
        alpha=0.6,
    )

    plt.xlabel("Chamfer distance (mean)")
    plt.ylabel("Grid-based distance (mean)")
    plt.title("Chamfer vs Grid (aggregated comparison)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "scatter_chamfer_vs_grid_aggregated.png", dpi=200)
    plt.close()


print(f"✅ Plots saved to {OUT_DIR.resolve()}")
