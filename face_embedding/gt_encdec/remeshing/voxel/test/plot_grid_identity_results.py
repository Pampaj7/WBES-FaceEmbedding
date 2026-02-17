#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

GRID_CSV = Path("grid_identity_results.csv")
CHAMFER_CSV = Path("chamfer_identity_results.csv")

OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

sns.set(style="whitegrid", font_scale=1.2)

# -------------------------------------------------
# UTILS
# -------------------------------------------------

def compute_ratio(df):
    intra = df[
        (df.subject_A == df.subject_B) &
        (df.variant_A != "original") &
        (df.variant_B == "original")
    ][["subject_A", "variant_A", "distance"]]

    intra = intra.rename(columns={"distance": "intra_dist"})

    merged = df.merge(
        intra,
        on=["subject_A", "variant_A"],
        how="inner"   
    )

    merged["ratio"] = merged["distance"] / merged["intra_dist"]
    return merged



def make_kde(df, title, outname):
    intra = df[df.subject_A == df.subject_B]
    inter = df[df.subject_A != df.subject_B]

    plt.figure(figsize=(7, 5))
    sns.kdeplot(intra.distance, label="Intra-subject", fill=True)
    sns.kdeplot(inter.distance, label="Inter-subject", fill=True)
    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / outname, dpi=200)
    plt.close()


def make_ratio_plot(df, title, outname):
    plt.figure(figsize=(6, 5))
    sns.histplot(df.ratio.dropna(), bins=40, kde=True)
    plt.axvline(1.0, color="red", linestyle="--", label="Chance")
    plt.xlabel("Inter / Intra ratio")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / outname, dpi=200)
    plt.close()


def make_boxplot(df, title, outname, ylabel):
    df = df[df.subject_A != df.subject_B].copy()
    df["variant_pair"] = df.variant_A + " vs " + df.variant_B

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="variant_pair", y="distance")
    plt.xticks(rotation=30)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / outname, dpi=200)
    plt.close()


# -------------------------------------------------
# GRID
# -------------------------------------------------

df_grid = pd.read_csv(GRID_CSV)

print("\n[GRID]")
print("INTRA mean:", df_grid[df_grid.subject_A == df_grid.subject_B].distance.mean())
print("INTER mean:", df_grid[df_grid.subject_A != df_grid.subject_B].distance.mean())

make_kde(
    df_grid,
    title="Grid — Intra vs Inter",
    outname="kde_grid_intra_vs_inter.png"
)

make_boxplot(
    df_grid,
    title="Grid — Cross-topology (inter-subject)",
    outname="boxplot_grid_cross_topology.png",
    ylabel="Grid Distance"
)

if "ratio" not in df_grid.columns:
    df_grid = compute_ratio(df_grid)

make_ratio_plot(
    df_grid,
    title="Grid — Ratio Distribution",
    outname="ratio_grid_distribution.png"
)

# -------------------------------------------------
# CHAMFER
# -------------------------------------------------

df_ch = pd.read_csv(CHAMFER_CSV)

print("\n[CHAMFER]")
print("INTRA mean:", df_ch[df_ch.subject_A == df_ch.subject_B].distance.mean())
print("INTER mean:", df_ch[df_ch.subject_A != df_ch.subject_B].distance.mean())

make_kde(
    df_ch,
    title="Chamfer — Intra vs Inter",
    outname="kde_chamfer_intra_vs_inter.png"
)

make_boxplot(
    df_ch,
    title="Chamfer — Cross-topology (inter-subject)",
    outname="boxplot_chamfer_cross_topology.png",
    ylabel="Chamfer Distance"
)

df_ch = compute_ratio(df_ch)

make_ratio_plot(
    df_ch,
    title="Chamfer — Ratio Distribution",
    outname="ratio_chamfer_distribution.png"
)

# -------------------------------------------------
# DONE
# -------------------------------------------------

print("\n✅ ALL PLOTS GENERATED")
print(f"📁 Output dir: {OUT_DIR.resolve()}")
