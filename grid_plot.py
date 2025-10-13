import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

VAR_EPS = 1e-6
F_SHOW = [1, 3, 5, 15]
COLOR_W = "#f6a600"
COLOR_B = "#0080c9"
RESULTS_DIR = "results"


def safe_kde(arr, color, label, ax):
    if arr.size == 0:
        return
    if np.std(arr) < VAR_EPS:
        ax.axvline(arr.mean(), color=color, lw=2, label=label)
    else:
        sns.kdeplot(arr, ax=ax, color=color, bw_adjust=0.6, label=label)


# --- Load data ---
density_data = {}
for method_dir in os.listdir(RESULTS_DIR):
    inter_csv = os.path.join(RESULTS_DIR, method_dir,
                             f"{method_dir}-wbes_inter_F.csv")
    if not os.path.exists(inter_csv):
        continue

    df = pd.read_csv(inter_csv)
    density_data[method_dir] = {}

    for _, row in df.iterrows():
        F = int(row["F"])
        if F not in F_SHOW:
            continue

        wbse = row["wbse"]
        prefix = os.path.join(RESULTS_DIR, method_dir, f"wbse_inter_F{F}")
        within_path = prefix + "_within.npy"
        between_path = prefix + "_between.npy"

        within = np.load(within_path) if os.path.exists(
            within_path) else np.array([])
        between = np.load(between_path) if os.path.exists(
            between_path) else np.array([])
        density_data[method_dir][F] = (within, between, wbse)

# --- Plot for each method ---
for method, frames in density_data.items():
    available_frames = [F for F in F_SHOW if F in frames]
    if len(available_frames) == 0:
        continue

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, F in enumerate(available_frames):
        ax = axes[idx]
        within, between, wbse = frames[F]
        safe_kde(within, COLOR_W, "Within-subject" if idx == 0 else None, ax)
        safe_kde(between, COLOR_B, "Between-subject" if idx == 0 else None, ax)

        if within.size and between.size:
            xmax = max(within.max(), between.max())
            xmin = min(within.min(), between.min())
            ax.set_xlim(xmin * 0.95, xmax * 1.05)

        ax.set_title(f"{method}, F={F} — WBES = {wbse:.2f}", fontsize=10)
        ax.set_xlabel("Distance")
        ax.set_ylabel("Density")

    for j in range(len(available_frames), 4):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.02))
    plt.subplots_adjust(hspace=0.6, wspace=0.3, bottom=0.12, top=0.9)
    fig.suptitle(f"WBES Distributions — {method}", fontsize=14)

    out_path = f"grid/{method}_wbes_kde_grid_final.png"
    fig.subplots_adjust(top=0.88)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {out_path}")
