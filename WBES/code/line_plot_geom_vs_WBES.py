import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# === CONFIG ===
MODE = "landmark"  # "landmark" oppure "mesh"

if MODE == "landmark":
    WBES_PATH = "results_landmarks"
    GEOM_PATH = "results_landmarks"
    WBES_KEY = "-wbes_landmark_inter_F.csv"
    GEOM_KEY = "-geom_error_vs_F_landmark.csv"
    output_file = "wbes_vs_geom_landmark.png"
else:
    WBES_PATH = "results"
    GEOM_PATH = "results"
    WBES_KEY = "-wbes_inter_F.csv"
    GEOM_KEY = "-geom_error_vs_F.csv"
    output_file = "wbes_vs_geom_mesh.png"

METHOD_LABELS = {
    "3DDFAV2_23470_neutral": "3DDFAv2",
    "3DDFAV3_neutral": "3DDFAv3",
    "Deep3DFace_23470_neutral": "Deep3DFace",
    "INORig_23470_neutral": "INORig",
    "SynergyNet_neutral": "SynergyNet",
    "3DI_neutral": "3DI",
}

COLORS = {
    "3DDFAV2_23470_neutral": "#1f77b4",
    "3DDFAV3_neutral": "#ff7f0e",
    "Deep3DFace_23470_neutral": "#2ca02c",
    "INORig_23470_neutral": "#9467bd",
    "SynergyNet_neutral": "#d62728",
    "3DI_neutral": "#8c564b",
}

def remap_f(F, method):
    if method == "INORig_23470_neutral":
        return F * 5
    if method == "3DI_neutral":
        return F * 10
    return F

# === PLOT ===
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()
GEOM_YLIM = (0.0015, 0.0033)  # oppure 0.0024–0.0033 per mesh
WBES_YLIM = (0.00, 2.2)  # oppure 0.0024–0.0033 per mesh
ax2.set_ylim(GEOM_YLIM)
ax1.set_ylim(WBES_YLIM)
for folder in os.listdir(WBES_PATH):
    full_folder = os.path.join(WBES_PATH, folder)
    if not os.path.isdir(full_folder) or folder not in METHOD_LABELS:
        continue

    method_label = METHOD_LABELS[folder]
    color = COLORS.get(folder, None)

    # --- WBES ---
    wbes_file = next((f for f in os.listdir(full_folder) if WBES_KEY in f), None)
    if wbes_file:
        df_wbes = pd.read_csv(os.path.join(full_folder, wbes_file))
        if "F" in df_wbes.columns and "wbse" in df_wbes.columns:
            df_wbes = df_wbes.sort_values("F")
            remapped_F = [remap_f(f, folder) for f in df_wbes["F"]]
            ax1.plot(remapped_F, df_wbes["wbse"], marker="o", linestyle="-",
                     label=method_label, color=color)

    # --- Geometric Error ---
    geom_folder = os.path.join(GEOM_PATH, folder)
    geom_file = next((f for f in os.listdir(geom_folder) if GEOM_KEY in f), None)
    if geom_file:
        df_geom = pd.read_csv(os.path.join(geom_folder, geom_file))
        if "F" in df_geom.columns:
            df_geom = df_geom.sort_values("F")
            remapped_F = [remap_f(f, folder) for f in df_geom["F"]]
            y_col = [col for col in df_geom.columns if "error" in col][0]  # automatico
            ax2.plot(remapped_F, df_geom[y_col], marker="x", linestyle="--", color=color)

# === STYLING ===
ax1.set_xlabel("F (media size)")
ax1.set_ylabel("WBES (Cohen's d)")
ax2.set_ylabel("Geometric Error", color="gray")

ax1.grid(True)
ax1.set_ylim(bottom=0)

# Legenda solo per WBES
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(
    handles, labels,
    loc='upper left',
    bbox_to_anchor=(-0.15, 1.2),
    fontsize=8,
    frameon=False,
    title="Metodo", title_fontsize=9
)


style_legend = [
    Line2D([0], [0], color="black", linestyle="-", label="WBES"),
    Line2D([0], [0], color="black", linestyle="--", label="Geom. Error")
]

ax2.legend(
    style_legend,
    ["WBES", "Geom. Error"],
    loc='upper right',
    bbox_to_anchor=(1, 1.15),
    frameon=False,
    fontsize=8,
    title="Metric", title_fontsize=9
)


plt.title(f"WBES vs Geometric Error across F ({MODE})")
plt.savefig(output_file, dpi=300)
plt.show()
