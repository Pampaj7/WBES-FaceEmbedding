import os
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIG === 
#BASE_PATH = "results"
#KEYWORD = "-wbes_inter_F.csv"

BASE_PATH = "results_landmarks"
KEYWORD = "-wbes_landmark_inter_F.csv"

# Mappa dei nomi leggibili per il plot
METHOD_LABELS = {
    "3DDFAV2_23470_neutral": "3DDFAv2",
    "3DDFAV3_neutral": "3DDFAv3",
    "Deep3DFace_23470_neutral": "Deep3DFace",
    "INORig_23470_neutral": "INORig",
    "SynergyNet_neutral": "SynergyNet",
    "3DI_neutral": "3DI",
    "Smirk_cropped_neutral": "Smirk",
    "Faceverse_cropped_neutral": "FaceVerse"
}

def remap_f(F, method):
    if method == "INORig_23470_neutral":
        return F * 5
    if method == "3DI_neutral":
        return F * 10
    return F


# === INIT PLOT ===
plt.figure(figsize=(10, 6))

for folder in os.listdir(BASE_PATH):
    full_folder = os.path.join(BASE_PATH, folder)
    if not os.path.isdir(full_folder):
        continue

    # Trova il file giusto
    for file in os.listdir(full_folder):
        if KEYWORD in file:
            csv_path = os.path.join(full_folder, file)
            df = pd.read_csv(csv_path)

            if "F" in df.columns and "wbse" in df.columns:
                df = df.sort_values("F")
                label = METHOD_LABELS.get(folder, folder)
                marker = "o" if len(df) > 1 else "x"
                remapped_F = [remap_f(f, folder) for f in df["F"]]
                plt.plot(remapped_F, df["wbse"], marker=marker, label=label)
            break

# === PLOT SETTINGS ===
#plt.title("WBES vs F for different methods on meshes")
plt.title("WBES vs F for different methods on landmarks")
plt.xlabel("F (media size)")
plt.ylabel("WBES (Cohen's d)")
plt.grid(True)
plt.legend()
plt.tight_layout()
#plt.savefig("wbse_vs_f_lineplot.png", dpi=300)
plt.savefig("wbse_vs_f_lineplot_land.png", dpi=300)
plt.show()
