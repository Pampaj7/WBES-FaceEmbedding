import pandas as pd
import matplotlib.pyplot as plt
import os

# === CONFIG ===
RESULTS_DIR = "results"
OUTPUT_DIR = "z_plot_wbes_breakdown"
os.makedirs(OUTPUT_DIR, exist_ok=True)

METHODS = [
    "3DDFAV2_23470_neutral",
    "3DDFAV3_neutral",
    "Deep3DFace_23470_neutral",
    "SynergyNet_neutral",
    "3DI_neutral",
    "Faceverse_cropped_neutral",
    "INORig_23470_neutral",
    "Smirk_cropped_neutral"
]
CSV_SUFFIX = "-wbes_inter_F.csv"

# === GENERA E SALVA PLOT ===
for method in METHODS:
    path = os.path.join(RESULTS_DIR, method, method + CSV_SUFFIX)
    if not os.path.exists(path):
        print(f"[!] Missing: {path}")
        continue

    df = pd.read_csv(path)
    F = df["F"]

    plt.figure(figsize=(8, 5))
    plt.plot(F, df["wbse"], label="WBES (Cohen's d)",
             linestyle='-', marker='o')
    plt.plot(F, df["within_mean"], label="Within-subject",
             linestyle='--', marker='x')
    plt.plot(F, df["between_mean"], label="Between-subject",
             linestyle=':', marker='s')

    plt.title(method)
    plt.xlabel("F (number of frames averaged)")
    plt.ylabel("Metric Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"{method}_breakdown.png")
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Saved: {out_path}")
