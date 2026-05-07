import glob, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE = "results"                       # root folder with sub‑dirs
summary_files = glob.glob(os.path.join(BASE, "**/*-wbes_summary.csv"), recursive=True)

if not summary_files:
    raise FileNotFoundError("No '*-wbes_summary.csv' found under 'results/'")

all_rows = []
for fp in summary_files:
    method = os.path.basename(fp).split("-")[0]    # e.g. 'cropped_faceverse'
    df = pd.read_csv(fp)
    df_inter = df[df["type"] == "inter"]
    all_rows.append(df_inter.assign(method=method))

df_plot = pd.concat(all_rows, ignore_index=True)

plt.figure(figsize=(8,5))
sns.lineplot(data=df_plot, x="F", y="wbse", hue="method", marker="o")
plt.title("WBES (inter‑subject) vs number of frames per reconstruction")
plt.xlabel("# Frames per reconstruction (F)")
plt.ylabel("WBES  (Cohen's d)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("wbes_inter_curve.png", dpi=300)
plt.show()
print("Saved wbes_inter_curve.png")
