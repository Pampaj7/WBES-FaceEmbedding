import glob, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE = "results"
summary_files = glob.glob(os.path.join(BASE, "**/*-wbes_summary.csv"), recursive=True)

if not summary_files:
    raise FileNotFoundError("No '*-wbes_summary.csv' found under 'results/'")

n_methods = len(summary_files)
fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 4), sharey=True)

for ax, fp in zip(axes, summary_files):
    method = os.path.basename(fp).split("-")[0]
    df      = pd.read_csv(fp)
    intra   = df[df["type"] == "intra"].copy()
    intra["pair"] = intra.apply(lambda r: f"{int(r.F1)}‑{int(r.F2)}", axis=1)

    sns.barplot(data=intra, x="pair", y="wbse", ax=ax, palette="viridis")
    ax.set_title(f"{method}  (F1 vs F2)")
    ax.set_xlabel("Frame groups compared")
    ax.set_ylabel("WBES" if ax is axes[0] else "")

    # annotate
    for p in ax.patches:
        ax.text(p.get_x()+p.get_width()/2., p.get_height()+0.05,
                f"{p.get_height():.2f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("wbes_intra_bars.png", dpi=300)
plt.show()
print("Saved wbes_intra_bars.png")
