"""
wbes_pipeline.py
================
• Compute WBES (inter & intra) for each reconstruction method
• Save CSVs under results/<method>/
• Plot a 2×3 density grid (FaceVerse vs Smirk, F = 1 | 15 | 16)
  with variance‑zero spikes handled gracefully
"""

import os, math, itertools
from glob import glob
from collections import defaultdict
from random import sample, seed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

seed(42)

# ------------------------------------------------------------------
STUDY_ROOT = "/home/pampalonil/data"          # folder containing cropped_* dirs
METHOD_DIRS = {
    "FaceVerse": "cropped_faceverse",
    "Smirk":     "cropped_smirk"
}
GROUP_SIZES = [1, 3, 6, 9, 12, 15]     # 16 = 50/3 disjoint blocks
F_SHOW      = GROUP_SIZES        # what to plot
N_REPS      = 3
OUT_ROOT    = "results"

COLOR_W = "#f6a600"
COLOR_B = "#0080c9"
VAR_EPS  = 1e-6
# ------------------------------------------------------------------


# ---------- helpers ----------------------------------------------------------
def cohens_d(w, b):
    """Cohen's d with pooled σ (w = within, b = between)."""
    pooled = np.sqrt(((len(w)-1)*w.var() + (len(b)-1)*b.var()) /
                     (len(w)+len(b)-2))
    return (b.mean() - w.mean()) / (pooled + 1e-12)


def load_meshes(folder):
    """Return {subject: [mesh, …]}."""
    d = defaultdict(list)
    for fp in glob(os.path.join(folder, "*.txt")):
        sid = os.path.basename(fp).split("_")[0]
        d[sid].append(np.loadtxt(fp))
    return d


def reps_disjoint(meshes, F, n_rep):
    """n_rep means built from disjoint F‑frame subsets when possible."""
    tot = len(meshes)
    if tot < F:
        return []
    if F * n_rep <= tot:
        idx = sample(range(tot), F * n_rep)
        return [np.mean(np.stack([meshes[i] for i in idx[k*F:(k+1)*F]]), 0)
                for k in range(n_rep)]
    # fallback (may overlap)
    return [np.mean(np.stack(sample(meshes, F)), 0) for _ in range(n_rep)]


def safe_kde(arr, color, label, ax):
    """Draw KDE or spike if var ~ 0."""
    if arr.size == 0:
        return
    if np.std(arr) < VAR_EPS:
        ax.axvline(arr.mean(), color=color, lw=2, label=label)
    else:
        sns.kdeplot(arr, ax=ax, color=color, bw_adjust=0.6, label=label)
# ------------------------------------------------------------------


# ---------- main -------------------------------------------------------------
density_data = {}          # {method: {F: (within_arr, between_arr, wbes)}}

for pretty, subdir in METHOD_DIRS.items():
    print(f"\n=== {pretty} ===")
    meshes_per_subj = load_meshes(os.path.join(STUDY_ROOT, subdir))
    out_dir = os.path.join(OUT_ROOT, subdir)
    os.makedirs(out_dir, exist_ok=True)

    reps_byF = {F: {sid: reps_disjoint(m, F, N_REPS)
                    for sid, m in meshes_per_subj.items()
                    if len(reps_disjoint(m, F, N_REPS)) == N_REPS}
                for F in GROUP_SIZES}

    rows_inter, rows_intra = [], []
    density_data[pretty] = {}

    # -------- inter (same Frame different subject) -------------------------------------------------
    for F in GROUP_SIZES:
        within, between = [], []
        subs = list(reps_byF[F])

        for sid in subs:                                   # within
            for r1, r2 in itertools.combinations(reps_byF[F][sid], 2):
                within.append(np.linalg.norm(r1 - r2))

        for i, sa in enumerate(subs):                      # between
            for sb in subs[i+1:]:
                for r1 in reps_byF[F][sa]:
                    for r2 in reps_byF[F][sb]:
                        between.append(np.linalg.norm(r1 - r2))

        if len(within) and len(between):
            within = np.array(within)
            between = np.array(between)
            rows_inter.append(dict(
                F=F,
                wbse=cohens_d(within, between),
                within_mean=within.mean(),
                between_mean=between.mean(),
                n_subjects=len(subs)
            ))

        # store densities for plotting if F in F_SHOW
        if F in F_SHOW and len(within) and len(between):
            density_data[pretty][F] = (within, between, cohens_d(within, between))

    # -------- intra (F1 vs F2 different frame same subject) ----------------------------------------------
    for F1, F2 in itertools.combinations(GROUP_SIZES, 2):
        subs = set(reps_byF[F1]) & set(reps_byF[F2])
        within, between = [], []

        for sid in subs:                                   # same subject
            d = np.linalg.norm(np.mean(reps_byF[F1][sid], 0) -
                               np.mean(reps_byF[F2][sid], 0))
            within.append(d)

        for sa, sb in itertools.combinations(subs, 2):     # cross subjects
            d = np.linalg.norm(np.mean(reps_byF[F1][sa], 0) -
                               np.mean(reps_byF[F2][sb], 0))
            between.append(d)

        if within and between:
            rows_intra.append(dict(
                F1=F1, F2=F2,
                wbse=cohens_d(np.array(within), np.array(between)),
                within_mean=np.mean(within),
                between_mean=np.mean(between),
                n_subjects=len(subs)
            ))

    # -------- save CSVs ------------------------------------------------------
    pd.DataFrame(rows_inter).to_csv(os.path.join(out_dir,
        f"{subdir}-wbes_inter_F.csv"), index=False)
    pd.DataFrame(rows_intra).to_csv(os.path.join(out_dir,
        f"{subdir}-wbes_intra_F1vsF2.csv"), index=False)
    pd.concat([
        pd.DataFrame(rows_inter).assign(type="inter", compare="F"),
        pd.DataFrame(rows_intra ).assign(type="intra", compare="F1_vs_F2")
    ]).to_csv(os.path.join(out_dir,
        f"{subdir}-wbes_summary.csv"), index=False)

    print(f"CSV saved to {out_dir}")

# ---------- density grid plot -----------------------------------------------
rows, cols = len(METHOD_DIRS), len(F_SHOW)
fig, axes = plt.subplots(rows, cols,
                         figsize=(4.4*cols, 3.4*rows),
                         sharey=False)

for r, (pretty, _) in enumerate(METHOD_DIRS.items()):
    for c, F in enumerate(F_SHOW):
        ax = axes[r, c] if rows > 1 else axes[c]
        within, between, wbes = density_data[pretty][F]

        safe_kde(within,  COLOR_W,
                 "Within‑subject"  if (r==0 and c==0) else None, ax)
        safe_kde(between, COLOR_B,
                 "Between‑subject" if (r==0 and c==0) else None, ax)

        xmax = max(within.max(), between.max())
        ax.set_xlim((0 if xmax > 1 else -0.01), xmax * 1.1)
        ax.set_xlabel("Distance")
        if c == 0:
            ax.set_ylabel("Density")
        ax.set_title(f"{pretty} (F = {F})\nWBES = {wbes:.2f}", fontsize=11)

# shared legend
h, l = axes[0,0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False)

plt.tight_layout(pad=2, rect=[0,0,1,0.93])
plt.savefig("wbes_density_grid.png", dpi=300)
plt.show()
print("Saved plot to wbes_density_grid.png")
