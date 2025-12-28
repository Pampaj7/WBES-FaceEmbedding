#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ============================================================
# CONFIG
# ============================================================
RESULTS_DIR = "2stage"
LAT_GLOBAL_PATH = os.path.join(RESULTS_DIR, "latents_global.npy")
LAT_MEAN_PATH   = os.path.join(RESULTS_DIR, "latents_mean.npy")
META_PATH       = os.path.join(RESULTS_DIR, "meta.npy")

OUT_DIR = "2stage"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_SUBJECTS = 40   # filtra soggetti id0000 ... id0039


# ============================================================
# LOAD
# ============================================================
print("Loading latents...")
Zg = np.load(LAT_GLOBAL_PATH)   # shape [N, 256]
Zm = np.load(LAT_MEAN_PATH)     # shape [N, 256]
meta = np.load(META_PATH, allow_pickle=True)

subjects = np.array([m["subject"] for m in meta])
variants = np.array([m["variant"] for m in meta])

print(f"Loaded {len(Zg)} embeddings (global + mean)")


# ============================================================
# FILTER only first MAX_SUBJECTS subjects
# ============================================================
keep_mask = np.zeros_like(subjects, dtype=bool)
for sid in range(MAX_SUBJECTS):
    sid_str = f"id{sid:04d}_GTready"
    keep_mask |= (subjects == sid_str)

Zg = Zg[keep_mask]
Zm = Zm[keep_mask]
subjects = subjects[keep_mask]
variants = variants[keep_mask]

print(f"Using {len(Zg)} embeddings from subjects 0..{MAX_SUBJECTS-1}")


# ============================================================
# PCA + t-SNE FUNCTIONS
# ============================================================
def run_pca(Z):
    pca = PCA(n_components=2)
    return pca.fit_transform(Z)

def run_tsne(Z):
    tsne = TSNE(n_components=2, learning_rate="auto", init="pca", perplexity=10)
    return tsne.fit_transform(Z)


# ============================================================
# PLOT FUNCTION
# ============================================================
def plot_embedding(X, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    unique = np.unique(labels)

    for u in unique:
        mask = labels == u
        ax.scatter(
            X[mask, 0], X[mask, 1],
            s=50, alpha=0.8,
            label=u,
        )

    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(loc="best", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


# ============================================================
# RUN PCA / TSNE
# ============================================================
print("Running PCA...")
Zg_pca = run_pca(Zg)
Zm_pca = run_pca(Zm)

print("Running t-SNE...")
Zg_tsne = run_tsne(Zg)
Zm_tsne = run_tsne(Zm)


# ============================================================
# SAVE PLOTS
# ============================================================

# --- GLOBAL ---
plot_embedding(Zg_pca, subjects,
               "PCA (GLOBAL) – by SUBJECT",
               os.path.join(OUT_DIR, "pca_global_by_subject.png"))

plot_embedding(Zg_pca, variants,
               "PCA (GLOBAL) – by VARIANT",
               os.path.join(OUT_DIR, "pca_global_by_variant.png"))

plot_embedding(Zg_tsne, subjects,
               "tSNE (GLOBAL) – by SUBJECT",
               os.path.join(OUT_DIR, "tsne_global_by_subject.png"))

plot_embedding(Zg_tsne, variants,
               "tSNE (GLOBAL) – by VARIANT",
               os.path.join(OUT_DIR, "tsne_global_by_variant.png"))


# --- MEAN PER-VERTEX ---
plot_embedding(Zm_pca, subjects,
               "PCA (MEAN Z_per) – by SUBJECT",
               os.path.join(OUT_DIR, "pca_mean_by_subject.png"))

plot_embedding(Zm_pca, variants,
               "PCA (MEAN Z_per) – by VARIANT",
               os.path.join(OUT_DIR, "pca_mean_by_variant.png"))

plot_embedding(Zm_tsne, subjects,
               "tSNE (MEAN Z_per) – by SUBJECT",
               os.path.join(OUT_DIR, "tsne_mean_by_subject.png"))

plot_embedding(Zm_tsne, variants,
               "tSNE (MEAN Z_per) – by VARIANT",
               os.path.join(OUT_DIR, "tsne_mean_by_variant.png"))


print("\n🎉 Analysis done!")
print(f"Plots saved in: {OUT_DIR}")
