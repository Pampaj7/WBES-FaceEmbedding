#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
RESULTS_DIR = "2stage"

LATENTS_PATH = os.path.join(RESULTS_DIR, "latents_global.npy")
META_PATH    = os.path.join(RESULTS_DIR, "meta.npy")

OUT_DIR = "2stage"
os.makedirs(OUT_DIR, exist_ok=True)

# usa solo questi soggetti (None = tutti)
MAX_SUBJECTS = 200

# per l'inter-subject: controlliamo la topologia
INTER_VARIANT = "original"

RNG_SEED = 0
N_INTER_SAMPLES = 200   # per ogni punto

# ============================================================
# HELPERS
# ============================================================
def l2(a, b):
    d = a - b
    return float(np.sqrt(np.dot(d, d)))

def subject_id(s):
    # "id0003_GTready" -> 3
    return int(s[2:6])

# ============================================================
# LOAD
# ============================================================
print("Loading data...")
Z = np.load(LATENTS_PATH)                 # [N, D]
meta = np.load(META_PATH, allow_pickle=True)

subjects = np.array([m["subject"] for m in meta])
variants = np.array([m["variant"] for m in meta])

print("Total embeddings:", len(Z))

# ============================================================
# FILTER SUBJECTS (optional)
# ============================================================
if MAX_SUBJECTS is not None:
    mask = np.array([subject_id(s) < MAX_SUBJECTS for s in subjects])
    Z = Z[mask]
    subjects = subjects[mask]
    variants = variants[mask]

print("Using embeddings:", len(Z))

# ============================================================
# GROUP INDICES
# ============================================================
by_subject = {}
for i, s in enumerate(subjects):
    by_subject.setdefault(s, []).append(i)

by_subject_variant = {}
for i, (s, v) in enumerate(zip(subjects, variants)):
    by_subject_variant.setdefault((s, v), []).append(i)

# ============================================================
# COMPUTE DISTANCES
# ============================================================
rng = np.random.default_rng(RNG_SEED)

xs_intra = []
ys_inter = []

for i in range(len(Z)):
    zi = Z[i]
    s  = subjects[i]

    # -------------------------
    # INTRA-SUBJECT
    # -------------------------
    intra_idxs = [j for j in by_subject[s] if j != i]
    if len(intra_idxs) == 0:
        continue

    intra_dists = [l2(zi, Z[j]) for j in intra_idxs]
    intra_mean = float(np.mean(intra_dists))

    # -------------------------
    # INTER-SUBJECT (controlled topology)
    # -------------------------
    if INTER_VARIANT is not None:
        candidates = [
            j for j in range(len(Z))
            if subjects[j] != s and variants[j] == INTER_VARIANT
        ]
    else:
        candidates = [j for j in range(len(Z)) if subjects[j] != s]

    if len(candidates) == 0:
        continue

    sampled = rng.choice(
        candidates,
        size=min(N_INTER_SAMPLES, len(candidates)),
        replace=False
    )

    inter_dists = [l2(zi, Z[j]) for j in sampled]
    inter_mean = float(np.mean(inter_dists))

    xs_intra.append(intra_mean)
    ys_inter.append(inter_mean)

xs_intra = np.array(xs_intra)
ys_inter = np.array(ys_inter)

print("Computed points:", len(xs_intra))

# ============================================================
# PLOT
# ============================================================
plt.figure(figsize=(6.5, 6.5))
plt.scatter(xs_intra, ys_inter, s=18, alpha=0.6)

mn = min(xs_intra.min(), ys_inter.min())
mx = max(xs_intra.max(), ys_inter.max())
plt.plot([mn, mx], [mn, mx], linestyle="--", color="black", linewidth=1)

plt.xlabel("mean intra-subject distance (cross-topology)")
plt.ylabel(f"mean inter-subject distance (variant={INTER_VARIANT})")
plt.title("Latent space: intra vs inter subject distances")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "scatter_intra_vs_inter.png"), dpi=250)
plt.close()

print("Saved plot to:", OUT_DIR)
