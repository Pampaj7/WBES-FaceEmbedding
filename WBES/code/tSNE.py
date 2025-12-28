import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from utils import WBES_helper as wh

# === CONFIG ===
STUDY_ROOT = "/Users/pampaj/PycharmProjects/data"
METHOD_DIR = "Deep3DFace_23470_neutral"
F_VALUES = [1, 3, 15]
N_REPS = 3
SCALE = 1
BANNED_SUBJECTS = {"id0099"}

mesh_dir = os.path.join(STUDY_ROOT, METHOD_DIR)
meshes_by_subject = wh.load_meshes_txt(mesh_dir)

# === Calcola t-SNE per ogni F ===
embeddings_by_F = {}
subject_ids_by_F = {}

for F in F_VALUES:
    mean_meshes = []
    subject_ids = []

    for sid, meshes in meshes_by_subject.items():
        if sid in BANNED_SUBJECTS:
            continue
        reps = wh.reps_disjoint(meshes, F * SCALE, N_REPS)
        if len(reps) == N_REPS:
            mean = np.mean(np.stack(reps), axis=0)
            mean_meshes.append(mean.flatten())
            subject_ids.append(sid)

    if not mean_meshes:
        continue

    X = np.stack(mean_meshes)
    tsne = TSNE(n_components=2, init='pca', perplexity=30, random_state=42)
    embedding = tsne.fit_transform(X)
    embeddings_by_F[F] = embedding
    subject_ids_by_F[F] = subject_ids

# === Limiti globali ===
all_x = np.concatenate([emb[:, 0] for emb in embeddings_by_F.values()])
all_y = np.concatenate([emb[:, 1] for emb in embeddings_by_F.values()])
xlim = (all_x.min(), all_x.max())
ylim = (all_y.min(), all_y.max())

# === Plot ===
fig, axes = plt.subplots(1, len(F_VALUES), figsize=(15, 5))

for ax, F in zip(axes, F_VALUES):
    embedding = embeddings_by_F[F]
    sids = subject_ids_by_F[F]

    ax.scatter(embedding[:, 0], embedding[:, 1], s=35, alpha=0.8, color='teal')
    for i, sid in enumerate(sids):
        ax.text(embedding[i, 0], embedding[i, 1], sid, fontsize=6)
    ax.set_title(f"F = {F}")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True)

plt.suptitle(f"t-SNE of Mean Meshes – {METHOD_DIR}", fontsize=14)
plt.tight_layout()
plt.savefig("tSNE_mean_meshes.png")