import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# === Load meshes ===
# Ground-truth mesh
G = np.loadtxt("/Users/pampaj/PycharmProjects/data/GT/GT_BFM/id0000.id.txt")/ 1e6
# Reconstructed mesh (already aligned)
R = np.loadtxt(
    "/Users/pampaj/PycharmProjects/data/3DDFAV3_neutral/id0000_006.txt")

# === Compute per-vertex error ===
errors = np.linalg.norm(G - R, axis=1)

# === Plot heatmap ===
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(G[:, 0], G[:, 1], G[:, 2], c=errors, cmap='viridis', s=2)
cbar = fig.colorbar(sc, ax=ax, shrink=0.5)
cbar.set_label("Per-vertex L2 Error")
ax.view_init(elev=20, azim=-90)
ax.set_title("Geometric error heatmap")
ax.axis('off')
plt.tight_layout()
plt.savefig("heatmap_error.png", dpi=300)
plt.show()
