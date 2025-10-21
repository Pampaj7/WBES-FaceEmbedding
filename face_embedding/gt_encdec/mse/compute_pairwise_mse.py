import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# === CONFIG ===
# <-- metti qui la tua cartella finale
DATA_DIR = "../../../datasets/GT_ready"
OUT_DIR = "./results_pairwise/"
os.makedirs(OUT_DIR, exist_ok=True)

# === 1️⃣ Load vertices ===
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".txt")])
meshes = [np.loadtxt(os.path.join(DATA_DIR, f)).flatten()
          for f in tqdm(files, desc="Loading meshes")]
X = np.stack(meshes)
print(f"Loaded {len(files)} meshes with {X.shape[1]} features each")

# === 2️⃣ Compute pairwise MSE ===
n = len(X)
D = np.zeros((n, n))

for i in tqdm(range(n), desc="Computing pairwise MSE"):
    for j in range(i + 1, n):
        mse = np.mean((X[i] - X[j]) ** 2)
        D[i, j] = D[j, i] = mse

# === 3️⃣ Save results ===
df = pd.DataFrame(D, index=files, columns=files)
df.to_csv(os.path.join(OUT_DIR, "pairwise_gt.csv"))
print("✅ Saved pairwise distance matrix to results_pairwise/pairwise_gt.csv")

# === 4️⃣ Visualization ===
plt.figure(figsize=(6, 5))
plt.imshow(D, cmap="viridis")
plt.colorbar(label="MSE distance")
plt.title("Pairwise MSE between GT meshes")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pairwise_heatmap.png"))
plt.close()

plt.figure(figsize=(5, 4))
plt.hist(D[np.triu_indices(n, 1)], bins=60, color="gray")
plt.xlabel("MSE distance")
plt.ylabel("Frequency")
plt.title("Distribution of pairwise MSE distances")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pairwise_hist.png"))
plt.close()

print("✅ Saved heatmap and histogram in results_pairwise/")
