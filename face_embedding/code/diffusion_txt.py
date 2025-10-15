import torch
import numpy as np
import diffusion_net
from diffusion_net.geometry import normalize_positions
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp

# === CONFIG ===
TXT_PATH = "/Users/pampaj/PycharmProjects/data/datasets/3DDFAV2_23470_neutral/id0000_000_23470.txt"
OUTPUT_PATH = "embedding_pointcloud.npy"
K_EIG = 64
C_WIDTH = 128
K_NN = 12  # numero vicini per grafo

# === LOAD POINT CLOUD ===
verts = np.loadtxt(TXT_PATH)
verts = torch.tensor(verts, dtype=torch.float32)
print("Loaded vertices:", verts.shape)

# === NORMALIZE ===
verts = normalize_positions(verts)

# === COSTRUISCI GRAFO KNN ===
A = kneighbors_graph(verts.numpy(), K_NN, mode='distance', include_self=False)
A = 0.5 * (A + A.T)  # simmetrizza
W = sp.csr_matrix(np.exp(-A.data**2 / (np.mean(A.data)**2)))  # pesi gaussiani
W = sp.csr_matrix((W.data, A.indices, A.indptr), shape=A.shape)

# === LAPLACIANO NON NORMALIZZATO ===
D = sp.diags(W.sum(axis=1).A1)
L = D - W
mass = D.diagonal()  # massa = grado dei vertici

# === CONVERTI IN TENSORI TORCH ===
L = torch.tensor(L.toarray(), dtype=torch.float32)
mass = torch.tensor(mass, dtype=torch.float32)
evals, evecs = torch.linalg.eigh(L)
evals = evals[:K_EIG]
evecs = evecs[:, :K_EIG]
gradX = torch.zeros_like(evecs)
gradY = torch.zeros_like(evecs)

# === ADD BATCH DIM ===
mass = mass.unsqueeze(0)
L = L.unsqueeze(0)
evals = evals.unsqueeze(0)
evecs = evecs.unsqueeze(0)
gradX = gradX.unsqueeze(0)
gradY = gradY.unsqueeze(0)

# === MODEL ===
model = diffusion_net.layers.DiffusionNet(
    C_in=3,
    C_out=C_WIDTH,
    C_width=C_WIDTH,
    N_block=4,
    outputs_at="vertices"
)

# === FEATURES ===
features = verts.unsqueeze(0).permute(0, 2, 1)  # (1, 3, N)

# === FORWARD ===
with torch.no_grad():
    embedding = model(
        features,
        mass=mass,
        L=L,
        evals=evals,
        evecs=evecs,
        gradX=gradX,
        gradY=gradY,
        faces=None
    )

print("✅ Embedding computed!")
print("Shape:", embedding.shape)

# === SAVE ===
np.save(OUTPUT_PATH, embedding.squeeze(0).cpu().numpy())
print(f"💾 Saved to {OUTPUT_PATH}")
