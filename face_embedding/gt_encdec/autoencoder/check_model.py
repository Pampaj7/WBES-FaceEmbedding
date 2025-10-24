import sys
# === Add DiffusionNet path ===
if "/equilibrium/lpampaloni/diffusion-net/src" not in sys.path:
    sys.path.append("/equilibrium/lpampaloni/diffusion-net/src")

import os
import torch
import numpy as np
import igl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp

from diffusion_autoencoder import DiffusionAutoencoder
from dataset_gtready import GTReadyDataset

# === CONFIG ===
BASE_DIR = "./results_diffusionAE"
CHECKPOINT = os.path.join(BASE_DIR, "diffusionAE_epoch20.pth")
DATA_DIR = "../../../datasets/GT_ready/"
OPS_DIR = os.path.join(DATA_DIR, "operators")

os.makedirs(BASE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 1. CARICA MODELLO ===
model = DiffusionAutoencoder(latent_dim=64).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()
print(f"✅ Loaded checkpoint: {CHECKPOINT}")

# === 2. CARICA UN ESEMPIO DAL DATASET ===
dataset = GTReadyDataset(DATA_DIR, ops_dir=OPS_DIR)
sample = dataset[0]
V = sample["verts"].to(device)
mass = sample["mass"].to(device)
L = sample["L"].to(device)
evals = sample["evals"].to(device)
evecs = sample["evecs"].to(device)
gradX, gradY = sample["gradX"], sample["gradY"]

# === 3. RICOSTRUZIONE ===
with torch.no_grad():
    V_rec, z = model(V, mass, L, evals, evecs, gradX, gradY)

V_gt = V.cpu().numpy()
V_rec = V_rec.cpu().numpy()
F = sample["faces"].numpy()

# === 4. SALVA MESH SU DISCO ===
igl.write_triangle_mesh(os.path.join(BASE_DIR, "sample_original.obj"), V_gt, F)
igl.write_triangle_mesh(os.path.join(BASE_DIR, "sample_reconstructed.obj"), V_rec, F)
print("💾 Saved sample_original.obj and sample_reconstructed.obj")

# === 5. CALCOLA METRICHE ===
def chamfer_distance(x, y):
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    dist = torch.sum((x - y)**2, dim=2)
    min_x, _ = torch.min(dist, dim=1)
    min_y, _ = torch.min(dist, dim=0)
    return (min_x.mean() + min_y.mean()).item()

def laplacian_loss(V, L):
    if L.device != V.device:
        L = L.to(V.device)
    LV = torch.sparse.mm(L, V)
    return torch.mean(torch.norm(LV, dim=1))

chamfer = chamfer_distance(V_gt, V_rec)
lap_loss = laplacian_loss(torch.tensor(V_rec, device=device), L).item()
err = np.linalg.norm(V_gt - V_rec, axis=1)
print(f"📊 Chamfer Distance: {chamfer:.6f}")
print(f"📊 Mean Vertex Error: {err.mean():.6f} | Max: {err.max():.6f}")
print(f"📊 Laplacian Smoothness: {lap_loss:.6f}")

np.save(os.path.join(BASE_DIR, "vertex_errors.npy"), err)

# === 6. VISUALIZZAZIONE INTERATTIVA SALVATA IN HTML ===
def mesh_trace(V, F, color=None, name="mesh"):
    return go.Mesh3d(
        x=V[:, 0], y=V[:, 1], z=V[:, 2],
        i=F[:, 0], j=F[:, 1], k=F[:, 2],
        intensity=color if color is not None else V[:, 2],
        colorscale="Viridis", opacity=1.0, showscale=color is not None,
        name=name
    )

fig = sp.make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'surface'}, {'type': 'surface'}]],
    subplot_titles=("Ground Truth", "Reconstructed")
)
fig.add_trace(mesh_trace(V_gt, F, name="GT"), row=1, col=1)
fig.add_trace(mesh_trace(V_rec, F, color=err, name="Reconstruction"), row=1, col=2)
fig.update_layout(scene_aspectmode="data", height=800)
html_path = os.path.join(BASE_DIR, "comparison_GT_vs_RECON.html")
fig.write_html(html_path)
print(f"💾 Saved interactive comparison: {html_path}")

# === 7. DISTRIBUZIONE LATENTE (STATIC PLOT) ===
z_np = z.cpu().numpy()
plt.figure(figsize=(6, 6))
if z_np.ndim == 2 and z_np.shape[1] >= 2:
    plt.scatter(z_np[:, 0], z_np[:, 1], alpha=0.7, s=8)
else:
    plt.hist(z_np.flatten(), bins=30)
plt.title("Latent Embedding Distribution")
plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
plt.grid(True)
plt.tight_layout()
latent_plot = os.path.join(BASE_DIR, "latent.png")
plt.savefig(latent_plot, dpi=200)
plt.close()
print(f"💾 Saved latent distribution plot: {latent_plot}")

print("\n✅ All done! Open these locally:")
print(f" - {html_path}")
print(f" - {latent_plot}")

# === 8. GLOBAL LATENT SPACE (PCA Test) ===
# Questo controlla se l'encoder mappa TUTTE le mesh
# diverse allo stesso punto (collasso globale).

print("\n🔬 Performing Global Latent Space PCA test...")
from sklearn.decomposition import PCA
from tqdm import tqdm

all_latents = []
N_MESHES_FOR_PCA = 500 # Usa quante ne vuoi

with torch.no_grad():
    for i in tqdm(range(min(N_MESHES_FOR_PCA, len(dataset)))):
        sample = dataset[i]
        
        # Carica solo i dati necessari per l'encoder
        V = sample["verts"].to(device)
        mass = sample["mass"].to(device)
        L = sample["L"].to(device)
        evals = sample["evals"].to(device)
        evecs = sample["evecs"].to(device)
        gradX, gradY = sample["gradX"], sample["gradY"]

        # 1. Esegui l'encoder
        _, z = model(V, mass, L, evals, evecs, gradX, gradY)
        
        # 2. Ottieni il codice latente GLOBALE (media dei vertici)
        # z è [N_verts, 64] -> z_global è [1, 64]
        z_global = z.mean(dim=0) 
        
        all_latents.append(z_global.cpu().numpy())

# 3. Esegui PCA
# all_latents è ora una lista di 500 array [64,]
X = np.array(all_latents) # Shape [500, 64]
if np.isnan(X).any():
    print("[WARN] NaN detected in global latents, skipping PCA.")
else:
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X) # Shape [500, 2]

    # 4. Plotta
    plt.figure(figsize=(6, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.7, s=10)
    plt.title(f"Global Latent Space PCA ({N_MESHES_FOR_PCA} meshes)")
    plt.xlabel("PCA 1"); plt.ylabel("PCA 2")
    plt.grid(True)
    plt.tight_layout()
    global_latent_plot = os.path.join(BASE_DIR, "global_latent_PCA.png")
    plt.savefig(global_latent_plot, dpi=200)
    plt.close()
    print(f"💾 Saved global latent PCA plot: {global_latent_plot}")
    print(f" - {global_latent_plot}")