import sys
# === Add DiffusionNet path ===
# Assicurati che questi path siano corretti per il tuo ambiente
if "/equilibrium/lpampaloni/diffusion-net/src" not in sys.path:
    sys.path.append("/equilibrium/lpampaloni/diffusion-net/src")
if "/home/pampaj/diffusion-net/src" not in sys.path:
     sys.path.append("/home/pampaj/diffusion-net/src")

import os
import torch
import numpy as np
import igl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp
import glob # Per trovare l'ultimo checkpoint

from diffusion_autoencoder import DiffusionAutoencoder
from dataset_gtready import GTReadyDataset # Assicurati sia la versione aggiornata con clamp!

# === CONFIG ===
BASE_DIR = "./results_diffusionAE"
# 🌟 FIX: Trova l'ultimo checkpoint se epoch20 non esiste
DEFAULT_CHECKPOINT = os.path.join(BASE_DIR, "diffusionAE_epoch15.pth")
CHECKPOINT_PATTERN = os.path.join(BASE_DIR, "diffusionAE_epoch*.pth")
if os.path.exists(DEFAULT_CHECKPOINT):
    CHECKPOINT = DEFAULT_CHECKPOINT
else:
    checkpoints = sorted(glob.glob(CHECKPOINT_PATTERN))
    if checkpoints:
        CHECKPOINT = checkpoints[-1] # Prendi l'ultimo
        print(f"[WARN] Checkpoint di default non trovato, uso l'ultimo: {CHECKPOINT}")
    else:
        print(f"[ERRORE] Nessun checkpoint trovato in {BASE_DIR}")
        sys.exit(1)

DATA_DIR = "../../../datasets/GT_ready/"
OPS_DIR = os.path.join(DATA_DIR, "operators")
SAMPLE_IDX = 0 # Indice del campione da visualizzare

# 🌟 FIX: Parametri del modello che abbiamo usato
LATENT_DIM = 256
WIDTH = 128
N_BLOCKS = 4
# --------------

os.makedirs(BASE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 1. CARICA MODELLO ===
# 🌟 FIX: Usa i parametri corretti
model = DiffusionAutoencoder(
    latent_dim=LATENT_DIM,
    width=WIDTH,
    n_blocks=N_BLOCKS
).to(device)
try:
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
except Exception as e:
    print(f"[ERRORE] Impossibile caricare il checkpoint {CHECKPOINT}: {e}")
    sys.exit(1)

model.eval()
print(f"✅ Loaded checkpoint: {CHECKPOINT}")

# === 2. CARICA UN ESEMPIO DAL DATASET ===
dataset = GTReadyDataset(DATA_DIR, ops_dir=OPS_DIR)
if SAMPLE_IDX >= len(dataset):
    print(f"[ERRORE] Indice campione {SAMPLE_IDX} fuori dai limiti (Dataset size: {len(dataset)})")
    sys.exit(1)

sample = dataset[SAMPLE_IDX]
if sample is None:
    print(f"[ERRORE] Impossibile caricare il campione all'indice {SAMPLE_IDX}. Potrebbe essere corrotto.")
    sys.exit(1)

# Sposta tutti i tensori su device
V = sample["verts"].to(device)
faces = sample["faces"].to(device) # 🌟 FIX: Carica e sposta faces
mass = sample["mass"].to(device)
L = sample["L"].to(device)
evals = sample["evals"].to(device)
evecs = sample["evecs"].to(device)
gradX = sample["gradX"].to(device) # 🌟 FIX: Sposta su device
gradY = sample["gradY"].to(device) # 🌟 FIX: Sposta su device
fname = sample["name"]
print(f"📄 Loaded sample: {fname}")


# === 3. RICOSTRUZIONE ===
with torch.no_grad():
    # 🌟 FIX: Passa faces, gradX, gradY. La variabile 'z' è Z_global.
    V_rec, Z_global = model(V, mass, L, evals, evecs, faces, gradX, gradY)

V_gt_np = V.cpu().numpy()
V_rec_np = V_rec.cpu().numpy()
F_np = faces.cpu().numpy() # 🌟 FIX: Usa faces da device
Z_global_np = Z_global.cpu().numpy() # Shape [1, latent_dim]

# === 4. SALVA MESH SU DISCO ===
igl.write_triangle_mesh(os.path.join(BASE_DIR, f"{fname}_original.obj"), V_gt_np, F_np)
igl.write_triangle_mesh(os.path.join(BASE_DIR, f"{fname}_reconstructed.obj"), V_rec_np, F_np)
print(f"💾 Saved {fname}_original.obj and {fname}_reconstructed.obj")

# === 5. CALCOLA METRICHE ===
# Funzione Chamfer (manuale, ok per ora)
def chamfer_distance(x_np, y_np):
    x = torch.from_numpy(x_np).float().unsqueeze(0) # Batch dim 1
    y = torch.from_numpy(y_np).float().unsqueeze(0) # Batch dim 1

    # Usa KNN da pytorch3d se disponibile, altrimenti calcolo manuale più lento
    try:
        from pytorch3d.loss import chamfer_distance as chamfer_pytorch3d
        # Pytorch3D chamfer richiede (N, P, D) e restituisce (loss, None)
        loss, _ = chamfer_pytorch3d(x, y)
        return loss.item()
    except ImportError:
        # Calcolo manuale (può essere lento/memory intensive per N grandi)
        x = x.squeeze(0).unsqueeze(1) # [N, 1, 3]
        y = y.squeeze(0).unsqueeze(0) # [1, M, 3]
        dist = torch.sum((x - y)**2, dim=2) # [N, M]
        min_dist_xy, _ = torch.min(dist, dim=1) # [N]
        min_dist_yx, _ = torch.min(dist, dim=0) # [M]
        return (min_dist_xy.mean() + min_dist_yx.mean()).item()


# Funzione Laplacian loss (usa operatore L già su device)
def laplacian_loss_metric(V_tensor, L_sparse):
    # L è già su device dal caricamento
    # V_tensor deve essere su device
    if V_tensor.device != L_sparse.device:
        V_tensor = V_tensor.to(L_sparse.device)
    LV = torch.sparse.mm(L_sparse, V_tensor)
    # Calcola la norma L2 media delle coordinate Laplaciane
    return torch.mean(torch.norm(LV, p=2, dim=1)).item()


chamfer = chamfer_distance(V_gt_np, V_rec_np)
# 🌟 FIX: Passa il tensore V_rec (che è su GPU)
lap_smoothness = laplacian_loss_metric(V_rec, L)
vertex_errors = np.linalg.norm(V_gt_np - V_rec_np, axis=1)
mean_vertex_error = vertex_errors.mean()
max_vertex_error = vertex_errors.max()

print(f"📊 Chamfer Distance: {chamfer:.6f}")
print(f"📊 Mean Vertex Error: {mean_vertex_error:.6f} | Max: {max_vertex_error:.6f}")
print(f"📊 Laplacian Smoothness (Rec): {lap_smoothness:.6f}") # Indica quanto è liscia la ricostruzione

np.save(os.path.join(BASE_DIR, f"{fname}_vertex_errors.npy"), vertex_errors)

# === 6. VISUALIZZAZIONE INTERATTIVA SALVATA IN HTML ===
def mesh_trace(V, F, vertex_colors=None, name="mesh", colorscale="Viridis"):
    trace = go.Mesh3d(
        x=V[:, 0], y=V[:, 1], z=V[:, 2],
        i=F[:, 0], j=F[:, 1], k=F[:, 2],
        opacity=1.0,
        name=name
    )
    if vertex_colors is not None:
        trace.update(intensity=vertex_colors, colorscale=colorscale, showscale=True)
    else:
        # Usa colore di default se non specificato
        trace.update(color='lightblue')
    return trace

fig = sp.make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'surface'}, {'type': 'surface'}]],
    subplot_titles=(f"Ground Truth ({fname})", f"Reconstructed (MVE: {mean_vertex_error:.4f})")
)
fig.add_trace(mesh_trace(V_gt_np, F_np, name="GT"), row=1, col=1)
# Colora la ricostruzione per errore
fig.add_trace(mesh_trace(V_rec_np, F_np, vertex_colors=vertex_errors, name="Reconstruction", colorscale="Plasma"), row=1, col=2)

fig.update_layout(
    scene=dict(aspectmode='data'), # Usa aspect ratio corretto per mesh 1
    scene2=dict(aspectmode='data'), # Usa aspect ratio corretto per mesh 2
    height=700,
    title_text=f"Mesh Reconstruction Comparison - Epoch {CHECKPOINT.split('epoch')[-1].split('.')[0]}",
    legend_title_text='Mesh'
    )
html_path = os.path.join(BASE_DIR, f"{fname}_comparison.html")
fig.write_html(html_path)
print(f"💾 Saved interactive comparison: {html_path}")

# === 7. DISTRIBUZIONE LATENTE (Istogramma) ===
# 🌟 FIX: Z_global è [1, latent_dim], plottiamo l'istogramma dei suoi valori
plt.figure(figsize=(8, 5))
plt.hist(Z_global_np.flatten(), bins=min(50, LATENT_DIM)) # Flatten per ottenere tutti i valori
plt.title(f"Global Latent Code Distribution ({fname})")
plt.xlabel("Latent Dimension Value")
plt.ylabel("Frequency")
plt.grid(True, axis='y')
plt.tight_layout()
latent_plot = os.path.join(BASE_DIR, f"{fname}_latent_hist.png")
plt.savefig(latent_plot, dpi=150)
plt.close()
print(f"💾 Saved latent distribution histogram: {latent_plot}")

print("\n✅ All done! Open these files:")
print(f" - {html_path}")
print(f" - {latent_plot}")


# === 8. GLOBAL LATENT SPACE (PCA Test) ===
print("\n🔬 Performing Global Latent Space PCA test...")
from sklearn.decomposition import PCA
from tqdm import tqdm

all_latents = []
N_MESHES_FOR_PCA = min(500, len(dataset)) # Usa max 500 o tutte se meno

with torch.no_grad():
    for i in tqdm(range(N_MESHES_FOR_PCA), desc="Encoding meshes for PCA"):
        sample_pca = dataset[i]
        if sample_pca is None: continue # Salta campioni corrotti

        # Carica solo i dati necessari e spostali su device
        V_pca = sample_pca["verts"].to(device)
        faces_pca = sample_pca["faces"].to(device) # 🌟 FIX: Carica faces
        mass_pca = sample_pca["mass"].to(device)
        L_pca = sample_pca["L"].to(device)
        evals_pca = sample_pca["evals"].to(device)
        evecs_pca = sample_pca["evecs"].to(device)
        gradX_pca = sample_pca["gradX"].to(device) # 🌟 FIX: Sposta su device
        gradY_pca = sample_pca["gradY"].to(device) # 🌟 FIX: Sposta su device

        # Esegui il modello per ottenere il codice latente GLOBALE
        # 🌟 FIX: Passa tutti gli argomenti, usa Z_global
        _, Z_global_pca = model(V_pca, mass_pca, L_pca, evals_pca, evecs_pca, faces_pca, gradX_pca, gradY_pca)

        # 🌟 FIX: Z_global_pca è già [1, latent_dim], non serve fare la media
        all_latents.append(Z_global_pca.squeeze(0).cpu().numpy()) # squeeze(0) per rimuovere la dim batch

# Esegui PCA
if not all_latents:
    print("[WARN] No valid latents found for PCA.")
else:
    X = np.array(all_latents) # Shape [N_MESHES_FOR_PCA, latent_dim]
    if np.isnan(X).any() or not np.isfinite(X).all():
        print("[WARN] NaN/Inf detected in global latents, skipping PCA.")
    elif X.shape[0] < 2:
        print("[WARN] Not enough samples for PCA (need at least 2).")
    else:
        try:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X) # Shape [N_MESHES_FOR_PCA, 2]

            # Plotta
            plt.figure(figsize=(7, 7))
            plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.6, s=12, cmap='viridis', c=np.arange(X_2d.shape[0]))
            plt.title(f"Global Latent Space PCA ({X_2d.shape[0]} meshes)")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.grid(True)
            plt.tight_layout()
            global_latent_plot = os.path.join(BASE_DIR, "global_latent_PCA.png")
            plt.savefig(global_latent_plot, dpi=150)
            plt.close()
            print(f"💾 Saved global latent PCA plot: {global_latent_plot}")
            print(f" - {global_latent_plot}")
        except Exception as e:
            print(f"[ERRORE] PCA failed: {e}")

print("\n✅ PCA Test finished.")