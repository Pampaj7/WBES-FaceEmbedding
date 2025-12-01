# ==========================================
# visualize_error_map_side.py  (final good version)
# GT a sinistra e Ricostruzione con colormap a destra
# ==========================================
import torch, numpy as np, plotly.graph_objects as go

import sys, os
# === aggiungi due livelli sopra al path ===
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# === Add DiffusionNet path ===
for p in [
    "/equilibrium/lpampaloni/diffusion-net/src",
    "/home/pampaj/diffusion-net/src",
    "/seidenas/users/lpampaloni/diffusion-net/src",
]:
    if p not in sys.path:
        sys.path.append(p)


# aggiungi la cartella dell'autoencoder al path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AUTOENC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # una directory sopra

if AUTOENC_DIR not in sys.path:
    sys.path.append(AUTOENC_DIR)

from diffusion_autoencoder import DiffusionAutoencoder
from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset

# === CONFIG ===
DATA_DIR = "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/GT_ready/npz_data_cropped_23470_with_ops"
CHECKPOINT = "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/gt_encdec/autoencoder/test_safe_latent/diffusionAE_epoch50.pth"
OUTPUT_HTML = "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/gt_encdec/autoencoder/test_safe_latent/error_map_visualization.html"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATENT_DIM = 256
WIDTH = 128
N_BLOCKS = 4

# === LOAD MODEL & DATA ===
model = DiffusionAutoencoder(latent_dim=LATENT_DIM, width=WIDTH, n_blocks=N_BLOCKS).to(DEVICE)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt)   # perché ckpt è già lo state_dict

model.eval()

dataset = GTReadyDataset(DATA_DIR)
print(f"✅ Dataset loaded: {len(dataset)} meshes")

# === SAMPLE ===
idx = 101
sample = dataset[idx]
V_gt = sample["verts"].to(DEVICE)
faces = sample["faces"].numpy()

# === INFERENCE ===
with torch.no_grad():
    V_rec, _ = model(
        V_gt,
        sample["mass"].to(DEVICE),
        sample["L"].to(DEVICE),
        sample["evals"].to(DEVICE),
        sample["evecs"].to(DEVICE),
        faces=sample["faces"].to(DEVICE),
        gradX=sample["gradX"].to(DEVICE),
        gradY=sample["gradY"].to(DEVICE),
    )

V_gt = V_gt.cpu().numpy()
V_rec = V_rec.cpu().numpy()
errors = np.linalg.norm(V_gt - V_rec, axis=1)
mean_err = errors.mean()
print(f"📏 Mean L2 error: {mean_err:.6f}")

# === PLOT ===
fig = go.Figure()

# --- Ground Truth (sinistra): solida, con luce morbida ---
fig.add_trace(go.Mesh3d(
    x=V_gt[:, 0],
    y=V_gt[:, 1],
    z=V_gt[:, 2],
    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
    color='lightblue',
    lighting=dict(ambient=0.6, diffuse=0.6, specular=0.2, roughness=0.9),
    lightposition=dict(x=0, y=0, z=2),
    flatshading=False,
    opacity=1.0,
    scene='scene1',
    name='Ground Truth'
))

# --- Reconstruction (destra): con colormap errore ---
fig.add_trace(go.Mesh3d(
    x=V_rec[:, 0] + 2.0,  # separazione
    y=V_rec[:, 1],
    z=V_rec[:, 2],
    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
    intensity=errors,
    colorscale='Turbo',
    showscale=True,
    colorbar_title='Error (L2)',
    lighting=dict(ambient=0.6, diffuse=0.6, specular=0.3, roughness=0.8),
    lightposition=dict(x=0, y=0, z=2),
    flatshading=False,
    scene='scene2',
    name='Reconstruction'
))

# === Layout ===
fig.update_layout(
    title=f"GT (sinistra) vs Ricostruzione con mappa d'errore (destra) — sample {idx} | Mean L2={mean_err:.5f}",
    width=1200, height=600,
    margin=dict(l=0, r=0, t=40, b=0),
    scene1=dict(
        domain=dict(x=[0, 0.48]),
        aspectmode='data',
        xaxis=dict(showbackground=False, visible=False),
        yaxis=dict(showbackground=False, visible=False),
        zaxis=dict(showbackground=False, visible=False),
    ),
    scene2=dict(
        domain=dict(x=[0.52, 1]),
        aspectmode='data',
        xaxis=dict(showbackground=False, visible=False),
        yaxis=dict(showbackground=False, visible=False),
        zaxis=dict(showbackground=False, visible=False),
    ),
    paper_bgcolor='white',
    plot_bgcolor='white',
    showlegend=False,
)

# === SAVE ===
os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
fig.write_html(OUTPUT_HTML, include_plotlyjs='cdn')
print(f"💾 Saved clean & visible side-by-side visualization → {OUTPUT_HTML}")
