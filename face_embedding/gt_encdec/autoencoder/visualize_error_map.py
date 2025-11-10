# ==========================================
# visualize_error_map_side.py  (final good version)
# GT a sinistra e Ricostruzione con colormap a destra
# ==========================================
import os, torch, numpy as np, plotly.graph_objects as go
from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset
from diffusion_autoencoder import DiffusionAutoencoder

# === CONFIG ===
DATA_DIR = "../../../datasets/GT_ready/npz_data/"
CHECKPOINT = "./results_diffusionAE/diffusionAE_5000_epoch45.pth"
OUTPUT_HTML = "./results_diffusionAE/error_map_side_final.html"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATENT_DIM = 256
WIDTH = 128
N_BLOCKS = 4

# === LOAD MODEL & DATA ===
model = DiffusionAutoencoder(latent_dim=LATENT_DIM, width=WIDTH, n_blocks=N_BLOCKS).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

dataset = GTReadyDataset(DATA_DIR)
print(f"✅ Dataset loaded: {len(dataset)} meshes")

# === SAMPLE ===
idx = 99
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
