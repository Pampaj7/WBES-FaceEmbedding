# ==========================================
# visualize_error_map.py  — versione FIXATA
# Compatibile con Stage-2 (EncoderFrozenWithDecoder)
# ==========================================
import torch, numpy as np, plotly.graph_objects as go
import sys, os

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
from diffusion_autoencoder import DiffusionEncoderOnly
from helper import patch_dataset_with_get_by_name, collate_skip

try:
    import diffusion_net
    DiffusionNet = diffusion_net.layers.DiffusionNet
except Exception:
    from diffusion_net import DiffusionNet


# === CONFIG ===
DATA_DIR = "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/GT_ready/npz_data_cropped_23470_with_ops"
CHECKPOINT = "../stage2_frozen/stage2_decoder_epoch50.pth"
OUTPUT_HTML = "../stage2_frozen/error_map_visualization.html"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATENT_DIM = 256
K_SPEC = 16
C_IN = LATENT_DIM + K_SPEC   # =272
WIDTH = 128
N_BLOCKS = 4


# ============================================================
# 1. MODELLO: SOLO DECODER, identico allo Stage-2
# ============================================================
class DecoderOnlyStage2(torch.nn.Module):
    def __init__(self, c_in=C_IN, width=WIDTH, n_blocks=N_BLOCKS):
        super().__init__()
        self.decoder = DiffusionNet(
            C_in=c_in,
            C_out=3,
            C_width=width,
            N_block=n_blocks,
            with_gradient_features=True,
            dropout=0.0,
        )

    def forward(self, Z_in, mass, L, evals, evecs, faces, gradX, gradY):
        return self.decoder(
            Z_in, mass, L, evals, evecs,
            faces=faces, gradX=gradX, gradY=gradY
        )


# ============================================================
# 2. CARICAMENTO CHECKPOINT DECODER-ONLY
# ============================================================
model = DecoderOnlyStage2().to(DEVICE)

ckpt = torch.load(CHECKPOINT, map_location=DEVICE)

decoder_state = {}
for k, v in ckpt.items():
    # Il checkpoint stage-2 contiene SOLO le chiavi del decoder
    # che combaciano perfettamente con self.decoder.*
    if k.startswith("decoder."):
        decoder_state[k.replace("decoder.", "")] = v
    else:
        # nel tuo checkpoint i nomi sono:
        # first_lin.*, block_*.*
        decoder_state[k] = v

missing, unexpected = model.decoder.load_state_dict(decoder_state, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)
print("✅ Decoder-only Stage-2 loaded.")


# ============================================================
# 3. CARICA DATASET
# ============================================================
dataset = GTReadyDataset(DATA_DIR)
print(f"📂 Dataset loaded: {len(dataset)} meshes")


# ============================================================
# 4. PREPARA INPUT al decoder: Z = [0...0 , evecs[:16]]
# ============================================================
idx = 101
sample = dataset[idx]

V_gt = sample["verts"].to(DEVICE)
faces = sample["faces"].cpu().numpy()

# latenti fissi = 0
Z_latent = torch.zeros((V_gt.shape[0], LATENT_DIM), device=DEVICE)

# spectral features
spec = sample["evecs"][:, :K_SPEC].to(DEVICE)

# concat
Z_in = torch.cat([Z_latent, spec], dim=1)  # [N,272]

# ============================================================
# 5. INFERENCE
# ============================================================
with torch.no_grad():
    V_rec = model(
        Z_in,
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
mean_err = float(errors.mean())

print(f"📏 Mean L2 error: {mean_err:.6f}")


# ============================================================
# 6. VISUALIZATION
# ============================================================
fig = go.Figure()

# GT (sx)
fig.add_trace(go.Mesh3d(
    x=V_gt[:, 0], y=V_gt[:, 1], z=V_gt[:, 2],
    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
    color='lightblue', scene='scene1'
))

# REC (dx)
fig.add_trace(go.Mesh3d(
    x=V_rec[:, 0] + 2.0,
    y=V_rec[:, 1],
    z=V_rec[:, 2],
    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
    intensity=errors, colorscale='Turbo', showscale=True,
    scene='scene2'
))

fig.update_layout(
    width=1200, height=600,
    scene1=dict(domain=dict(x=[0, 0.48]), aspectmode='data'),
    scene2=dict(domain=dict(x=[0.52, 1]), aspectmode='data'),
)

os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
fig.write_html(OUTPUT_HTML)
print("💾 Saved:", OUTPUT_HTML)
