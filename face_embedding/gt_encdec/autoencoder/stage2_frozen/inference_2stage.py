# ==========================================
# visualize_error_map.py  — versione FIXATA
# Compatibile con Stage-2 (EncoderFrozenWithDecoder)
# ==========================================
import torch, numpy as np, plotly.graph_objects as go
import sys, os
import torch.nn as nn

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
# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/GT_ready/npz_data_cropped_23470_with_ops"
ENCODER_CKPT = "../encoder_only/encoder_only_epoch50.pth"
DECODER_CKPT = "../stage2_frozen/stage2_decoder_epoch50.pth"
OUTPUT_HTML = "../stage2_frozen/error_map_visualization_stage2.html"

LATENT_DIM = 256
WIDTH = 128
N_BLOCKS = 4
K_SPEC = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# FULL STAGE-2 MODEL
# ============================================================
class Stage2(nn.Module):
    def __init__(self):
        super().__init__()

        # -------- Encoder (frozen) --------
        self.encoder = DiffusionEncoderOnly(
            latent_dim=LATENT_DIM,
            width=WIDTH,
            n_blocks=N_BLOCKS,
            dropout=0.1,
        )
        print("📂 Loading encoder:", ENCODER_CKPT)
        enc_state = torch.load(ENCODER_CKPT, map_location=DEVICE)
        self.encoder.load_state_dict(enc_state)

        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        # -------- Decoder (trainable in stage-2) --------
        self.decoder = DiffusionNet(
            C_in=LATENT_DIM + K_SPEC,
            C_out=3,
            C_width=WIDTH,
            N_block=N_BLOCKS,
            with_gradient_features=True,
            dropout=0.0,
        )

        print("📂 Loading decoder:", DECODER_CKPT)
        dec_state = torch.load(DECODER_CKPT, map_location=DEVICE)
        self.decoder.load_state_dict(dec_state)

        self.decoder.eval()

    def pad_evecs(self, evecs):
        if evecs.shape[1] >= K_SPEC:
            return evecs[:, :K_SPEC]
        pad = torch.zeros(evecs.shape[0], K_SPEC - evecs.shape[1], device=evecs.device)
        return torch.cat([evecs, pad], dim=1)

    def forward(self, sample):
        V = sample["verts"].to(DEVICE)
        mass = sample["mass"].to(DEVICE)
        L = sample["L"].to(DEVICE)
        evals = sample["evals"].to(DEVICE)
        evecs = sample["evecs"].to(DEVICE)
        faces = sample["faces"].to(DEVICE)
        gradX = sample["gradX"].to(DEVICE)
        gradY = sample["gradY"].to(DEVICE)

        # ---- 1. ENCODER (frozen) → gives Z_per_vertex ----
        Z_per_vertex, _ = self.encoder(
            V, mass, L, evals, evecs,
            faces, gradX, gradY,
            return_per_vertex=True,
            add_noise=False
        )

        # ---- 2. spectral features ----
        S = self.pad_evecs(evecs)

        # ---- 3. concat ----
        Z_in = torch.cat([Z_per_vertex, S], dim=1)

        # ---- 4. decoder ----
        V_rec = self.decoder(
            Z_in, mass, L, evals, evecs,
            faces=faces, gradX=gradX, gradY=gradY
        )

        return V_rec


# ============================================================
# LOAD MODEL
# ============================================================
model = Stage2().to(DEVICE)
model.eval()

# ============================================================
# LOAD DATA
# ============================================================
dataset = GTReadyDataset(DATA_DIR)
print(f"✅ Loaded {len(dataset)} meshes")

idx = 101
sample = dataset[idx]
faces_np = sample["faces"].numpy()

# ============================================================
# RUN
# ============================================================
with torch.no_grad():
    V_rec = model(sample)

V_gt = sample["verts"].cpu().numpy()
V_rec = V_rec.cpu().numpy()
errors = np.linalg.norm(V_gt - V_rec, axis=1)
mean_err = errors.mean()
print("📏 Mean L2:", mean_err)

# ============================================================
# PLOT (GT left, REC right)
# ============================================================
fig = go.Figure()

fig.add_trace(go.Mesh3d(
    x=V_gt[:, 0], y=V_gt[:, 1], z=V_gt[:, 2],
    i=faces_np[:, 0], j=faces_np[:, 1], k=faces_np[:, 2],
    color='lightblue', opacity=1.0, scene='scene1'
))

fig.add_trace(go.Mesh3d(
    x=V_rec[:, 0] + 2.0, y=V_rec[:, 1], z=V_rec[:, 2],
    i=faces_np[:, 0], j=faces_np[:, 1], k=faces_np[:, 2],
    intensity=errors, colorscale='Turbo', showscale=True,
    scene='scene2'
))

fig.update_layout(
    title=f"Stage-2 Reconstruction — Mean L2 = {mean_err:.4f}",
    width=1200, height=600,
    scene1=dict(domain=dict(x=[0, 0.48]), aspectmode='data'),
    scene2=dict(domain=dict(x=[0.52,1]), aspectmode='data'),
)

fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn")
print("💾 Saved:", OUTPUT_HTML)