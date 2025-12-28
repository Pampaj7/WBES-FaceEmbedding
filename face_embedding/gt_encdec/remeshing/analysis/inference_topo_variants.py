#!/usr/bin/env python3
import os
import re
import sys
import numpy as np
import torch

sys.path.append(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/gt_encdec/autoencoder"
)

from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset
from diffusion_autoencoder import DiffusionEncoderOnly

try:
    import diffusion_net
    DiffusionNet = diffusion_net.layers.DiffusionNet
except Exception:
    from diffusion_net import DiffusionNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# CONFIG
# ============================
DATA_DIR = (
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/REMESH/"
    "npz_data_topo_500_withops"
)
ENCODER_CKPT = "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/gt_encdec/remeshing/cross_topo_model/encoder_stage1_multitopo/encoder_stage1_epoch50.pth"
DECODER_CKPT = "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/gt_encdec/remeshing/cross_topo_model/stage2_frozen/stage2_decoder_epoch50.pth"

OUT_DIR = "2stage"
os.makedirs(OUT_DIR, exist_ok=True)

RECONS_DIR = os.path.join(OUT_DIR, "recon_npz")
os.makedirs(RECONS_DIR, exist_ok=True)

LATENTS_GLOBAL_PATH = os.path.join(OUT_DIR, "latents_global.npy")
LATENTS_MEAN_PATH   = os.path.join(OUT_DIR, "latents_mean.npy")
META_PATH           = os.path.join(OUT_DIR, "meta.npy")

fname_re = re.compile(r"^(id\d+_GTready)_(original|remesh|crop|noisy)\.npz$")
NUM_LIMIT = 200


# ============================
# UTILS
# ============================
def take_or_pad_evecs(evecs, k):
    n, kvar = evecs.shape
    if kvar >= k:
        return evecs[:, :k]
    pad = torch.zeros(n, k - kvar, device=evecs.device, dtype=evecs.dtype)
    return torch.cat([evecs, pad], dim=1)


def load_models():
    print("🔹 Loading EncoderOnly...")
    encoder = DiffusionEncoderOnly(
        latent_dim=256,
        width=128,
        n_blocks=4,
        dropout=0.0,
    ).to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_CKPT, map_location=DEVICE))
    encoder.eval()

    print("🔹 Loading Stage2 Decoder...")
    decoder = DiffusionNet(
        C_in=256 + 16,
        C_out=3,
        C_width=128,
        N_block=4,
        with_gradient_features=True,
        dropout=0.0
    ).to(DEVICE)
    decoder.load_state_dict(torch.load(DECODER_CKPT, map_location=DEVICE))
    decoder.eval()

    return encoder, decoder


# ============================
# INFERENCE
# ============================
@torch.no_grad()
def run_inference():

    dataset = GTReadyDataset(DATA_DIR)

    indexed_files = []
    for idx, path in enumerate(sorted(dataset.files)):
        base = os.path.basename(path)
        m = fname_re.match(base)
        if m:
            subj, variant = m.groups()
            indexed_files.append((idx, subj, variant, path))

    if NUM_LIMIT is not None:
        indexed_files = indexed_files[:NUM_LIMIT]

    print(f"⭐ Running inference on {len(indexed_files)} meshes...")

    encoder, decoder = load_models()

    all_latents_global = []
    all_latents_mean = []
    all_meta = []

    for i, (idx, subj, variant, path) in enumerate(indexed_files):

        sample = dataset[idx]

        V     = sample["verts"].to(DEVICE)
        mass  = sample["mass"].to(DEVICE)
        L     = sample["L"].to(DEVICE)
        evals = sample["evals"].to(DEVICE)
        evecs = sample["evecs"].to(DEVICE)
        faces = sample["faces"].to(DEVICE)
        gradX = sample["gradX"].to(DEVICE)
        gradY = sample["gradY"].to(DEVICE)

        # --------------------------
        # Z_global
        # --------------------------
        Zg = encoder(
            V, mass, L, evals, evecs,
            faces, gradX, gradY,
            return_per_vertex=False,
            add_noise=False
        )
        Zg = Zg.cpu().numpy()[0]
        all_latents_global.append(Zg)

        # --------------------------
        # Z_per → Z_mean
        # --------------------------
        Z_per, _ = encoder(
            V, mass, L, evals, evecs,
            faces, gradX, gradY,
            return_per_vertex=True,
            add_noise=False
        )
        Z_mean = Z_per.mean(dim=0).cpu().numpy()
        all_latents_mean.append(Z_mean)

        # --------------------------
        # Reconstruction
        # --------------------------
        S = take_or_pad_evecs(evecs, 16)
        Zin = torch.cat([Z_per, S], dim=1)

        Vrec = decoder(
            Zin, mass, L, evals, evecs,
            faces=faces, gradX=gradX, gradY=gradY
        ).cpu().numpy()

        out_npz = os.path.basename(path).replace(".npz", "_recon.npz")
        np.savez(os.path.join(RECONS_DIR, out_npz), V=Vrec)

        # Meta info
        all_meta.append({
            "subject": subj,
            "variant": variant,
            "path": path
        })

        print(f"[{i+1}/{len(indexed_files)}] processed")

    # --------------------------
    # SAVE RESULTS
    # --------------------------
    np.save(LATENTS_GLOBAL_PATH, np.stack(all_latents_global))
    np.save(LATENTS_MEAN_PATH, np.stack(all_latents_mean))
    np.save(META_PATH, np.array(all_meta, dtype=object))

    print("\n🎉 DONE.")
    print(f" - {LATENTS_GLOBAL_PATH}")
    print(f" - {LATENTS_MEAN_PATH}")
    print(f" - {META_PATH}")
    print(f" - {RECONS_DIR}/*.npz")


if __name__ == "__main__":
    run_inference()
