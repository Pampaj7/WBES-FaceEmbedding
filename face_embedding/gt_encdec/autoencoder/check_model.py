import os
import sys
import torch
import numpy as np
import trimesh
from torch.utils.data import DataLoader

# === Add DiffusionNet path ===
if "/equilibrium/lpampaloni/diffusion-net/src" not in sys.path:
    sys.path.append("/equilibrium/lpampaloni/diffusion-net/src")

from diffusion_autoencoder import DiffusionAutoencoder
from dataset_gtready import GTReadyDataset

# === Config ===
DATA_DIR = "../../../datasets/GT_ready/"
OPS_DIR = os.path.join(DATA_DIR, "operators")
CKPT_PATH = "results_diffusionAE/diffusionAE_epoch30.pth"
RECON_DIR = "results_diffusionAE/reconstructions_subset/"
os.makedirs(RECON_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load model ===
model = DiffusionAutoencoder(latent_dim=32).to(device)
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.eval()

# === Dataset (solo primi 1000 campioni validi) ===
dataset = GTReadyDataset(DATA_DIR, ops_dir=OPS_DIR, device=device)
dataset.files = dataset.files[:1000]

# ✅ collate_fn personalizzato per evitare batching di tensori sparsi
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=lambda x: [s for s in x if s is not None]
)

n_total = 0
n_skipped_in = 0
n_skipped_out = 0
n_saved = 0

with torch.no_grad(), torch.amp.autocast("cuda"):
    for i, batch_list in enumerate(loader):
        sample = batch_list[0]
        if sample is None:
            continue
        n_total += 1

        # === Input check ===
        tensors_to_check = [
            ("verts", sample["verts"]),
            ("mass", sample["mass"]),
            ("L", sample["L"]),
            ("evals", sample["evals"]),
            ("evecs", sample["evecs"]),
        ]
        corrupted = False
        for name, t in tensors_to_check:
            if t is None:
                corrupted = True
                break
            if torch.isnan(t).any() or torch.isinf(t).any():
                print(f"[WARN] NaN/Inf in input {name} for {sample['name']}, skipping.")
                corrupted = True
                break
        if corrupted:
            n_skipped_in += 1
            continue

        # === Forward pass ===
        try:
            V = sample["verts"].to(device)
            mass = sample["mass"].to(device)
            L = sample["L"].to(device)
            evals = sample["evals"].to(device)
            evecs = sample["evecs"].to(device)
            gradX = sample.get("gradX")
            gradY = sample.get("gradY")

            V_rec, _ = model(V, mass, L, evals, evecs, gradX, gradY)
        except RuntimeError as e:
            print(f"[ERROR] Runtime error for {sample['name']}: {e}")
            n_skipped_out += 1
            continue

        # === Check output ===
        if torch.isnan(V_rec).any() or torch.isinf(V_rec).any():
            print(f"[WARN] NaN/Inf in output for {sample['name']}, skipping save.")
            n_skipped_out += 1
            continue

        mean, std = V_rec.mean().item(), V_rec.std().item()
        vmin, vmax = V_rec.min().item(), V_rec.max().item()
        print(f"[OK] {sample['name']}: mean={mean:.4f}, std={std:.4f}, range=[{vmin:.3f},{vmax:.3f}]")

        # === Salva mesh solo se sana ===
        verts_np = V_rec.cpu().numpy()
        faces_np = sample["faces"].cpu().numpy()
        mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
        mesh.export(os.path.join(RECON_DIR, f"recon_{i:04d}.obj"))
        n_saved += 1

        if n_saved >= 10:  # solo 10 sane
            break

print("✅ Reconstruction Summary:")
print(f"  Total samples processed: {n_total}")
print(f"  Skipped (invalid input): {n_skipped_in}")
print(f"  Skipped (invalid output): {n_skipped_out}")
print(f"  Saved clean reconstructions: {n_saved}")
print(f"📂 Saved in: {RECON_DIR}")
