import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import pickle

from dataset_gtready import GTReadyDataset
from diffusion_autoencoder import DiffusionAutoencoder

# === Add DiffusionNet path ===
if "/home/pampaj/diffusion-net/src" not in sys.path:
    sys.path.append("/home/pampaj/diffusion-net/src")

# === CONFIG ===
DATA_DIR = "../../../datasets/GT_ready/"
OPS_DIR = os.path.join(DATA_DIR, "operators")   # nuova directory con .pkl
OUT_DIR = "./results_diffusionAE/"
LATENT_DIM = 32
EPOCHS = 30
LR = 1e-4
BATCH_SIZE = 8
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training on {device} | batch={BATCH_SIZE}")

# === Dataset ===
dataset = GTReadyDataset(DATA_DIR, ops_dir=OPS_DIR, device=device)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    collate_fn=lambda x: x  # custom batching handled inside loop
)

# === Model ===
model = DiffusionAutoencoder(latent_dim=LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
scaler = torch.cuda.amp.GradScaler()  # mixed precision scaler

# === Training ===
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", dynamic_ncols=True)
    for batch_list in pbar:
        # batch_list is a list of dicts (batch_size)
        batch_loss = 0.0
        for sample in batch_list:
            V = sample["verts"]
            mass = sample["mass"]
            L = sample["L"]
            evals = sample["evals"]
            evecs = sample["evecs"]
            gradX = sample.get("gradX", None)
            gradY = sample.get("gradY", None)

            if V.shape[0] == 3 and V.shape[1] != 3:
                V = V.T
            elif V.ndim == 1:
                V = V.unsqueeze(0)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                V_rec, _ = model(V, mass, L, evals, evecs, gradX, gradY)
                loss = criterion(V_rec, V)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_loss += loss.item()
        batch_mean = batch_loss / len(batch_list)
        epoch_loss += batch_mean
        pbar.set_postfix(loss=f"{batch_mean:.4f}")

    print(f"🧠 Epoch {epoch+1}/{EPOCHS} | Mean Loss: {epoch_loss/len(loader):.6f}")

torch.save(model.state_dict(), os.path.join(OUT_DIR, "diffusionAE.pth"))
print("✅ Training completed")

# === Reconstruction test ===
model.eval()
with torch.no_grad(), torch.cuda.amp.autocast():
    for sample in tqdm(dataset, desc="Reconstructing"):
        V = sample["verts"]
        mass = sample["mass"]
        L = sample["L"]
        evals = sample["evals"]
        evecs = sample["evecs"]
        gradX = sample.get("gradX", None)
        gradY = sample.get("gradY", None)

        if V.shape[0] == 3 and V.shape[1] != 3:
            V = V.T

        V_rec, _ = model(V, mass, L, evals, evecs, gradX, gradY)
        np.save(
            os.path.join(OUT_DIR, sample["name"].replace(".obj", "_rec.npy")),
            V_rec.cpu().numpy()
        )

print("✅ Reconstructions saved")
