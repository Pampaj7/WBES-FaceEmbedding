import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader

from dataset_gtready import GTReadyDataset
from diffusion_autoencoder import DiffusionAutoencoder

# === Add DiffusionNet path ===
if "/home/pampaj/diffusion-net/src" not in sys.path:
    sys.path.append("/home/pampaj/diffusion-net/src")

torch.cuda.empty_cache()

def chamfer_distance(x, y):
    """
    Chamfer Distance (L2) tra due point cloud (x,y) [N,3] e [M,3].
    Restituisce uno scalare.
    """
    # x: [N,3], y: [M,3]
    x = x.unsqueeze(1)  # [N,1,3]
    y = y.unsqueeze(0)  # [1,M,3]
    dist = torch.sum((x - y) ** 2, dim=2)  # [N,M]
    min_x, _ = torch.min(dist, dim=1)
    min_y, _ = torch.min(dist, dim=0)
    cd = min_x.mean() + min_y.mean()
    return cd


# === CONFIG ===
DATA_DIR = "../../../datasets/GT_ready/"
OPS_DIR = os.path.join(DATA_DIR, "operators")
OUT_DIR = "./results_diffusionAE/"
LATENT_DIM = 32
EPOCHS = 5
LR = 1e-4
BATCH_SIZE = 16
VAL_SPLIT = 0.1
CHECKPOINT_EVERY = 5
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"🚀 Training on {device} | batch={BATCH_SIZE}")

# === Dataset ===
dataset = GTReadyDataset(DATA_DIR, ops_dir=OPS_DIR, device=device)
dataset.files = dataset.files[:1000]
print(f"🧩 Using subset of {len(dataset.files)} meshes")

# --- Train/Val split ---
n_val = int(len(dataset) * VAL_SPLIT)
train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val])
print(f"📚 Split: {len(train_set)} train / {len(val_set)} val")

def collate_skip(batch):
    return [s for s in batch if s is not None]

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=False, collate_fn=collate_skip)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=collate_skip)

# === Model, optimizer, loss ===
model = DiffusionAutoencoder(latent_dim=LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
criterion = nn.MSELoss()

# === Logging ===
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(log_dir=os.path.join(OUT_DIR, "runs", run_name))
log_csv = os.path.join(OUT_DIR, "train_log.csv")
with open(log_csv, "w") as f:
    f.write("epoch,train_loss,val_loss\n")

# === Training loop ===
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    valid_batches = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", dynamic_ncols=True)

    for batch_list in pbar:
        if len(batch_list) == 0:
            continue

        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0

        for sample in batch_list:
            try:
                V = sample["verts"].to(device)
                mass = sample["mass"].to(device)
                L = sample["L"].to(device)
                evals = sample["evals"].to(device)
                evecs = sample["evecs"].to(device)
                gradX, gradY = sample.get("gradX"), sample.get("gradY")

                V = torch.nan_to_num(V, nan=0.0)
                with torch.no_grad():
                    pass  # placeholder for later stability checks if needed

                V_rec, _ = model(V, mass, L, evals, evecs, gradX, gradY)
                loss = criterion(V_rec, V)
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()
            except RuntimeError as e:
                print(f"[WARN] Runtime error in batch ({sample['name']}): {e}")
                batch_loss += 0

        if len(batch_list) > 0:
            batch_mean = batch_loss / len(batch_list)
            epoch_loss += batch_mean
            valid_batches += 1
            pbar.set_postfix(loss=f"{batch_mean:.4f}")

    if valid_batches == 0:
        print(f"[WARN] No valid batches at epoch {epoch+1}, skipping.")
        continue

    train_mean = epoch_loss / valid_batches

    # === Validation ===
    model.eval()
    val_loss = 0.0
    val_cd = 0.0  # Chamfer distance accumulata

    with torch.no_grad():
        for sample in val_loader:
            sample = sample[0]
            V = sample["verts"].to(device)
            mass, L, evals, evecs = sample["mass"], sample["L"], sample["evals"], sample["evecs"]
            gradX, gradY = sample.get("gradX"), sample.get("gradY")

            V_rec, _ = model(V, mass, L, evals, evecs, gradX, gradY)

            # Mean Squared Error
            val_loss += criterion(V_rec, V).item()

            # Chamfer Distance
            cd = chamfer_distance(V_rec, V)
            val_cd += cd.item()

    # Medie
    val_loss /= max(1, len(val_loader))
    val_cd /= max(1, len(val_loader))

    print(f"🧠 Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_mean:.6f} | Val Loss: {val_loss:.6f} | Chamfer: {val_cd:.6f}")

    # === Logging ===
    with open(log_csv, "a") as f:
        f.write(f"{epoch+1},{train_mean:.6f},{val_loss:.6f},{val_cd:.6f}\n")
    writer.add_scalar("Loss/train", train_mean, epoch+1)
    writer.add_scalar("Loss/val", val_loss, epoch+1)
    writer.add_scalar("Chamfer/val", val_cd, epoch+1)

    # === Checkpoint ===
    if (epoch + 1) % CHECKPOINT_EVERY == 0 or (epoch + 1) == EPOCHS:
        ckpt_path = os.path.join(OUT_DIR, f"diffusionAE_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Saved checkpoint: {ckpt_path}")

writer.close()
print("✅ Training + Validation completed")

# === Quick Reconstruction Test ===
model.eval()
rec_dir = os.path.join(OUT_DIR, "reconstructions")
os.makedirs(rec_dir, exist_ok=True)

with torch.no_grad():
    for i, sample in enumerate(val_set):
        if sample is None:
            continue
        V = sample["verts"]
        mass, L, evals, evecs = sample["mass"], sample["L"], sample["evals"], sample["evecs"]
        gradX, gradY = sample.get("gradX"), sample.get("gradY")
        V_rec, _ = model(V, mass, L, evals, evecs, gradX, gradY)
        basename = os.path.basename(sample["name"]).replace(".obj", "")
        np.save(os.path.join(rec_dir, f"{basename}_rec.npy"), V_rec.cpu().numpy())
        if i >= 9:  # salva solo 10 esempi
            break

print("✅ Reconstructions saved in:", rec_dir)
print("📊 TensorBoard logs in:", os.path.join(OUT_DIR, "runs", run_name))
