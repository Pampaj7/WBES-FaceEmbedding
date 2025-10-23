import torch.multiprocessing as mp

def collate_skip(batch):
    return [s for s in batch if s is not None]

def main():
    import torch
    import os
    from torch.utils.data import DataLoader, random_split
    from dataset_gtready import GTReadyDataset
    from diffusion_autoencoder import DiffusionAutoencoder
    from tqdm import tqdm
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from datetime import datetime
    from torch.utils.tensorboard import SummaryWriter

    mp.set_start_method("spawn", force=True)

    DATA_DIR = "../../../datasets/GT_ready/"
    OPS_DIR = os.path.join(DATA_DIR, "operators")
    OUT_DIR = "./results_diffusionAE/"
    LATENT_DIM = 64
    EPOCHS = 20
    LR = 1e-4
    BATCH_SIZE = 16
    VAL_SPLIT = 0.1
    CHECKPOINT_EVERY = 5
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"🚀 Training on {device} | batch={BATCH_SIZE}")

    dataset = GTReadyDataset(DATA_DIR, ops_dir=OPS_DIR)
    dataset.files = dataset.files[:1000]
    print(f"🧩 Using subset of {len(dataset.files)} meshes")

    n_val = int(len(dataset) * VAL_SPLIT)
    train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val])
    print(f"📚 Split: {len(train_set)} train / {len(val_set)} val")


    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, pin_memory=False, collate_fn=collate_skip)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=8, pin_memory=False, collate_fn=collate_skip)

    model = DiffusionAutoencoder(latent_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    criterion = nn.MSELoss()

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(OUT_DIR, "runs", run_name))
    log_csv = os.path.join(OUT_DIR, "train_log.csv")
    with open(log_csv, "w") as f:
        f.write("epoch,train_loss,val_loss\n")

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
                V = sample["verts"].to(device, non_blocking=True)
                mass = sample["mass"].to(device, non_blocking=True)
                L = sample["L"].to(device)
                evals = sample["evals"].to(device, non_blocking=True)
                evecs = sample["evecs"].to(device, non_blocking=True)
                gradX, gradY = sample.get("gradX"), sample.get("gradY")
                V_rec, _ = model(V, mass, L, evals, evecs, gradX, gradY)
                loss = criterion(V_rec, V)
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()

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
        with torch.no_grad():
            for sample in val_loader:
                sample = sample[0]
                V = sample["verts"].to(device)
                mass = sample["mass"].to(device)
                L = sample["L"].to(device)
                evals = sample["evals"].to(device)
                evecs = sample["evecs"].to(device)
                gradX, gradY = sample.get("gradX"), sample.get("gradY")
                V_rec, _ = model(V, mass, L, evals, evecs, gradX, gradY)
                val_loss += criterion(V_rec, V).item()

        val_loss /= max(1, len(val_loader))
        print(f"🧠 Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_mean:.6f} | Val Loss: {val_loss:.6f}")

        with open(log_csv, "a") as f:
            f.write(f"{epoch+1},{train_mean:.6f},{val_loss:.6f}\n")
        writer.add_scalar("Loss/train", train_mean, epoch+1)
        writer.add_scalar("Loss/val", val_loss, epoch+1)

        if (epoch + 1) % CHECKPOINT_EVERY == 0 or (epoch + 1) == EPOCHS:
            ckpt_path = os.path.join(OUT_DIR, f"diffusionAE_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

    writer.close()
    print("✅ Training + Validation completed")

if __name__ == "__main__":
    main()
