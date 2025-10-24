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
import torch.multiprocessing as mp

# Importiamo la loss pura, senza dipendenze
from geometric_loss import GeometricLoss 

def collate_skip(batch):
    # Filtra i campioni che sono None (es. file corrotti dal dataset)
    return [s for s in batch if s is not None]

def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass 

    DATA_DIR = "../../../datasets/GT_ready/"
    OPS_DIR = os.path.join(DATA_DIR, "operators")
    OUT_DIR = "./results_diffusionAE/"
    
    # --- IPERPARAMETRI ---
    LATENT_DIM = 256
    WIDTH = 128
    N_BLOCKS = 4
    
    EPOCHS = 20
    LR = 1e-4 # MANTENIAMO 1e-5
    BATCH_SIZE = 8 
    VAL_SPLIT = 0.1
    CHECKPOINT_EVERY = 5
    
    # 🌟 === CORREZIONE PESI === 🌟
    # L1 e Normal sono già sulla scala di ~1.0
    # Laplacian è sulla scala di ~1e9
    # Dobbiamo scalare Laplacian per portarlo a ~1.0
    
    W_L1 = 1         # ⬅️ MODIFICATO: Riduciamo la "trazione" verso il collasso
    W_NORMAL = 1.0     # Manteniamo alto (segnale "gonfia")
    W_LAPLACIAN = 0.1  # ⬅️ MODIFICATO: Aumentiamo il peso della curvatura
    # 🌟 =========================
    
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"🚀 Training on {device} | logical batch={BATCH_SIZE} | LR={LR}")
    print(f"🧬 Latent Dim={LATENT_DIM} | Width={WIDTH} | Blocks={N_BLOCKS}")
    print(f"⚖️ Pesi Loss: L1={W_L1} | Normal={W_NORMAL} | Laplacian={W_LAPLACIAN}")

    dataset = GTReadyDataset(DATA_DIR, ops_dir=OPS_DIR)
    
    dataset.files = dataset.files[:1000]
    print(f"🧩 Using subset of {len(dataset.files)} meshes (potenziali None inclusi)")

    n_val = int(len(dataset) * VAL_SPLIT)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    print(f"📚 Split: {len(train_set)} train / {len(val_set)} val")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False, collate_fn=collate_skip)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=False, collate_fn=collate_skip)

    model = DiffusionAutoencoder(
        latent_dim=LATENT_DIM,
        width=WIDTH,
        n_blocks=N_BLOCKS
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    
    # Inizializza la loss con i nuovi pesi
    criterion = GeometricLoss(
        w_l1=W_L1, 
        w_normal=W_NORMAL, 
        w_laplacian=W_LAPLACIAN,
        device=device
    ).to(device)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(OUT_DIR, "runs", run_name))
    
    log_csv = os.path.join(OUT_DIR, "train_log.csv")
    with open(log_csv, "w") as f:
        f.write("epoch,train_loss,val_loss,train_l1,val_l1,train_normal,val_normal,train_laplacian,val_laplacian\n")


    for epoch in range(EPOCHS):
        model.train()
        
        epoch_loss_total = 0.0
        epoch_loss_l1 = 0.0
        epoch_loss_normal = 0.0
        epoch_loss_laplacian = 0.0
        valid_batches = 0
        
        printed_epoch_stats = False
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", dynamic_ncols=True)
        
        for batch_list in pbar:
            if len(batch_list) == 0:
                continue

            optimizer.zero_grad(set_to_none=True)
            
            batch_total_loss_value = 0.0
            batch_l1_value = 0.0
            batch_normal_value = 0.0
            batch_laplacian_value = 0.0

            for i, sample in enumerate(batch_list):
                try:
                    V = sample["verts"].to(device, non_blocking=True)
                    mass = sample["mass"].to(device, non_blocking=True)
                    evals = sample["evals"].to(device, non_blocking=True)
                    evecs = sample["evecs"].to(device, non_blocking=True)
                    faces = sample["faces"].to(device, non_blocking=True) 
                    L = sample["L"].to(device)
                    gradX = sample["gradX"].to(device)
                    gradY = sample["gradY"].to(device)

                    V_rec, Z_global = model(V, mass, L, evals, evecs, faces, gradX, gradY)
                    
                    loss, loss_breakdown = criterion(V_rec, V, faces, L)
                    
                    loss_scaled = loss / len(batch_list)
                    loss_scaled.backward()
                    
                    batch_total_loss_value += loss_breakdown["loss_total"]
                    batch_l1_value += loss_breakdown["loss_l1"]
                    batch_normal_value += loss_breakdown["loss_normal"]
                    batch_laplacian_value += loss_breakdown["loss_laplacian"]
                    
                    if not printed_epoch_stats and i == 0:
                        print(f"\n--- 🕵️ Debug Stats (Epoch {epoch+1}, 1st sample) ---")
                        print(f"  Verts_IN:  mean={V.mean():.4f}, std={V.std():.4f}, max_abs={V.abs().max():.4f}")
                        print(f"  Verts_OUT: mean={V_rec.mean():.4f}, std={V_rec.std():.4f}, max_abs={V_rec.abs().max():.4f}")
                        print(f"  Latent_Z:  mean={Z_global.mean():.4f}, std={Z_global.std():.4f}, max_abs={Z_global.abs().max():.4f}")
                        print(f"  Loss_Total: {loss_breakdown['loss_total']:.6f}")
                        # 🌟 Stampa le loss *PRIMA* del peso per il debug
                        print(f"  L1 (raw): {loss_breakdown['loss_l1']:.6f} | Normal (raw): {loss_breakdown['loss_normal']:.6f} | Laplacian (raw): {loss_breakdown['loss_laplacian']:.2f}")
                        print("-------------------------------------------------")
                        printed_epoch_stats = True
                
                except Exception as e:
                    print(f"\n[ERRORE] Salto campione {sample.get('name', 'N/A')} a causa di: {e}")
                    optimizer.zero_grad(set_to_none=True) 
                    batch_total_loss_value = 0.0
                    break 
            
            if batch_total_loss_value == 0.0:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            n_in_batch = len(batch_list)
            batch_mean_total = batch_total_loss_value / n_in_batch
            batch_mean_l1 = batch_l1_value / n_in_batch
            batch_mean_normal = batch_normal_value / n_in_batch
            
            epoch_loss_total += batch_mean_total
            epoch_loss_l1 += batch_mean_l1
            epoch_loss_normal += batch_mean_normal
            epoch_loss_laplacian += (batch_laplacian_value / n_in_batch) # Anche se non lo mostriamo
            
            valid_batches += 1
            pbar.set_postfix(loss=f"{batch_mean_total:.4f} (L1:{batch_mean_l1:.4f}, N:{batch_mean_normal:.4f})")

        if valid_batches == 0:
            print(f"[WARN] No valid batches at epoch {epoch+1}, skipping.")
            continue
            
        train_loss_total = epoch_loss_total / valid_batches
        train_loss_l1 = epoch_loss_l1 / valid_batches
        train_loss_normal = epoch_loss_normal / valid_batches
        train_loss_laplacian = epoch_loss_laplacian / valid_batches

        # === Validazione ===
        model.eval()
        val_loss_total = 0.0
        val_loss_l1 = 0.0
        val_loss_normal = 0.0
        val_loss_laplacian = 0.0
        n_val_samples = 0
        
        with torch.no_grad():
            for sample_list in val_loader: 
                if len(sample_list) == 0:
                    continue
                    
                sample = sample_list[0] 
                n_val_samples += 1
                
                V = sample["verts"].to(device)
                mass = sample["mass"].to(device)
                evals = sample["evals"].to(device)
                evecs = sample["evecs"].to(device)
                faces = sample["faces"].to(device)
                L = sample["L"].to(device)
                gradX = sample["gradX"].to(device)
                gradY = sample["gradY"].to(device)

                V_rec, _ = model(V, mass, L, evals, evecs, faces, gradX, gradY)
                
                _, loss_breakdown = criterion(V_rec, V, faces, L)
                
                val_loss_total += loss_breakdown["loss_total"]
                val_loss_l1 += loss_breakdown["loss_l1"]
                val_loss_normal += loss_breakdown["loss_normal"]
                val_loss_laplacian += loss_breakdown["loss_laplacian"]

        val_loss_total /= max(1, n_val_samples)
        val_loss_l1 /= max(1, n_val_samples)
        val_loss_normal /= max(1, n_val_samples)
        val_loss_laplacian /= max(1, n_val_samples)
        
        print(f"🧠 Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss_total:.6f} | Val Loss: {val_loss_total:.6f}")
        # 🌟 Stampa le loss *raw* (non pesate)
        print(f"    Train (L1/N/L): {train_loss_l1:.6f} / {train_loss_normal:.6f} / {train_loss_laplacian:.2f}")
        print(f"    Val   (L1/N/L): {val_loss_l1:.6f} / {val_loss_normal:.6f} / {val_loss_laplacian:.2f}")
        
        with open(log_csv, "a") as f:
            f.write(f"{epoch+1},{train_loss_total:.6f},{val_loss_total:.6f},"
                    f"{train_loss_l1:.6f},{val_loss_l1:.6f},"
                    f"{train_loss_normal:.6f},{val_loss_normal:.6f},"
                    f"{train_loss_laplacian:.6f},{val_loss_laplacian:.6f}\n")
                    
        writer.add_scalar("Loss_Total/train", train_loss_total, epoch+1)
        writer.add_scalar("Loss_Total/val", val_loss_total, epoch+1)
        writer.add_scalars("Loss_Breakdown_RAW/train", {
            'L1': train_loss_l1,
            'Normal': train_loss_normal,
            'Laplacian': train_loss_laplacian
        }, epoch+1)
        writer.add_scalars("Loss_Breakdown_RAW/val", {
            'L1': val_loss_l1,
            'Normal': val_loss_normal,
            'Laplacian': val_loss_laplacian
        }, epoch+1)

        if (epoch + 1) % CHECKPOINT_EVERY == 0 or (epoch + 1) == EPOCHS:
            ckpt_path = os.path.join(OUT_DIR, f"diffusionAE_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

    writer.close()
    print("✅ Training + Validation completed")

if __name__ == "__main__":
    main()