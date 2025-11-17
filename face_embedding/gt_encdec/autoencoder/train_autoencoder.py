import torch
import os
from torch.utils.data import DataLoader, random_split
# 🌟 1. Importa la nuova classe Dataset NPZ (rinominandola per comodità)
# from dataset_gtready import GTReadyDataset # Vecchio import
from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset # Nuovo import
from diffusion_autoencoder import DiffusionAutoencoder
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import math # Per math.isfinite
import torch.nn.functional as F

# Importiamo la loss pura, senza dipendenze
from geometric_loss import GeometricLoss

def collate_skip(batch):
    # Filtra i campioni che sono None
    return [s for s in batch if s is not None]

def main():
    try:
        current_start_method = mp.get_start_method(allow_none=True)
        if current_start_method is None:
             mp.set_start_method("spawn", force=True)
        elif current_start_method != "spawn":
             print(f"[WARN] Metodo start multiprocessing già impostato su '{current_start_method}'.")
    except RuntimeError: pass
    except Exception as e: print(f"[ERRORE] Errore imprevisto setup multiprocessing: {e}")

    # 🌟 2. Aggiorna DATA_DIR per puntare alla cartella NPZ
    # DATA_DIR = "../../../datasets/GT_ready/" # Vecchio path (.obj)
    DATA_DIR = "../../../datasets/GT_ready/npz_data_cropped_23470_with_ops/" # Nuovo path (.npz)
    # 🌟 3. Rimuovi OPS_DIR (non più necessario con NPZ)
    # OPS_DIR = os.path.join(DATA_DIR, "operators") # Rimosso

    OUT_DIR = "howwwwwwwww"

    # --- IPERPARAMETRI ---
    LATENT_DIM = 256
    WIDTH = 128
    N_BLOCKS = 4
    EPOCHS = 50
    LR = 1e-4 # Learning rate iniziale
    BATCH_SIZE = 16 # Dimensione batch (accumulazione)
    N_WORKERS = 0 # Numero worker (prova 8 o 16)
    PIN_MEMORY = False # Abilita pin_memory se usi GPU
    VAL_SPLIT = 0.1
    CHECKPOINT_EVERY = 5

    # Pesi Loss (configurazione finale)
    W_L1 = 0.3
    W_NORMAL = 1.0
    W_LAPLACIAN = 0.7

    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda': torch.backends.cudnn.benchmark = True
    else: torch.backends.cudnn.benchmark = False

    print(f"🚀 Training on {device} | logical batch={BATCH_SIZE} | LR (start)={LR}")
    print(f"🧬 Latent Dim={LATENT_DIM} | Width={WIDTH} | Blocks={N_BLOCKS}")
    print(f"⚖️ Pesi Loss: L1={W_L1} | Normal={W_NORMAL} | Laplacian={W_LAPLACIAN}")
    print(f"💾 Using NPZ dataset from: {DATA_DIR}") # Aggiunto log per NPZ
    print(f"⚙️ DataLoader: num_workers={N_WORKERS}, pin_memory={PIN_MEMORY}")

    # 🌟 4. Crea l'istanza del dataset usando la nuova classe e il nuovo path
    # dataset = GTReadyDataset(DATA_DIR, ops_dir=OPS_DIR) # Vecchio modo
    dataset = GTReadyDataset(DATA_DIR) # Nuovo modo (usa GTReadyDatasetNPZ importato come GTReadyDataset)

    # Limita opzionalmente il dataset (es. per debug)
    dataset.files = dataset.files[:3000]
    print(f"🧩 Using subset of {len(dataset.files)} meshes")

    n_samples = len(dataset)
    if n_samples == 0: print("[ERRORE] Dataset NPZ vuoto."); return
    n_val = int(n_samples * VAL_SPLIT); n_train = n_samples - n_val
    if n_train <= 0 or n_val <= 0: print(f"[ERRORE] Dataset troppo piccolo ({n_samples}) per split."); return
    try: train_set, val_set = random_split(dataset, [n_train, n_val])
    except Exception as e: print(f"[ERRORE] Fallimento random_split: {e}"); return
    print(f"📚 Split: {len(train_set)} train / {len(val_set)} val")

    # DataLoaders configurati per la parallelizzazione
    train_loader = DataLoader(train_set, 
                              batch_size=BATCH_SIZE, 
                              shuffle=True,
                              num_workers=N_WORKERS, 
                              pin_memory=PIN_MEMORY, 
                              collate_fn=collate_skip,
                              persistent_workers=True if N_WORKERS > 0 else False,
                              prefetch_factor=1 if N_WORKERS > 0 else None)
    
    val_loader = DataLoader(val_set, 
                            batch_size=1, 
                            shuffle=False,
                            num_workers=max(1, N_WORKERS // 2), 
                            pin_memory=PIN_MEMORY, collate_fn=collate_skip,
                            persistent_workers=True if N_WORKERS > 0 else False,
                            prefetch_factor=1 if N_WORKERS > 0 else None)

    model = DiffusionAutoencoder(latent_dim=LATENT_DIM, width=WIDTH, n_blocks=N_BLOCKS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)

    # Scheduler (senza verbose)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7)
    criterion = GeometricLoss(w_l1=W_L1, w_normal=W_NORMAL, w_laplacian=W_LAPLACIAN, device=device).to(device)

    # Setup Logging
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(OUT_DIR, "runs", run_name)
    try: writer = SummaryWriter(log_dir=log_dir); print(f"📊 TensorBoard logs salvati in: {log_dir}")
    except Exception as e: print(f"[ERRORE] Creazione SummaryWriter fallita: {e}"); return
    log_csv = os.path.join(OUT_DIR, "train_log.csv")
    try:
        with open(log_csv, "w") as f: f.write("epoch,train_loss,val_loss,train_l1,val_l1,train_normal,val_normal,train_laplacian,current_lr\n")
    except IOError as e: print(f"[ERRORE] Scrittura log CSV fallita: {e}"); return

    # === Ciclo di Training ===
    print(f"\n--- Inizio Training ---")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss_total, epoch_loss_l1, epoch_loss_normal, epoch_loss_laplacian = 0.0, 0.0, 0.0, 0.0
        valid_batches = 0
        printed_epoch_stats = False
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", dynamic_ncols=True, unit="batch")

        for batch_list in pbar:
            current_lr_pbar = optimizer.param_groups[0]['lr']
            pbar.set_postfix(lr=f"{current_lr_pbar:.1e}")
            if len(batch_list) == 0: continue
            optimizer.zero_grad(set_to_none=True)

            batch_total_loss_value, batch_l1_value, batch_normal_value, batch_laplacian_value = 0.0, 0.0, 0.0, 0.0
            batch_processed_samples = 0

            for i, sample in enumerate(batch_list):
                try:
                    V = sample["verts"].to(device, non_blocking=PIN_MEMORY) # Usa non_blocking se pin_memory=True
                    mass = sample["mass"].to(device, non_blocking=PIN_MEMORY)
                    evals = sample["evals"].to(device, non_blocking=PIN_MEMORY)
                    evecs = sample["evecs"].to(device, non_blocking=PIN_MEMORY)
                    faces = sample["faces"].to(device, non_blocking=PIN_MEMORY)
                    L = sample["L"].to(device)
                    gradX = sample["gradX"].to(device)
                    gradY = sample["gradY"].to(device)

                    V_rec, Z_Global = model(V, mass, L, evals, evecs, faces, gradX, gradY)
                    loss, loss_breakdown = criterion(V_rec, V, faces, L)

                        
                    # === Metriche di similarità globale (solo per logging qualitativo) ===
                    try:
                        cosine_mesh = F.cosine_similarity(V_rec.flatten(), V.flatten(), dim=0).item()
                        corr_matrix = torch.corrcoef(torch.stack([V_rec.flatten(), V.flatten()]))
                        corr_mesh = corr_matrix[0, 1].item()
                    except Exception as e:
                        cosine_mesh, corr_mesh = float('nan'), float('nan')

    
                    if not torch.isfinite(loss):
                        print(f"\n[ERRORE] Loss non finita campione {sample.get('name', 'N/A')}. Salto."); continue

                    loss_scaled = loss / len(batch_list)
                    loss_scaled.backward()

                    batch_total_loss_value += loss_breakdown["loss_total"]
                    batch_l1_value += loss_breakdown["loss_l1"]
                    batch_normal_value += loss_breakdown["loss_normal"]
                    batch_laplacian_value += loss_breakdown["loss_laplacian"]
                    batch_processed_samples += 1

                    if not printed_epoch_stats and batch_processed_samples == 1:
                        print(f"\n--- 🕵️ Debug Stats (Epoch {epoch+1}, 1st valid sample) ---")
                        print(f"  Sample: {sample.get('name', 'N/A')}")
                        print(f"  Verts_IN:  mean={V.mean():.4f}, std={V.std():.4f}, max_abs={V.abs().max():.4f}")
                        print(f"  Verts_OUT: mean={V_rec.mean():.4f}, std={V_rec.std():.4f}, max_abs={V_rec.abs().max():.4f}")
                        if torch.isfinite(Z_Global).all(): print(f"  Latent_Z:  mean={Z_Global.mean():.4f}, std={Z_Global.std():.4f}, max_abs={Z_Global.abs().max():.4f}")
                        else: print(f"  Latent_Z:  Contiene NaN/Inf!")
                        print(f"  Loss_Total: {loss_breakdown['loss_total']:.6f}")
                        print(f"  L1(raw): {loss_breakdown['loss_l1']:.6f} | Normal(raw): {loss_breakdown['loss_normal']:.6f} | LapCos(raw): {loss_breakdown['loss_laplacian']:.6f}")
                        print(f"  Cosine Similarity (mesh): {cosine_mesh:.4f}")
                        print(f"  Pearson Corr (mesh): {corr_mesh:.4f}")
                        print("-------------------------------------------------")

                        printed_epoch_stats = True

                except Exception as e:
                    import traceback
                    print(f"\n[ERRORE GRAVE] Eccezione loop interno: {e}. Campione: {sample.get('name', 'N/A')}")
                    # traceback.print_exc()
                    optimizer.zero_grad(set_to_none=True); batch_total_loss_value = 0.0; break

            if batch_processed_samples == 0: continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if not all(torch.isfinite(p).all() for p in model.parameters() if p.requires_grad):
                 print("\n[DISASTRO] NaN/Inf nei pesi DOPO step. Interrompo."); writer.close(); return

            batch_mean_total = batch_total_loss_value / batch_processed_samples
            batch_mean_l1 = batch_l1_value / batch_processed_samples
            batch_mean_normal = batch_normal_value / batch_processed_samples
            batch_mean_laplacian = batch_laplacian_value / batch_processed_samples

            epoch_loss_total += batch_mean_total
            epoch_loss_l1 += batch_mean_l1
            epoch_loss_normal += batch_mean_normal
            epoch_loss_laplacian += batch_mean_laplacian
            valid_batches += 1
            pbar.set_postfix(loss=f"{batch_mean_total:.4f} (L1:{batch_mean_l1:.4f}, N:{batch_mean_normal:.4f})")

        if valid_batches == 0: print(f"[WARN] Nessun batch valido epoca {epoch+1}."); continue

        train_loss_total = epoch_loss_total / valid_batches
        train_loss_l1 = epoch_loss_l1 / valid_batches
        train_loss_normal = epoch_loss_normal / valid_batches
        train_loss_laplacian = epoch_loss_laplacian / valid_batches

        # === Validazione ===
        model.eval()
        val_loss_total, val_loss_l1, val_loss_normal, val_loss_laplacian = 0.0, 0.0, 0.0, 0.0
        n_val_samples = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{EPOCHS}", dynamic_ncols=True, unit="sample", leave=False)
            for sample_list in val_pbar:
                if len(sample_list) == 0: continue
                sample = sample_list[0]
                try:
                    V = sample["verts"].to(device, non_blocking=PIN_MEMORY)
                    mass = sample["mass"].to(device, non_blocking=PIN_MEMORY)
                    evals = sample["evals"].to(device, non_blocking=PIN_MEMORY)
                    evecs = sample["evecs"].to(device, non_blocking=PIN_MEMORY)
                    faces = sample["faces"].to(device, non_blocking=PIN_MEMORY)
                    L = sample["L"].to(device)
                    gradX = sample["gradX"].to(device)
                    gradY = sample["gradY"].to(device)

                    V_rec, _ = model(V, mass, L, evals, evecs, faces, gradX, gradY)
                    _, loss_breakdown = criterion(V_rec, V, faces, L)

                    current_val_loss = loss_breakdown["loss_total"]
                    if not math.isfinite(current_val_loss):
                         print(f"[WARN] Loss non finita in val su {sample.get('name', 'N/A')}. Salto."); continue

                    val_loss_total += current_val_loss
                    val_loss_l1 += loss_breakdown["loss_l1"]
                    val_loss_normal += loss_breakdown["loss_normal"]
                    val_loss_laplacian += loss_breakdown["loss_laplacian"]
                    n_val_samples += 1
                except Exception as e:
                    print(f"\n[ERRORE GRAVE] Eccezione validazione: {e}. Campione: {sample.get('name', 'N/A')}")

        if n_val_samples == 0:
            print(f"[WARN] Nessun campione valido validazione epoca {epoch+1}.")
            val_loss_total = float('inf') # Imposta a inf per sicurezza scheduler
        else:
            val_loss_total /= n_val_samples
            val_loss_l1 /= n_val_samples
            val_loss_normal /= n_val_samples
            val_loss_laplacian /= n_val_samples

        if math.isfinite(val_loss_total): scheduler.step(val_loss_total)
        else: print("[WARN] Val loss infinita, step scheduler saltato.")
        current_lr = optimizer.param_groups[0]['lr']

        print(f"🧠 Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss_total:.6f} | Val Loss: {val_loss_total:.6f} | LR: {current_lr:.1e}")
        print(f"    Train (L1/N/L-cos): {train_loss_l1:.6f} / {train_loss_normal:.6f} / {train_loss_laplacian:.6f}")
        print(f"    Val   (L1/N/L-cos): {val_loss_l1:.6f} / {val_loss_normal:.6f} / {val_loss_laplacian:.6f}")

        try:
            with open(log_csv, "a") as f:
                f.write(f"{epoch+1},{train_loss_total:.6f},{val_loss_total:.6f},"
                        f"{train_loss_l1:.6f},{val_loss_l1:.6f},"
                        f"{train_loss_normal:.6f},{val_loss_normal:.6f},"
                        f"{train_loss_laplacian:.6f},{val_loss_laplacian:.6f},{current_lr:.1e}\n")
        except IOError as e: print(f"[ERRORE] Scrittura log CSV fallita: {e}")

        # Logga su TensorBoard
        writer.add_scalar("Loss_Total/train", train_loss_total, epoch+1)
        writer.add_scalar("Loss_Total/val", val_loss_total, epoch+1)
        writer.add_scalar("Learning_Rate", current_lr, epoch+1)
        writer.add_scalars("Loss_Breakdown_RAW/train", {'L1': train_loss_l1, 'Normal': train_loss_normal, 'Laplacian_Cos': train_loss_laplacian}, epoch+1)
        writer.add_scalars("Loss_Breakdown_RAW/val", {'L1': val_loss_l1, 'Normal': val_loss_normal, 'Laplacian_Cos': val_loss_laplacian}, epoch+1)

        # Salva checkpoint
        if (epoch + 1) % CHECKPOINT_EVERY == 0 or (epoch + 1) == EPOCHS:
            ckpt_path = os.path.join(OUT_DIR, f"diffusionAE_5000_epoch{epoch+1}.pth")
            try: torch.save(model.state_dict(), ckpt_path); print(f"💾 Saved checkpoint: {ckpt_path}")
            except Exception as e: print(f"[ERRORE] Salvataggio checkpoint fallito: {e}")

    writer.close()
    print("\n✅ Training + Validation completed")

if __name__ == "__main__":
    main()