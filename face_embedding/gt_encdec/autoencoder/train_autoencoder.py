import torch
import os
from torch.utils.data import DataLoader, random_split
from dataset_gtready import GTReadyDataset
from diffusion_autoencoder import DiffusionAutoencoder
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
# Importiamo lo scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import math # Per math.isfinite

# Importiamo la loss pura, senza dipendenze
from geometric_loss import GeometricLoss

def collate_skip(batch):
    # Filtra i campioni che sono None
    return [s for s in batch if s is not None]

def main():
    try:
        # Tenta di impostare il metodo di start per multiprocessing
        # 'spawn' è generalmente più sicuro con CUDA
        current_start_method = mp.get_start_method(allow_none=True)
        if current_start_method is None:
             mp.set_start_method("spawn", force=True)
        elif current_start_method != "spawn":
             print(f"[WARN] Metodo start multiprocessing già impostato su '{current_start_method}'. Potrebbero esserci problemi con CUDA se non è 'spawn'.")
             # Non forzare se già impostato, potrebbe causare errori
             # mp.set_start_method("spawn", force=True)

    except RuntimeError as e:
         # Potrebbe dare errore se chiamato più volte o in contesti non supportati
         # print(f"[INFO] Impossibile impostare start_method 'spawn': {e}")
         pass
    except Exception as e: # Cattura altre eccezioni impreviste
         print(f"[ERRORE] Errore imprevisto durante l'impostazione di multiprocessing: {e}")


    DATA_DIR = "../../../datasets/GT_ready/"
    OPS_DIR = os.path.join(DATA_DIR, "operators")
    OUT_DIR = "./results_diffusionAE/"


    N_WORKERS = 20 # Inizia con un valore basso/moderato
    PIN_MEMORY = torch.cuda.is_available() # Abilitiamo pin_memory con workers > 0    
    
    # --- IPERPARAMETRI (Ultima configurazione testata) ---
    LATENT_DIM = 256
    WIDTH = 128
    N_BLOCKS = 4

    EPOCHS = 20 # O più
    LR = 1e-4 # Iniziamo alti
    BATCH_SIZE = 16
    VAL_SPLIT = 0.1
    CHECKPOINT_EVERY = 5

    # Pesi finali
    W_L1 = 1.0
    W_NORMAL = 1.0
    W_LAPLACIAN = 0.1
    
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Abilita CUDNN benchmark solo se CUDA è disponibile e ci sono GPUs fisse
    if device.type == 'cuda' and torch.cuda.device_count() > 0:
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = False

    print(f"🚀 Training on {device} | logical batch={BATCH_SIZE} | LR (start)={LR}")
    print(f"🧬 Latent Dim={LATENT_DIM} | Width={WIDTH} | Blocks={N_BLOCKS}")
    print(f"⚖️ Pesi Loss: L1={W_L1} | Normal={W_NORMAL} | Laplacian={W_LAPLACIAN}")

    dataset = GTReadyDataset(DATA_DIR, ops_dir=OPS_DIR)

    # Considera di non limitare il dataset se possibile
    dataset.files = dataset.files[:1000] # O usa l'intero dataset
    print(f"🧩 Using subset of {len(dataset.files)} meshes (potenziali None inclusi)")

    n_samples = len(dataset)
    if n_samples == 0:
        print("[ERRORE] Dataset vuoto o nessun file .obj trovato.")
        return

    n_val = int(n_samples * VAL_SPLIT)
    n_train = n_samples - n_val

    # Assicurati che ci siano abbastanza campioni per entrambi gli split
    if n_train <= 0 or n_val <= 0:
        print(f"[ERRORE] Dataset troppo piccolo ({n_samples} campioni) per lo split richiesto. "
              f"Train: {n_train}, Val: {n_val}. Riduci VAL_SPLIT o aumenta il dataset.")
        return

    try:
        train_set, val_set = random_split(dataset, [n_train, n_val])
    except Exception as e:
        print(f"[ERRORE] Fallimento in random_split: {e}")
        return
    print(f"📚 Split: {len(train_set)} train / {len(val_set)} val")



    # num_workers=0 è spesso più sicuro con dati complessi caricati da pickle/sparse
    # pin_memory=True può dare un piccolo boost se la memoria CPU lo permette, ma False è più sicuro
    
    print(f"DataLoader: num_workers={N_WORKERS}, pin_memory={PIN_MEMORY}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=N_WORKERS,
                              pin_memory=PIN_MEMORY,
                              collate_fn=collate_skip,
                              # persistent_workers e prefetch_factor possono aiutare, aggiungili se N_WORKERS > 0 funziona
                              persistent_workers=True if N_WORKERS > 0 else False,
                              prefetch_factor=2 if N_WORKERS > 0 else None
                              )

    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            # Usa meno worker per val se vuoi, es. la metà
                            num_workers=max(1, N_WORKERS // 2),
                            pin_memory=PIN_MEMORY,
                            collate_fn=collate_skip,
                            persistent_workers=True if N_WORKERS > 0 else False,
                            prefetch_factor=2 if N_WORKERS > 0 else None
                            )
    
    model = DiffusionAutoencoder(
        latent_dim=LATENT_DIM,
        width=WIDTH,
        n_blocks=N_BLOCKS
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)

    # 🌟 INIZIALIZZA LO SCHEDULER (senza verbose)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5, # Dimezza LR
        patience=2, # Dopo 2 epoche senza miglioramento Val Loss
        # verbose=True, # ⬅️ RIMOSSO
        min_lr=1e-7  # Limite inferiore per LR
    )

    criterion = GeometricLoss(
        w_l1=W_L1,
        w_normal=W_NORMAL,
        w_laplacian=W_LAPLACIAN,
        device=device
    ).to(device)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(OUT_DIR, "runs", run_name)
    try:
        writer = SummaryWriter(log_dir=log_dir)
        print(f"📊 TensorBoard logs salvati in: {log_dir}")
    except Exception as e:
        print(f"[ERRORE] Impossibile creare SummaryWriter in {log_dir}: {e}")
        return # Non proseguire se non possiamo loggare


    log_csv = os.path.join(OUT_DIR, "train_log.csv")
    try:
        with open(log_csv, "w") as f:
            f.write("epoch,train_loss,val_loss,train_l1,val_l1,train_normal,val_normal,train_laplacian,current_lr\n")
    except IOError as e:
        print(f"[ERRORE] Impossibile scrivere il file di log CSV {log_csv}: {e}")
        return

    # === Ciclo di Training ===
    print(f"\n--- Inizio Training ---")
    for epoch in range(EPOCHS):
        model.train() # Imposta il modello in modalità training

        # Accumulatori per le medie dell'epoca
        epoch_loss_total, epoch_loss_l1, epoch_loss_normal, epoch_loss_laplacian = 0.0, 0.0, 0.0, 0.0
        valid_batches = 0
        printed_epoch_stats = False

        # Barra di progresso per l'epoca
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", dynamic_ncols=True, unit="batch")

        for batch_list in pbar:
            # Mostra LR corrente nel pbar all'inizio di ogni batch
            current_lr_pbar = optimizer.param_groups[0]['lr']
            pbar.set_postfix(lr=f"{current_lr_pbar:.1e}")

            if len(batch_list) == 0: continue # Salta batch vuoti (filtrati da collate_fn)

            optimizer.zero_grad(set_to_none=True) # Azzera gradienti (più efficiente con set_to_none=True)

            # Accumulatori per il batch (per media e gestione errori)
            batch_total_loss_value, batch_l1_value, batch_normal_value, batch_laplacian_value = 0.0, 0.0, 0.0, 0.0
            batch_processed_samples = 0 # Contatore campioni validi nel batch

            # Loop sui campioni nel batch (accumulazione gradienti)
            for i, sample in enumerate(batch_list):
                try:
                    # Sposta i dati su GPU (o CPU se CUDA non è disponibile)
                    V = sample["verts"].to(device, non_blocking=True)
                    mass = sample["mass"].to(device, non_blocking=True)
                    evals = sample["evals"].to(device, non_blocking=True)
                    evecs = sample["evecs"].to(device, non_blocking=True)
                    faces = sample["faces"].to(device, non_blocking=True)
                    L = sample["L"].to(device) # Gli sparsi potrebbero non beneficiare di non_blocking
                    gradX = sample["gradX"].to(device)
                    gradY = sample["gradY"].to(device)

                    # Forward pass
                    V_rec, Z_Global = model(V, mass, L, evals, evecs, faces, gradX, gradY)

                    # Calcolo Loss
                    loss, loss_breakdown = criterion(V_rec, V, faces, L)

                    # Controllo stabilità loss prima del backward
                    if not torch.isfinite(loss):
                        print(f"\n[ERRORE] Loss non finita (NaN/Inf) nel campione {sample.get('name', 'N/A')}. Salto campione.")
                        # Non interrompere il batch, salta solo questo campione
                        continue # Passa al prossimo campione nel batch

                    # Scalatura loss per media batch e backward pass
                    loss_scaled = loss / len(batch_list)
                    loss_scaled.backward() # Accumula gradienti

                    # Accumula valori della loss (non scalati) per il logging
                    batch_total_loss_value += loss_breakdown["loss_total"]
                    batch_l1_value += loss_breakdown["loss_l1"]
                    batch_normal_value += loss_breakdown["loss_normal"]
                    batch_laplacian_value += loss_breakdown["loss_laplacian"]
                    batch_processed_samples += 1 # Incrementa contatore campioni validi

                    # Stampa statistiche del primo campione valido della prima epoca
                    if not printed_epoch_stats and batch_processed_samples == 1:
                        print(f"\n--- 🕵️ Debug Stats (Epoch {epoch+1}, 1st valid sample) ---")
                        print(f"  Sample: {sample.get('name', 'N/A')}")
                        print(f"  Verts_IN:  mean={V.mean():.4f}, std={V.std():.4f}, max_abs={V.abs().max():.4f}")
                        print(f"  Verts_OUT: mean={V_rec.mean():.4f}, std={V_rec.std():.4f}, max_abs={V_rec.abs().max():.4f}")
                        # Controlla se Z_Global è valido prima di calcolare stats
                        if torch.isfinite(Z_Global).all():
                             print(f"  Latent_Z:  mean={Z_Global.mean():.4f}, std={Z_Global.std():.4f}, max_abs={Z_Global.abs().max():.4f}")
                        else:
                             print(f"  Latent_Z:  Contiene NaN/Inf!")
                        print(f"  Loss_Total: {loss_breakdown['loss_total']:.6f}")
                        print(f"  L1(raw): {loss_breakdown['loss_l1']:.6f} | Normal(raw): {loss_breakdown['loss_normal']:.6f} | LapCos(raw): {loss_breakdown['loss_laplacian']:.6f}")
                        print("-------------------------------------------------")
                        printed_epoch_stats = True

                except Exception as e:
                    # Gestione eccezioni più robusta
                    import traceback
                    print(f"\n[ERRORE GRAVE] Eccezione non gestita nel loop interno: {e}. Campione: {sample.get('name', 'N/A')}")
                    # traceback.print_exc() # Uncommenta per debug dettagliato
                    # Annulla i gradienti accumulati finora per questo batch
                    optimizer.zero_grad(set_to_none=True)
                    batch_total_loss_value = 0.0 # Segnala che il batch è fallito
                    break # Interrompe il loop interno per questo batch

            # Se nessun campione valido è stato processato nel batch, salta lo step
            if batch_processed_samples == 0:
                # print(f"[WARN] Nessun campione valido processato nel batch corrente.") # Opzionale: logga batch saltati
                continue

            # Applica Gradient Clipping *dopo* aver accumulato tutti i gradienti del batch
            # Utile se i gradienti tendono ad esplodere
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Esegui lo step dell'ottimizzatore (aggiorna i pesi)
            optimizer.step()

            # Controllo post-step: verifica se i pesi sono ancora finiti
            if not all(torch.isfinite(p).all() for p in model.parameters() if p.requires_grad):
                 print("\n[DISASTRO] Rilevati NaN/Inf nei pesi del modello DOPO optimizer.step(). Interrompo il training.")
                 writer.close() # Chiudi writer prima di uscire
                 return # Esce dalla funzione main()

            # Calcola le medie del batch basandosi sui campioni processati
            batch_mean_total = batch_total_loss_value / batch_processed_samples
            batch_mean_l1 = batch_l1_value / batch_processed_samples
            batch_mean_normal = batch_normal_value / batch_processed_samples
            batch_mean_laplacian = batch_laplacian_value / batch_processed_samples

            # Aggiorna le somme totali dell'epoca
            epoch_loss_total += batch_mean_total
            epoch_loss_l1 += batch_mean_l1
            epoch_loss_normal += batch_mean_normal
            epoch_loss_laplacian += batch_mean_laplacian

            valid_batches += 1 # Incrementa contatore batch validi
            # Aggiorna pbar con le medie del batch
            pbar.set_postfix(loss=f"{batch_mean_total:.4f} (L1:{batch_mean_l1:.4f}, N:{batch_mean_normal:.4f})")
            # Fine loop sui batch

        # Calcola le medie dell'epoca di training
        if valid_batches == 0:
            print(f"[WARN] Nessun batch valido completato nell'epoca {epoch+1}. Salto epoca.")
            continue # Passa alla prossima epoca

        train_loss_total = epoch_loss_total / valid_batches
        train_loss_l1 = epoch_loss_l1 / valid_batches
        train_loss_normal = epoch_loss_normal / valid_batches
        train_loss_laplacian = epoch_loss_laplacian / valid_batches

        # === Ciclo di Validazione ===
        model.eval() # Imposta il modello in modalità valutazione
        val_loss_total, val_loss_l1, val_loss_normal, val_loss_laplacian = 0.0, 0.0, 0.0, 0.0
        n_val_samples = 0

        # Disabilita calcolo gradienti per la validazione
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{EPOCHS}", dynamic_ncols=True, unit="sample", leave=False)
            for sample_list in val_pbar:
                if len(sample_list) == 0: continue
                sample = sample_list[0] # Batch size = 1

                try:
                    # Sposta dati val su device
                    V = sample["verts"].to(device, non_blocking=True)
                    mass = sample["mass"].to(device, non_blocking=True)
                    evals = sample["evals"].to(device, non_blocking=True)
                    evecs = sample["evecs"].to(device, non_blocking=True)
                    faces = sample["faces"].to(device, non_blocking=True)
                    L = sample["L"].to(device)
                    gradX = sample["gradX"].to(device)
                    gradY = sample["gradY"].to(device)

                    # Forward pass val
                    V_rec, _ = model(V, mass, L, evals, evecs, faces, gradX, gradY)
                    # Calcolo loss val
                    _, loss_breakdown = criterion(V_rec, V, faces, L)

                    # Controllo loss val finita
                    current_val_loss = loss_breakdown["loss_total"]
                    if not math.isfinite(current_val_loss):
                         print(f"[WARN] Loss non finita in val su {sample.get('name', 'N/A')}. Salto campione.")
                         continue # Salta questo campione

                    # Accumula loss val
                    val_loss_total += current_val_loss
                    val_loss_l1 += loss_breakdown["loss_l1"]
                    val_loss_normal += loss_breakdown["loss_normal"]
                    val_loss_laplacian += loss_breakdown["loss_laplacian"]
                    n_val_samples += 1 # Incrementa solo se il campione è valido

                except Exception as e:
                    import traceback
                    print(f"\n[ERRORE GRAVE] Eccezione non gestita in validazione: {e}. Campione: {sample.get('name', 'N/A')}")
                    # traceback.print_exc() # Uncommenta per debug

        # Calcola medie validazione
        if n_val_samples == 0:
            print(f"[WARN] Nessun campione valido nella validazione dell'epoca {epoch+1}.")
            # Cosa fare? Possiamo assegnare loss infinite o saltare lo step dello scheduler?
            # Per ora, assegniamo valori alti per evitare riduzioni LR errate
            val_loss_total = float('inf')
            val_loss_l1 = float('inf')
            val_loss_normal = float('inf')
            val_loss_laplacian = float('inf')
        else:
            val_loss_total /= n_val_samples
            val_loss_l1 /= n_val_samples
            val_loss_normal /= n_val_samples
            val_loss_laplacian /= n_val_samples

        # PASSA LA VAL LOSS ALLO SCHEDULER (solo se finita)
        if math.isfinite(val_loss_total):
            scheduler.step(val_loss_total)
        else:
            print("[WARN] Val loss infinita, step dello scheduler saltato.")

        current_lr = optimizer.param_groups[0]['lr'] # Prendi il LR attuale (potrebbe essere cambiato)

        # Stampa statistiche epoca
        print(f"🧠 Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss_total:.6f} | Val Loss: {val_loss_total:.6f} | LR: {current_lr:.1e}")
        print(f"    Train (L1/N/L-cos): {train_loss_l1:.6f} / {train_loss_normal:.6f} / {train_loss_laplacian:.6f}")
        print(f"    Val   (L1/N/L-cos): {val_loss_l1:.6f} / {val_loss_normal:.6f} / {val_loss_laplacian:.6f}")

        # Scrivi log CSV
        try:
            with open(log_csv, "a") as f:
                f.write(f"{epoch+1},{train_loss_total:.6f},{val_loss_total:.6f},"
                        f"{train_loss_l1:.6f},{val_loss_l1:.6f},"
                        f"{train_loss_normal:.6f},{val_loss_normal:.6f},"
                        f"{train_loss_laplacian:.6f},{val_loss_laplacian:.6f},{current_lr:.1e}\n")
        except IOError as e:
            print(f"[ERRORE] Impossibile scrivere nel log CSV: {e}")


        # Logga su TensorBoard
        writer.add_scalar("Loss_Total/train", train_loss_total, epoch+1)
        writer.add_scalar("Loss_Total/val", val_loss_total, epoch+1)
        writer.add_scalar("Learning_Rate", current_lr, epoch+1)
        # Logga componenti loss raw (non pesate)
        writer.add_scalars("Loss_Breakdown_RAW/train", {
            'L1': train_loss_l1, 'Normal': train_loss_normal, 'Laplacian_Cos': train_loss_laplacian
        }, epoch+1)
        writer.add_scalars("Loss_Breakdown_RAW/val", {
            'L1': val_loss_l1, 'Normal': val_loss_normal, 'Laplacian_Cos': val_loss_laplacian
        }, epoch+1)

        # Salva checkpoint
        if (epoch + 1) % CHECKPOINT_EVERY == 0 or (epoch + 1) == EPOCHS:
            ckpt_path = os.path.join(OUT_DIR, f"diffusionAE_epoch{epoch+1}.pth")
            try:
                torch.save(model.state_dict(), ckpt_path)
                print(f"💾 Saved checkpoint: {ckpt_path}")
            except Exception as e:
                print(f"[ERRORE] Impossibile salvare il checkpoint {ckpt_path}: {e}")
        # Fine loop sulle epoche

    writer.close()
    print("\n✅ Training + Validation completed")

if __name__ == "__main__":
    main()