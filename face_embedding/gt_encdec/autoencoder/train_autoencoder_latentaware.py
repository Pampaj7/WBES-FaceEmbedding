#!/usr/bin/env python3
import os
import math
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset
from diffusion_autoencoder import DiffusionAutoencoder
from geometric_loss import GeometricLoss

# helper per identity check sul subset
from helper import (
    patch_dataset_with_get_by_name,
    latent_identity_check,
    build_name_index_map,
)


def collate_skip(batch):
    """Drop None samples che il dataset potrebbe produrre."""
    return [s for s in batch if s is not None]


def main():
    # ============================================================
    # Multiprocessing setup
    # ============================================================
    try:
        current_start_method = mp.get_start_method(allow_none=True)
        if current_start_method is None:
            mp.set_start_method("spawn", force=True)
        elif current_start_method != "spawn":
            print(f"[WARN] Metodo start multiprocessing già impostato su '{current_start_method}'.")
    except RuntimeError:
        pass
    except Exception as e:
        print(f"[ERRORE] Errore imprevisto setup multiprocessing: {e}")

    # ============================================================
    # CONFIG
    # ============================================================
    DATA_DIR = "../../../datasets/GT_ready/npz_data_cropped_23470_with_ops/"

    # matrice GT normalizzata che usavi per l’identity check
    DIST_PATH = (
        "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/"
        "gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/"
        "normalized_matrix_distances.npz"
    )
        
    OUT_DIR = "test_latent_safe"

    LATENT_DIM = 256
    WIDTH = 128
    N_BLOCKS = 4
    EPOCHS = 50
    LR = 1e-4
    BATCH_SIZE = 64
    N_WORKERS = 0
    PIN_MEMORY = False
    VAL_SPLIT = 0.1
    CHECKPOINT_EVERY = 5

    # ogni quanti batch stampare il mega debug block
    DEBUG_EVERY = 25

    # pesi loss geometrica
    W_L1 = 0.3
    W_NORMAL = 1.0
    W_LAPLACIAN = 0.7

    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = False

    print(f"🚀 Training on {device} | logical batch={BATCH_SIZE} | LR (start)={LR}")
    print(f"🧬 Latent Dim={LATENT_DIM} | Width={WIDTH} | Blocks={N_BLOCKS}")
    print(f"⚖️ Pesi Loss: L1={W_L1} | Normal={W_NORMAL} | Laplacian={W_LAPLACIAN}")
    print(f"💾 Using NPZ dataset from: {DATA_DIR}")
    print(f"⚙️ DataLoader: num_workers={N_WORKERS}, pin_memory={PIN_MEMORY}")

    # ============================================================
    # Dataset
    # ============================================================
    dataset = GTReadyDataset(DATA_DIR)
    dataset = patch_dataset_with_get_by_name(dataset)

    # subset per debug / velocità
    dataset.files = dataset.files[:3000]
    print(f"🧩 Using subset of {len(dataset.files)} meshes")

    n_samples = len(dataset)
    if n_samples == 0:
        print("[ERRORE] Dataset NPZ vuoto.")
        return

    n_val = int(n_samples * VAL_SPLIT)
    n_train = n_samples - n_val
    if n_train <= 0 or n_val <= 0:
        print(f"[ERRORE] Dataset troppo piccolo ({n_samples}) per split.")
        return

    try:
        train_set, val_set = random_split(dataset, [n_train, n_val])
    except Exception as e:
        print(f"[ERRORE] Fallimento random_split: {e}")
        return

    print(f"📚 Split: {len(train_set)} train / {len(val_set)} val")

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=N_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_skip,
        persistent_workers=True if N_WORKERS > 0 else False,
        prefetch_factor=1 if N_WORKERS > 0 else None,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=N_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_skip,
        persistent_workers=True if N_WORKERS > 0 else False,
        prefetch_factor=1 if N_WORKERS > 0 else None,
    )

    # ============================================================
    # Modello + Loss + Optimizer
    # ============================================================
    model = DiffusionAutoencoder(
        latent_dim=LATENT_DIM, width=WIDTH, n_blocks=N_BLOCKS
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7
    )

    criterion = GeometricLoss(
        w_l1=W_L1, w_normal=W_NORMAL, w_laplacian=W_LAPLACIAN, device=device
    ).to(device)

    # ============================================================
    # GT distance matrix per identity check
    # ============================================================
    print("📂 Loading GT distance matrix (per identity check)...")
    D_pack = np.load(DIST_PATH, allow_pickle=True)
    D_orig = D_pack["D_orig"].astype(np.float64)
    # normalizzazione come in latent-aware
    norm_factor = np.max(D_orig[D_orig > 0])
    D_orig = D_orig / norm_factor
    name_to_idx = build_name_index_map(D_pack["names"])

    FIXED_N = min(100, len(dataset.files))
    fixed_names = dataset.files[:FIXED_N]

    fixed_idx = []
    for nm in fixed_names:
        base = nm[:-4] if nm.endswith(".npz") else nm
        if base in name_to_idx:
            fixed_idx.append(name_to_idx[base])
    fixed_idx = np.array(fixed_idx)

    if len(fixed_idx) >= 2:
        D_ref = D_orig[np.ix_(fixed_idx, fixed_idx)]
    else:
        print("⚠️ STRICT latent/identity validation disabilitata (meno di 2 match).")
        fixed_names = None
        D_ref = None

    # ============================================================
    # Logging
    # ============================================================
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(OUT_DIR, "runs", run_name)
    try:
        writer = SummaryWriter(log_dir=log_dir)
        print(f"📊 TensorBoard logs salvati in: {log_dir}")
    except Exception as e:
        print(f"[ERRORE] Creazione SummaryWriter fallita: {e}")
        return

    log_csv = os.path.join(OUT_DIR, "train_log.csv")
    try:
        with open(log_csv, "w") as f:
            f.write(
                "epoch,"
                "train_loss,val_loss,"
                "train_l1,val_l1,"
                "train_normal,val_normal,"
                "train_laplacian,val_laplacian,"
                "train_cos_mesh,val_cos_mesh,"
                "train_pear_mesh,val_pear_mesh,"
                "ident_pearson,ident_spearman,ident_r2,ident_slope,"
                "current_lr\n"
            )
    except IOError as e:
        print(f"[ERRORE] Scrittura log CSV fallita: {e}")
        return

    # ============================================================
    # Training Loop
    # ============================================================
    print("\n--- Inizio Training ---")

    for epoch in range(EPOCHS):
        model.train()

        epoch_loss_total = 0.0
        epoch_loss_l1 = 0.0
        epoch_loss_normal = 0.0
        epoch_loss_laplacian = 0.0
        valid_batches = 0

        # per medie di cosine / pearson mesh (train)
        train_cos_sum = 0.0
        train_pear_sum = 0.0
        train_corr_count = 0

        pbar = tqdm(
            enumerate(train_loader),
            desc=f"Epoch {epoch+1}/{EPOCHS}",
            dynamic_ncols=True,
            unit="batch",
        )

        for batch_idx, batch_list in pbar:
            current_lr_pbar = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(lr=f"{current_lr_pbar:.1e}")

            if len(batch_list) == 0:
                continue

            optimizer.zero_grad(set_to_none=True)

            batch_total_loss_value = 0.0
            batch_l1_value = 0.0
            batch_normal_value = 0.0
            batch_laplacian_value = 0.0
            batch_processed_samples = 0

            debug_this_batch = (batch_idx % DEBUG_EVERY == 0)

            for i, sample in enumerate(batch_list):
                try:
                    V = sample["verts"].to(device, non_blocking=PIN_MEMORY)
                    mass = sample["mass"].to(device, non_blocking=PIN_MEMORY)
                    evals = sample["evals"].to(device, non_blocking=PIN_MEMORY)
                    evecs = sample["evecs"].to(device, non_blocking=PIN_MEMORY)
                    faces = sample["faces"].to(device, non_blocking=PIN_MEMORY)
                    L = sample["L"].to(device)
                    gradX = sample["gradX"].to(device)
                    gradY = sample["gradY"].to(device)

                    V_rec, Z_Global = model(
                        V, mass, L, evals, evecs, faces, gradX, gradY
                    )
                    loss, loss_breakdown = criterion(V_rec, V, faces, L)

                    # metriche di similarità mesh (solo logging)
                    try:
                        cosine_mesh = F.cosine_similarity(
                            V_rec.flatten(), V.flatten(), dim=0
                        ).item()
                        corr_matrix = torch.corrcoef(
                            torch.stack([V_rec.flatten(), V.flatten()])
                        )
                        corr_mesh = corr_matrix[0, 1].item()
                    except Exception:
                        cosine_mesh, corr_mesh = float("nan"), float("nan")

                    if math.isfinite(cosine_mesh) and math.isfinite(corr_mesh):
                        train_cos_sum += float(cosine_mesh)
                        train_pear_sum += float(corr_mesh)
                        train_corr_count += 1

                    if not torch.isfinite(loss):
                        print(
                            f"\n[ERRORE] Loss non finita campione {sample.get('name', 'N/A')}. Salto."
                        )
                        continue

                    loss_scaled = loss / len(batch_list)
                    loss_scaled.backward()

                    batch_total_loss_value += loss_breakdown["loss_total"]
                    batch_l1_value += loss_breakdown["loss_l1"]
                    batch_normal_value += loss_breakdown["loss_normal"]
                    batch_laplacian_value += loss_breakdown["loss_laplacian"]
                    batch_processed_samples += 1

                    # --- DEBUG BLOCK ogni DEBUG_EVERY batch (sul primo sample del batch)
                    if debug_this_batch and i == 0:
                        print(
                            f"\n--- 🕵️ Debug Stats (Epoch {epoch+1}, batch {batch_idx+1}) ---"
                        )
                        print(f"  Sample: {sample.get('name', 'N/A')}")
                        print(
                            f"  Verts_IN:  mean={V.mean():.4f}, std={V.std():.4f}, "
                            f"max_abs={V.abs().max():.4f}"
                        )
                        print(
                            f"  Verts_OUT: mean={V_rec.mean():.4f}, std={V_rec.std():.4f}, "
                            f"max_abs={V_rec.abs().max():.4f}"
                        )
                        if torch.isfinite(Z_Global).all():
                            print(
                                f"  Latent_Z:  mean={Z_Global.mean():.4f}, "
                                f"std={Z_Global.std():.4f}, "
                                f"max_abs={Z_Global.abs().max():.4f}"
                            )
                        else:
                            print("  Latent_Z:  Contiene NaN/Inf!")

                        print(f"  Loss_Total: {loss_breakdown['loss_total']:.6f}")
                        print(
                            f"  L1(raw): {loss_breakdown['loss_l1']:.6f} | "
                            f"Normal(raw): {loss_breakdown['loss_normal']:.6f} | "
                            f"LapCos(raw): {loss_breakdown['loss_laplacian']:.6f}"
                        )
                        print(
                            f"  Cosine Similarity (mesh): {cosine_mesh:.4f}"
                        )
                        print(f"  Pearson Corr (mesh): {corr_mesh:.4f}")
                        print("-------------------------------------------------")

                except Exception as e:
                    print(
                        f"\n[ERRORE GRAVE] Eccezione loop interno: {e}. "
                        f"Campione: {sample.get('name', 'N/A')}"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    batch_total_loss_value = 0.0
                    break

            if batch_processed_samples == 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # check NaN/Inf nei pesi
            if not all(
                torch.isfinite(p).all()
                for p in model.parameters()
                if p.requires_grad
            ):
                print("\n[DISASTRO] NaN/Inf nei pesi DOPO step. Interrompo.")
                writer.close()
                return

            batch_mean_total = batch_total_loss_value / batch_processed_samples
            batch_mean_l1 = batch_l1_value / batch_processed_samples
            batch_mean_normal = batch_normal_value / batch_processed_samples
            batch_mean_laplacian = batch_laplacian_value / batch_processed_samples

            epoch_loss_total += batch_mean_total
            epoch_loss_l1 += batch_mean_l1
            epoch_loss_normal += batch_mean_normal
            epoch_loss_laplacian += batch_mean_laplacian
            valid_batches += 1

            pbar.set_postfix(
                loss=f"{batch_mean_total:.4f} (L1:{batch_mean_l1:.4f}, N:{batch_mean_normal:.4f})",
                lr=f"{current_lr_pbar:.1e}",
            )

        if valid_batches == 0:
            print(f"[WARN] Nessun batch valido epoca {epoch+1}.")
            continue

        train_loss_total = epoch_loss_total / valid_batches
        train_loss_l1 = epoch_loss_l1 / valid_batches
        train_loss_normal = epoch_loss_normal / valid_batches
        train_loss_laplacian = epoch_loss_laplacian / valid_batches

        if train_corr_count > 0:
            train_cos_avg = train_cos_sum / train_corr_count
            train_pear_avg = train_pear_sum / train_corr_count
        else:
            train_cos_avg = float("nan")
            train_pear_avg = float("nan")

        # ========================================================
        # Validazione (anche qui cosine / pearson sulle mesh)
        # ========================================================
        model.eval()
        val_loss_total = 0.0
        val_loss_l1 = 0.0
        val_loss_normal = 0.0
        val_loss_laplacian = 0.0
        n_val_samples = 0

        val_cos_sum = 0.0
        val_pear_sum = 0.0
        val_corr_count = 0

        with torch.no_grad():
            val_pbar = tqdm(
                val_loader,
                desc=f"Validation Epoch {epoch+1}/{EPOCHS}",
                dynamic_ncols=True,
                unit="sample",
                leave=False,
            )
            for sample_list in val_pbar:
                if len(sample_list) == 0:
                    continue
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

                    V_rec, _ = model(
                        V, mass, L, evals, evecs, faces, gradX, gradY
                    )
                    _, loss_breakdown = criterion(V_rec, V, faces, L)

                    current_val_loss = loss_breakdown["loss_total"]
                    if not math.isfinite(current_val_loss):
                        print(
                            f"[WARN] Loss non finita in val su {sample.get('name', 'N/A')}. Salto."
                        )
                        continue

                    val_loss_total += current_val_loss
                    val_loss_l1 += loss_breakdown["loss_l1"]
                    val_loss_normal += loss_breakdown["loss_normal"]
                    val_loss_laplacian += loss_breakdown["loss_laplacian"]
                    n_val_samples += 1

                    # cosine / pearson mesh sulla val
                    try:
                        cosine_mesh = F.cosine_similarity(
                            V_rec.flatten(), V.flatten(), dim=0
                        ).item()
                        corr_matrix = torch.corrcoef(
                            torch.stack([V_rec.flatten(), V.flatten()])
                        )
                        corr_mesh = corr_matrix[0, 1].item()
                    except Exception:
                        cosine_mesh, corr_mesh = float("nan"), float("nan")

                    if math.isfinite(cosine_mesh) and math.isfinite(corr_mesh):
                        val_cos_sum += float(cosine_mesh)
                        val_pear_sum += float(corr_mesh)
                        val_corr_count += 1

                except Exception as e:
                    print(
                        f"\n[ERRORE GRAVE] Eccezione validazione: {e}. "
                        f"Campione: {sample.get('name', 'N/A')}"
                    )

        if n_val_samples == 0:
            print(f"[WARN] Nessun campione valido validazione epoca {epoch+1}.")
            val_loss_total = float("inf")
        else:
            val_loss_total /= n_val_samples
            val_loss_l1 /= n_val_samples
            val_loss_normal /= n_val_samples
            val_loss_laplacian /= n_val_samples

        if val_corr_count > 0:
            val_cos_avg = val_cos_sum / val_corr_count
            val_pear_avg = val_pear_sum / val_corr_count
        else:
            val_cos_avg = float("nan")
            val_pear_avg = float("nan")

        if math.isfinite(val_loss_total):
            scheduler.step(val_loss_total)
        else:
            print("[WARN] Val loss infinita, step scheduler saltato.")
        current_lr = optimizer.param_groups[0]["lr"]

        # ========================================================
        # Identity check (pairwise) su subset fisso
        # ========================================================
        if fixed_names is not None and D_ref is not None:
            stats = latent_identity_check(
                model, dataset, fixed_names, fixed_idx, D_ref, device
            )
            ident_pear = float(stats.get("pearson", np.nan)) if stats else float("nan")
            ident_spear = float(stats.get("spearman", np.nan)) if stats else float("nan")
            ident_r2 = float(stats.get("r2", np.nan)) if stats else float("nan")
            ident_slope = float(stats.get("slope", np.nan)) if stats else float("nan")
        else:
            stats = None
            ident_pear = ident_spear = ident_r2 = ident_slope = float("nan")

        # ========================================================
        # PRINT EPOCH SUMMARY (multi-line, leggibile)
        # ========================================================
        print(f"🧠 Epoch {epoch+1}/{EPOCHS}")
        print(f"    Train Loss: {train_loss_total:.6f} | Val Loss: {val_loss_total:.6f} | LR: {current_lr:.1e}")
        print(f"    Train (L1/N/L-cos): {train_loss_l1:.6f} / {train_loss_normal:.6f} / {train_loss_laplacian:.6f}")
        print(f"    Val   (L1/N/L-cos): {val_loss_l1:.6f} / {val_loss_normal:.6f} / {val_loss_laplacian:.6f}")
        print(f"    Train mesh cos/pear: {train_cos_avg:.6f} / {train_pear_avg:.6f}")
        print(f"    Val   mesh cos/pear: {val_cos_avg:.6f} / {val_pear_avg:.6f}")
        print(
            f"    Identity (pear/spear/R2/slope): "
            f"{ident_pear:.4f} / {ident_spear:.4f} / {ident_r2:.4f} / {ident_slope:.4f}"
        )

        # ========================================================
        # CSV logging
        # ========================================================
        try:
            with open(log_csv, "a") as f:
                f.write(
                    f"{epoch+1},"
                    f"{train_loss_total:.6f},{val_loss_total:.6f},"
                    f"{train_loss_l1:.6f},{val_loss_l1:.6f},"
                    f"{train_loss_normal:.6f},{val_loss_normal:.6f},"
                    f"{train_loss_laplacian:.6f},{val_loss_laplacian:.6f},"
                    f"{train_cos_avg:.6f},{val_cos_avg:.6f},"
                    f"{train_pear_avg:.6f},{val_pear_avg:.6f},"
                    f"{ident_pear:.4f},{ident_spear:.4f},{ident_r2:.4f},{ident_slope:.4f},"
                    f"{current_lr:.1e}\n"
                )
        except IOError as e:
            print(f"[ERRORE] Scrittura log CSV fallita: {e}")

        # TensorBoard
        writer.add_scalar("Loss_Total/train", train_loss_total, epoch + 1)
        writer.add_scalar("Loss_Total/val", val_loss_total, epoch + 1)
        writer.add_scalar("Learning_Rate", current_lr, epoch + 1)
        writer.add_scalars(
            "Loss_Breakdown_RAW/train",
            {
                "L1": train_loss_l1,
                "Normal": train_loss_normal,
                "Laplacian_Cos": train_loss_laplacian,
            },
            epoch + 1,
        )
        writer.add_scalars(
            "Loss_Breakdown_RAW/val",
            {
                "L1": val_loss_l1,
                "Normal": val_loss_normal,
                "Laplacian_Cos": val_loss_laplacian,
            },
            epoch + 1,
        )

        # Checkpoint
        if (epoch + 1) % CHECKPOINT_EVERY == 0 or (epoch + 1) == EPOCHS:
            ckpt_path = os.path.join(
                OUT_DIR, f"diffusionAE_5000_epoch{epoch+1}.pth"
            )
            try:
                torch.save(model.state_dict(), ckpt_path)
                print(f"💾 Saved checkpoint: {ckpt_path}")
            except Exception as e:
                print(f"[ERRORE] Salvataggio checkpoint fallito: {e}")

    writer.close()
    print("\n✅ Training + Validation completed")


if __name__ == "__main__":
    main()
