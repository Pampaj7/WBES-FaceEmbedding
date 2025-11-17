#!/usr/bin/env python3
import os
import math
from datetime import datetime

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset
from diffusion_autoencoder import DiffusionEncoderOnly
from latent_loss import stress_loss, scale_loss

from helper import (
    patch_dataset_with_get_by_name,
    latent_identity_check,
    build_name_index_map,
    collate_skip,
)

# ========================================================================
# TRAINING ENCODER-ONLY (NO DECODER, NO GEOMETRIC LOSS, NO RECON)
# ========================================================================

def main():

    # multiprocessing
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # paths
    DATA_DIR = "../../../datasets/GT_ready/npz_data_cropped_23470_with_ops/"
    DIST_PATH = (
        "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/"
        "gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/"
        "normalized_matrix_distances.npz"
    )
    OUT_DIR = "encoder_only"
    os.makedirs(OUT_DIR, exist_ok=True)

    # model config
    LATENT_DIM = 256
    WIDTH = 128
    N_BLOCKS = 4

    # training config
    EPOCHS = 50
    LR = 1e-4
    BATCH_SIZE = 16
    VAL_SPLIT = 0.1

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training encoder-only on {device}")

    # dataset
    dataset = GTReadyDataset(DATA_DIR)
    dataset = patch_dataset_with_get_by_name(dataset)
    dataset.files = dataset.files[:5000]  # debug subset

    n_samples = len(dataset)
    n_val = int(n_samples * VAL_SPLIT)
    n_train = n_samples - n_val

    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, collate_fn=collate_skip
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=collate_skip
    )

    # model
    model = DiffusionEncoderOnly(
        latent_dim=LATENT_DIM,
        width=WIDTH,
        n_blocks=N_BLOCKS,
        dropout=0.1,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3)

    # -------------------------
    # Load GT distance matrix
    # -------------------------
    print("📂 Loading GT distance matrix...")
    D_pack = np.load(D_PATH := DIST_PATH, allow_pickle=True)
    D_orig = D_pack["D_orig"].astype(np.float64)
    D_orig /= np.max(D_orig[D_orig > 0])
    name_to_idx = build_name_index_map(D_pack["names"])

    # subset for identity check
    FIXED_N = min(100, len(dataset.files))
    fixed_names = dataset.files[:FIXED_N]
    fixed_idx = []

    for nm in fixed_names:
        base = nm[:-4] if nm.endswith(".npz") else nm
        if base in name_to_idx:
            fixed_idx.append(name_to_idx[base])

    fixed_idx = np.array(fixed_idx)
    if len(fixed_idx) < 2:
        print("⚠️ Not enough samples for identity validation.")
        fixed_names = None
        D_ref = None
    else:
        D_ref = D_orig[np.ix_(fixed_idx, fixed_idx)]

    # -------------------------
    # Logging
    # -------------------------
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(OUT_DIR, "runs", run_name))
    log_csv = os.path.join(OUT_DIR, "train_log.csv")

    with open(log_csv, "w") as f:
        f.write(
            "epoch,train_loss,val_loss,"
            "ident_pearson,ident_spearman,ident_r2,ident_slope,"
            "current_lr\n"
        )

    # =====================================================================
    # TRAINING LOOP
    # =====================================================================

    print("\n--- START TRAINING (ENCODER ONLY) ---")

    for epoch in range(EPOCHS):

        model.train()
        epoch_loss = 0.0
        valid_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch_list in pbar:
            if len(batch_list) == 0:
                continue

            optimizer.zero_grad(set_to_none=True)

            Z_batch = []
            names_batch = []

            # -----------------
            # forward encoder
            # -----------------
            for sample in batch_list:
                try:
                    nm = sample["name"]
                    base = nm[:-4] if nm.endswith(".npz") else nm

                    if base not in name_to_idx:
                        continue  # IGNORA nomi non presenti nella matrice GT

                    V = sample["verts"].to(device)
                    mass = sample["mass"].to(device)
                    evals = sample["evals"].to(device)
                    evecs = sample["evecs"].to(device)
                    faces = sample["faces"].to(device)
                    L = sample["L"].to(device)
                    gradX = sample["gradX"].to(device)
                    gradY = sample["gradY"].to(device)

                    Z_global = model(V, mass, L, evals, evecs, faces, gradX, gradY)

                    Z_batch.append(Z_global)
                    names_batch.append(base)

                except Exception as e:
                    print(f"[ERROR] Skipping sample {sample.get('name')} → {e}")

            if len(Z_batch) == 0:
                continue

            Z_batch = torch.cat(Z_batch, dim=0)

            # stress loss
            # -----------------
            # Costruisci D_batch: matrice delle distanze target per il batch corrente
            idx_batch = [name_to_idx[nm] for nm in names_batch]
            idx_batch = np.array(idx_batch, dtype=int)

            D_batch = torch.tensor(
                D_orig[np.ix_(idx_batch, idx_batch)],
                dtype=Z_batch.dtype,
                device=Z_batch.device
            )

            # 1) stress loss (preserva struttura)
            loss_stress = stress_loss(Z_batch, D_batch)

            # 2) pairwise latent distances
            dist_lat = torch.cdist(Z_batch, Z_batch, p=2)

            # mask per evitare la diagonale
            mask = (D_batch > 0)

            # 3) ratio = dist_latente / dist_GT  → la media del ratio è la slope
            ratio = dist_lat[mask] / (D_batch[mask] + 1e-8)

            # 4) loss che forza slope ≈ 1
            loss_slope = (ratio.mean() - 1.0) ** 2

            # 5) combinazione finale
            loss = loss_stress + 0.5 * loss_slope

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            valid_batches += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = epoch_loss / max(1, valid_batches)

        # =====================================================================
        # VALIDATION
        # =====================================================================
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for sample_list in val_loader:
                if len(sample_list) == 0:
                    continue

                sample = sample_list[0]

                # produce embedding
                V = sample["verts"].to(device)
                mass = sample["mass"].to(device)
                evals = sample["evals"].to(device)
                evecs = sample["evecs"].to(device)
                faces = sample["faces"].to(device)
                L = sample["L"].to(device)
                gradX = sample["gradX"].to(device)
                gradY = sample["gradY"].to(device)

                Z_global = model(V, mass, L, evals, evecs, faces, gradX, gradY)

                # no stress loss with batch=1
                val_loss += 0.0

        val_loss = val_loss / max(1, len(val_loader))

        # LR scheduling
        scheduler.step(train_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # =====================================================================
        # IDENTITY CHECK
        # =====================================================================
        if fixed_names is not None:
            stats = latent_identity_check(model, dataset, fixed_names, fixed_idx, D_ref, device)
            ident_pear = stats.get("pearson", np.nan)
            ident_spear = stats.get("spearman", np.nan)
            ident_r2 = stats.get("r2", np.nan)
            ident_slope = stats.get("slope", np.nan)
        else:
            ident_pear = ident_spear = ident_r2 = ident_slope = np.nan

        # =====================================================================
        # PRINT SUMMARY
        # =====================================================================
        print(f"\n🧠 Epoch {epoch+1}/{EPOCHS}")
        print(f"   Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"   Identity (pear/spear/R2/slope): "
              f"{ident_pear:.4f} / {ident_spear:.4f} / {ident_r2:.4f} / {ident_slope:.4f}")
        print(f"   LR: {current_lr:.1e}")

        # CSV
        with open(log_csv, "a") as f:
            f.write(
                f"{epoch+1},"
                f"{train_loss:.6f},{val_loss:.6f},"
                f"{ident_pear:.4f},{ident_spear:.4f},{ident_r2:.4f},{ident_slope:.4f},"
                f"{current_lr:.1e}\n"
            )

        # Checkpoint
        if ((epoch + 1) % 5 == 0) or (epoch + 1 == EPOCHS):
            ckpt = os.path.join(OUT_DIR, f"encoder_only_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"💾 Saved checkpoint: {ckpt}")

    print("\n✅ DONE.")


if __name__ == "__main__":
    main()