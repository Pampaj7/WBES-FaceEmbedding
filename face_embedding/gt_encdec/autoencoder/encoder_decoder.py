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

# dataset + model + loss
from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset
from diffusion_autoencoder import DiffusionAutoencoder
from geometric_loss import GeometricLoss

# latent
from latent_loss import stress_loss, scale_loss

# helper
from helper import (
    patch_dataset_with_get_by_name,
    latent_identity_check,
    build_name_index_map,
)


def collate_skip(batch):
    """Drop None samples che il dataset potrebbe produrre."""
    return [s for s in batch if s is not None]


def main():
    # =============================== MULTIPROCESSING ===============================
    try:
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # già inizializzato, va bene così
        pass

    # =============================== CONFIG ===============================
    DATA_DIR = "../../../datasets/GT_ready/npz_data_cropped_23470_with_ops/"
    DIST_PATH = (
        "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/"
        "gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/"
        "normalized_matrix_distances.npz"
    )

    OUT_DIR = "test_safe_latent"
    os.makedirs(OUT_DIR, exist_ok=True)

    LATENT_DIM = 256
    WIDTH = 128
    N_BLOCKS = 4

    EPOCHS = 50
    LR = 1e-4
    BATCH_SIZE = 8

    N_WORKERS = 0
    PIN_MEMORY = False
    VAL_SPLIT = 0.1
    CHECKPOINT_EVERY = 5
    DEBUG_EVERY = 25

    # geometric loss weights
    W_L1 = 0.3
    W_NORMAL = 1.0
    W_LAPLACIAN = 0.7

    # latent reg
    LAMBDA_STRESS = 0.05
    LAMBDA_SCALE = 0.01
    WARMUP_EPOCHS = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    # =============================== DATASET ===============================
    dataset = GTReadyDataset(DATA_DIR)
    dataset = patch_dataset_with_get_by_name(dataset)

    dataset.files = dataset.files[:5000]
    print(f"🧩 Dataset subset: {len(dataset.files)} meshes")

    n_samples = len(dataset)
    if n_samples == 0:
        print("[ERRORE] Dataset vuoto.")
        return

    n_val = int(n_samples * VAL_SPLIT)
    n_train = n_samples - n_val
    if n_train <= 0 or n_val <= 0:
        print(f"[ERRORE] Dataset troppo piccolo per split: n={n_samples}")
        return

    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=N_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_skip,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=N_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_skip,
    )

    # =============================== MODEL ===============================
    model = DiffusionAutoencoder(
        latent_dim=LATENT_DIM,
        width=WIDTH,
        n_blocks=N_BLOCKS,
    ).to(device)

    criterion = GeometricLoss(
        w_l1=W_L1,
        w_normal=W_NORMAL,
        w_laplacian=W_LAPLACIAN,
        device=device,
    )

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
    )

    # =============================== GT DIST MATRIX ===============================
    print("📂 Loading GT distance matrix…")
    D_pack = np.load(DIST_PATH, allow_pickle=True)
    D_orig = D_pack["D_orig"].astype(np.float64)
    D_orig /= np.max(D_orig[D_orig > 0])
    name_to_idx = build_name_index_map(D_pack["names"])

    FIXED_N = min(100, len(dataset.files))
    fixed_names = dataset.files[:FIXED_N]

    fixed_idx = []
    for nm in fixed_names:
        base = nm[:-4] if nm.endswith(".npz") else nm
        if base in name_to_idx:
            fixed_idx.append(name_to_idx[base])
    fixed_idx = np.array(fixed_idx, dtype=int)

    if len(fixed_idx) >= 2:
        D_ref = D_orig[np.ix_(fixed_idx, fixed_idx)]
    else:
        print("⚠️ Identity check disabilitato (meno di 2 match nella GT matrix).")
        D_ref = None
        fixed_names = None

    # =============================== LOGGING ===============================
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(OUT_DIR, "runs", run_name))

    log_csv = os.path.join(OUT_DIR, "train_log.csv")
    with open(log_csv, "w") as f:
        f.write(
            "epoch,"
            "train_loss,val_loss,"
            "train_l1,val_l1,"
            "train_normal,val_normal,"
            "train_laplacian,val_laplacian,"
            "train_cos,val_cos,"
            "train_pear,val_pear,"
            "ident_pear,ident_spear,ident_r2,ident_slope,"
            "lr\n"
        )

    # =============================== TRAINING ===============================
    print("\n=== START TRAINING ===")

    for epoch in range(EPOCHS):
        model.train()

        epoch_loss_total = 0.0
        epoch_loss_l1 = 0.0
        epoch_loss_normal = 0.0
        epoch_loss_lapl = 0.0
        valid_batches = 0

        train_cos_sum = 0.0
        train_pear_sum = 0.0
        train_corr_count = 0

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{EPOCHS}",
            dynamic_ncols=True,
        )

        for batch_idx, batch_list in pbar:
            if len(batch_list) == 0:
                continue

            optimizer.zero_grad(set_to_none=True)

            # ===============================
            # GEO ACCUMULATORS (con grad)
            # ===============================
            geo_sum = None         # tensore che accumula loss con grad
            l1_sum = 0.0           # numeri per logging
            normal_sum = 0.0
            lapl_sum = 0.0
            geo_count = 0

            # ===============================
            # LATENT ACCUMULATORS
            # ===============================
            Z_batch_list = []
            names_batch = []

            debug_flag = (batch_idx % DEBUG_EVERY == 0)

            for sample in batch_list:
                nm = sample["name"]
                base = nm[:-4] if nm.endswith(".npz") else nm

                V = sample["verts"].to(device)
                mass = sample["mass"].to(device)
                evals = sample["evals"].to(device)
                evecs = sample["evecs"].to(device)
                faces = sample["faces"].to(device)
                L = sample["L"].to(device)
                gradX = sample["gradX"].to(device)
                gradY = sample["gradY"].to(device)

                # ===============================
                # FORWARD
                # ===============================
                V_rec, Z_global = model(
                    V,
                    mass,
                    L,
                    evals,
                    evecs,
                    faces,
                    gradX,
                    gradY,
                )

                # ===============================
                # GEOMETRIC LOSS (con grad)
                # ===============================
                geo_loss, breakdown = criterion(V_rec, V, faces, L)

                if not torch.isfinite(geo_loss):
                    continue

                # 1) ACCUMULO TENSORE PER IL BACKWARD
                if geo_sum is None:
                    geo_sum = geo_loss
                else:
                    geo_sum = geo_sum + geo_loss

                geo_count += 1

                # 2) LOGGING SENZA GRAD
                l1_sum     += float(breakdown["loss_l1"])
                normal_sum += float(breakdown["loss_normal"])
                lapl_sum   += float(breakdown["loss_laplacian"])

                try:
                    cos_m = F.cosine_similarity(
                        V_rec.flatten(), V.flatten(), dim=0
                    ).item()
                    pear_m = torch.corrcoef(
                        torch.stack([V_rec.flatten(), V.flatten()])
                    )[0, 1].item()
                except Exception:
                    cos_m = pear_m = float("nan")

                if math.isfinite(cos_m):
                    train_cos_sum += cos_m
                    train_pear_sum += pear_m
                    train_corr_count += 1

                # LATENT STORAGE
                if base in name_to_idx:
                    Z_batch_list.append(Z_global)
                    names_batch.append(base)

                if debug_flag:
                    print(f"\n--- DEBUG epoch {epoch+1} batch {batch_idx} ---")
                    print(f"sample: {nm}")
                    print(
                        f"latent mean={Z_global.mean():.4f} "
                        f"std={Z_global.std():.4f}"
                    )
                    print(f"geo loss={breakdown['loss_total']:.6f}")
                    print("-------------------------------")

            # ===============================
            # FINE LOOP SAMPLES DEL BATCH
            # ===============================
            if geo_count == 0:
                continue

            geo_loss_batch = geo_sum / geo_count  # <-- TENSORE con grad

            # ===============================
            # LATENT LOSS
            # ===============================
            if (epoch + 1) > WARMUP_EPOCHS and len(Z_batch_list) >= 2:
                Z_batch = torch.cat(Z_batch_list, dim=0)
                idxs = np.array([name_to_idx[n] for n in names_batch], dtype=int)
                D_batch = torch.tensor(
                    D_orig[np.ix_(idxs, idxs)],
                    device=device,
                    dtype=Z_batch.dtype,
                )

                loss_stress = stress_loss(Z_batch, D_batch)
                loss_scale_val = scale_loss(Z_batch, target_mean=1.0)

                latent_loss = (
                    LAMBDA_STRESS * loss_stress +
                    LAMBDA_SCALE  * loss_scale_val
                )
            else:
                latent_loss = torch.tensor(0.0, device=device)

            # ===============================
            # BACKPROP
            # ===============================
            total_loss = geo_loss_batch + latent_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # ===============================
            # LOGGING PER EPOCH
            # ===============================
            epoch_loss_total += float(geo_loss_batch.item())
            epoch_loss_l1    += l1_sum / geo_count
            epoch_loss_normal += normal_sum / geo_count
            epoch_loss_lapl   += lapl_sum / geo_count
            valid_batches     += 1

            pbar.set_postfix(
                geo=f"{geo_loss_batch.item():.4f}",
                lat=f"{latent_loss.item():.4e}",
                lr=optimizer.param_groups[0]["lr"],
            )

        # ----------- end epoch training metrics -----------
        if valid_batches == 0:
            print(f"[WARN] Nessun batch valido in epoch {epoch+1}.")
            continue

        train_loss = epoch_loss_total / valid_batches
        train_l1 = epoch_loss_l1 / valid_batches
        train_n = epoch_loss_normal / valid_batches
        train_lapl = epoch_loss_lapl / valid_batches

        if train_corr_count > 0:
            train_cos_avg = train_cos_sum / train_corr_count
            train_pear_avg = train_pear_sum / train_corr_count
        else:
            train_cos_avg = float("nan")
            train_pear_avg = float("nan")

        # =============================== VALIDATION ===============================
        model.eval()
        val_total = val_l1 = val_n = val_lapl = 0.0
        val_cos_sum = val_pear_sum = 0.0
        val_corr_count = 0
        n_val_s = 0

        with torch.no_grad():
            for sample_list in tqdm(
                val_loader,
                desc="Validation",
                leave=False,
                dynamic_ncols=True,
            ):
                if len(sample_list) == 0:
                    continue
                sample = sample_list[0]

                V = sample["verts"].to(device)
                mass = sample["mass"].to(device)
                evals = sample["evals"].to(device)
                evecs = sample["evecs"].to(device)
                faces = sample["faces"].to(device)
                L = sample["L"].to(device)
                gradX = sample["gradX"].to(device)
                gradY = sample["gradY"].to(device)

                V_rec, _ = model(
                    V,
                    mass,
                    L,
                    evals,
                    evecs,
                    faces,
                    gradX,
                    gradY,
                )

                loss, breakdown = criterion(V_rec, V, faces, L)
                if not torch.isfinite(loss):
                    continue

                val_total += float(breakdown["loss_total"])
                val_l1    += float(breakdown["loss_l1"])
                val_n     += float(breakdown["loss_normal"])
                val_lapl  += float(breakdown["loss_laplacian"])
                n_val_s += 1

                try:
                    cos_m = F.cosine_similarity(
                        V_rec.flatten(), V.flatten(), dim=0
                    ).item()
                    pear_m = torch.corrcoef(
                        torch.stack([V_rec.flatten(), V.flatten()])
                    )[0, 1].item()
                except Exception:
                    cos_m = pear_m = float("nan")

                if math.isfinite(cos_m):
                    val_cos_sum += cos_m
                    val_pear_sum += pear_m
                    val_corr_count += 1

        if n_val_s == 0:
            val_loss = float("inf")
            val_l1_avg = val_n_avg = val_lapl_avg = float("nan")
        else:
            val_loss = val_total / n_val_s
            val_l1_avg = val_l1 / n_val_s
            val_n_avg = val_n / n_val_s
            val_lapl_avg = val_lapl / n_val_s

        if val_corr_count > 0:
            val_cos_avg = val_cos_sum / val_corr_count
            val_pear_avg = val_pear_sum / val_corr_count
        else:
            val_cos_avg = float("nan")
            val_pear_avg = float("nan")

        if math.isfinite(val_loss):
            scheduler.step(val_loss)
        else:
            print("[WARN] val_loss non finita, scheduler.step saltato.")
        lr_now = optimizer.param_groups[0]["lr"]

        # =============================== IDENTITY ===============================
        if D_ref is not None and fixed_names is not None:
            ident = latent_identity_check(
                model, dataset, fixed_names, fixed_idx, D_ref, device
            )
            ident_pear = ident["pearson"]
            ident_spear = ident["spearman"]
            ident_r2 = ident["r2"]
            ident_slope = ident["slope"]
        else:
            ident_pear = ident_spear = ident_r2 = ident_slope = float("nan")

        # =============================== PRINT ===============================
        print(f"\n🧠 Epoch {epoch+1}/{EPOCHS}")
        print(
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | LR={lr_now:.2e}"
        )
        print(
            f"Train L1/N/Lap: {train_l1:.4f}/{train_n:.4f}/{train_lapl:.4f}"
        )
        print(
            f"Val   L1/N/Lap: {val_l1_avg:.4f}/{val_n_avg:.4f}/{val_lapl_avg:.4f}"
        )
        print(
            f"Train cos/pear: {train_cos_avg:.4f}/{train_pear_avg:.4f}"
        )
        print(
            f"Val   cos/pear: {val_cos_avg:.4f}/{val_pear_avg:.4f}"
        )
        print(
            f"Identity: {ident_pear:.4f}/"
            f"{ident_spear:.4f}/"
            f"{ident_r2:.4f}/"
            f"{ident_slope:.4f}"
        )

        # =============================== CSV ===============================
        with open(log_csv, "a") as f:
            f.write(
                f"{epoch+1},"
                f"{train_loss:.6f},{val_loss:.6f},"
                f"{train_l1:.6f},{val_l1_avg:.6f},"
                f"{train_n:.6f},{val_n_avg:.6f},"
                f"{train_lapl:.6f},{val_lapl_avg:.6f},"
                f"{train_cos_avg:.6f},{val_cos_avg:.6f},"
                f"{train_pear_avg:.6f},{val_pear_avg:.6f},"
                f"{ident_pear:.4f},{ident_spear:.4f},{ident_r2:.4f},{ident_slope:.4f},"
                f"{lr_now:.1e}\n"
            )

        # =============================== CHECKPOINT ===============================
        if (epoch + 1) % CHECKPOINT_EVERY == 0 or (epoch + 1) == EPOCHS:
            ckpt = os.path.join(OUT_DIR, f"diffusionAE_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"💾 Saved: {ckpt}")

        # =============================== TENSORBOARD ===============================
        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Loss/val", val_loss, epoch + 1)
        writer.add_scalar("LR", lr_now, epoch + 1)

    writer.close()
    print("\n✅ Training Complete")


if __name__ == "__main__":
    main()