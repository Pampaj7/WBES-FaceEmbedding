#!/usr/bin/env python3
import os
import numpy as np
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler

from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset
from diffusion_autoencoder import DiffusionAutoencoder
from geometric_loss import GeometricLoss
from latent_loss import multiscale_distance_loss

from helper import (
    patch_dataset_with_get_by_name,
    latent_identity_check,
    collate_skip,
    build_name_index_map,
)


# ============================================================
# Main
# ============================================================
def main():
    try:
        torch.multiprocessing.set_start_method("spawn", force=False)
    except RuntimeError:
        pass

    # === CONFIG ===
    DATA_DIR = "../../../datasets/GT_ready/npz_data_cropped_23470_with_ops/"
    DIST_PATH = (
        "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/"
        "gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/"
        "normalized_matrix_distances.npz"
    )

    OUT_DIR = "./results_diffusionAE_latentaware_v4/"
    os.makedirs(OUT_DIR, exist_ok=True)

    LATENT_DIM = 256
    WIDTH = 128
    N_BLOCKS = 4
    EPOCHS = 50

    LR = 1e-4
    BATCH_SIZE = 8
    VAL_SPLIT = 0.1
    CHECKPOINT_EVERY = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Geometric loss weights
    W_L1, W_NORMAL, W_LAPLACIAN = 0.1, 0.3, 0.2
    GEO_W = 1.0
    LAT_W = 1.0

    ENABLE_LATENT_LOSSES = False

    use_amp = torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None

    print(f"🚀 Training Latent-Aware Diffusion AE on {DEVICE}")

    # === Dataset ===
    dataset = GTReadyDataset(DATA_DIR)
    dataset = patch_dataset_with_get_by_name(dataset)

    valid_files = []
    skipped_csv = os.path.join(OUT_DIR, "skipped_files.csv")
    with open(skipped_csv, "w") as ff:
        ff.write("filename,error\n")

    for f in dataset.files[:1000]:
        path = os.path.join(DATA_DIR, f)
        try:
            with np.load(path) as data:
                _ = data.files
            valid_files.append(f)
        except Exception as e:
            with open(skipped_csv, "a") as ff:
                ff.write(f"{f},{e}\n")

    dataset.files = valid_files
    print(f"📊 Valid NPZ count: {len(dataset.files)}")

    # === Split ===
    n_val = int(len(dataset) * VAL_SPLIT)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, collate_fn=collate_skip
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False,
        collate_fn=collate_skip
    )

    # === Model ===
    model = DiffusionAutoencoder(
        latent_dim=LATENT_DIM,
        width=WIDTH,
        n_blocks=N_BLOCKS
    ).to(DEVICE)

    geom_loss = GeometricLoss(W_L1, W_NORMAL, W_LAPLACIAN, device=DEVICE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7
    )

    # === Load GT matrix ===
    print("📂 Loading GT distance matrix...")
    D_pack = np.load(DIST_PATH, allow_pickle=True)
    D_orig = D_pack["D_orig"].astype(np.float64)
    norm_factor = np.max(D_orig[D_orig > 0])
    D_orig = D_orig / norm_factor
    name_to_idx = build_name_index_map(D_pack["names"])

    # === STRICT subset ===
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
        fixed_names = None
        D_ref = None
        print("⚠️ STRICT validation disabled.")

    # === Logger ===
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(OUT_DIR, "runs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    log_csv = os.path.join(OUT_DIR, "train_log.csv")
    with open(log_csv, "w") as f:
        f.write(
            "epoch,train_total,train_geo,train_lat,val_geo,"
            "pearson,spearman,r2,slope,lr\n"
        )

    # ============================================================
    # TRAINING LOOP
    # ============================================================
    best_val_metric = -np.inf
    best_ckpt_path = os.path.join(OUT_DIR, "latentaware_best.pth")

    for epoch in range(EPOCHS):
        model.train()
        total = geo = lat = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Train {epoch+1}", ncols=140)

        for b_idx, batch_list in enumerate(pbar):

            if len(batch_list) == 0:
                continue

            optimizer.zero_grad(set_to_none=True)

            geo_loss_accum = 0.0
            Zg_list = []
            batch_names = []     # <--- IMPORTANTISSIMO

            # ===== forward per sample =====
            for sample in batch_list:
                V = sample["verts"].to(DEVICE)
                mass = sample["mass"].to(DEVICE)
                evals = sample["evals"].to(DEVICE)
                evecs = sample["evecs"].to(DEVICE)
                faces = sample["faces"].to(DEVICE)
                L = sample["L"].to(DEVICE)
                gX = sample["gradX"].to(DEVICE)
                gY = sample["gradY"].to(DEVICE)

                out = model(V, mass, L, evals, evecs, faces, gX, gY)
                V_rec, Z_global = out

                L_geo, _ = geom_loss(V_rec, V, faces, L)
                geo_loss_accum += L_geo / len(batch_list)

                if Z_global.dim() == 1:
                    Z_global = Z_global.unsqueeze(0)
                Zg_list.append(Z_global)

                batch_names.append(sample["name"])   # <--- SALVO I NOMI

            # ===== LATENT LOSS =====
            L_lat = torch.tensor(0.0, device=DEVICE)

            if ENABLE_LATENT_LOSSES:

                Zg = torch.cat(Zg_list, dim=0)
                B = Zg.shape[0]

                idxs = []
                for nm in batch_names:
                    base = nm[:-4] if nm.endswith(".npz") else nm
                    if base in name_to_idx:
                        idxs.append(name_to_idx[base])

                if len(idxs) == B:
                    idxs = np.array(idxs)
                    D_gt_sub = D_orig[np.ix_(idxs, idxs)]
                    D_gt_sub = torch.tensor(D_gt_sub, dtype=torch.float32, device=DEVICE)

                    L_lat = multiscale_distance_loss(Zg, D_gt_sub)

            # ===== TOTAL LOSS =====
            L_total = GEO_W * geo_loss_accum + LAT_W * L_lat

            if use_amp:
                scaler.scale(L_total).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                L_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total += float(L_total.detach())
            geo += float(geo_loss_accum.detach())
            lat += float(L_lat.detach())
            n_batches += 1

        # ===== metrics epoch =====
        train_total = total / n_batches
        train_geo = geo / n_batches
        train_lat = lat / n_batches

        # ===================== VALIDATION =====================
        model.eval()
        val_geo = 0.0

        with torch.no_grad():
            for val_sample in val_loader:
                s = val_sample[0]
                V = s["verts"].to(DEVICE)
                mass = s["mass"].to(DEVICE)
                evals = s["evals"].to(DEVICE)
                evecs = s["evecs"].to(DEVICE)
                faces = s["faces"].to(DEVICE)
                L = s["L"].to(DEVICE)
                gX = s["gradX"].to(DEVICE)
                gY = s["gradY"].to(DEVICE)

                out = model(V, mass, L, evals, evecs, faces, gX, gY)
                V_rec = out[0]
                Lg, _ = geom_loss(V_rec, V, faces, L)
                val_geo += float(Lg)

        val_geo /= max(1, len(val_loader))

        # STRICT latent validation
        stats = latent_identity_check(
            model, dataset, fixed_names, fixed_idx, D_ref, DEVICE
        )

        #scheduler.step(train_total)

        lr = optimizer.param_groups[0]["lr"]


        # === LOGGING ===
        with open(log_csv, "a") as f:
            f.write(
                f"{epoch+1},{train_total:.6f},{train_geo:.6f},{train_lat:.6f},"
                f"{val_geo:.6f},"
                f"{(stats['pearson'] if stats else np.nan):.4f},"
                f"{(stats['spearman'] if stats else np.nan):.4f},"
                f"{(stats['r2'] if stats else np.nan):.4f},"
                f"{(stats['slope'] if stats else np.nan):.4f},"
                f"{lr:.1e}\n"
            )

        metric = stats["pearson"] if stats else -val_geo

        if metric > best_val_metric:
            best_val_metric = metric
            torch.save(
                {
                    "epoch": epoch+1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_val_metric,
                },
                best_ckpt_path
            )
            print(f"💾 New BEST checkpoint → Pearson={metric:.4f}")
        else:
            print(f"⚠️ No improvement (metric={metric:.4f})")

        # === CHECKPOINT OGNI 5 EPOCH ===
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(OUT_DIR, f"ckpt_epoch{epoch+1}.pth")
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            print(f"💾 Saved checkpoint at epoch {epoch+1} → {ckpt_path}")
            
    # === ALWAYS SAVE LAST CHECKPOINT ===
    last_ckpt_path = os.path.join(OUT_DIR, "latentaware_last.pth")
    torch.save(
        {
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        last_ckpt_path
    )
    
    print(f"\n🏆 Training completed.")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Last checkpoint: {last_ckpt_path}")


if __name__ == "__main__":
    main()
