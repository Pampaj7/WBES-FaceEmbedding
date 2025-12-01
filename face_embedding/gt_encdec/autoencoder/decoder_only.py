#!/usr/bin/env python3
import os
import math
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# dataset e operatori
from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset
from geometric_loss import GeometricLoss

# metriche latenti (qui non le usiamo, ma lasciamo l'import)
from latent_loss import stress_loss, scale_loss

# helper
from helper import (
    patch_dataset_with_get_by_name,
    latent_identity_check,
    build_name_index_map,
)

# ============================================================
# IMPORT DIFFUSIONNET
# ============================================================
try:
    import diffusion_net
    DiffusionNet = diffusion_net.layers.DiffusionNet
except Exception:
    from diffusion_net import DiffusionNet


# ============================================================
# DECODER-ONLY MODEL (PER-VERTEX LATENTS)
# ============================================================
class DecoderOnlyPerVertex(nn.Module):
    """
    Decoder-only:
    - Nessun encoder
    - Z_per_vertex è UN PARAMETRO ottimizzabile
    - Decoder identico a quello del tuo AE
    - Input: [Z_per_vertex, evecs]
    """

    def __init__(self, num_vertices, latent_dim=256, width=128, n_blocks=4, k_spec=16):
        super().__init__()

        self.latent_dim = latent_dim
        self.k_spec = k_spec

        # Parametri per-vertex: [N_verts, latent_dim]
        self.Z_per_vertex = nn.Parameter(
            torch.randn(num_vertices, latent_dim) * 0.01
        )

        # Decoder identico al tuo AE
        cin_decoder = latent_dim + k_spec
        self.decoder = DiffusionNet(
            C_in=cin_decoder,
            C_out=3,
            C_width=width,
            N_block=n_blocks,
            with_gradient_features=True,
            dropout=0.0,
        )

    @staticmethod
    def _take_or_pad_evecs(evecs, k):
        n, kvar = evecs.shape
        if kvar >= k:
            return evecs[:, :k]
        pad = torch.zeros(n, k - kvar, device=evecs.device, dtype=evecs.dtype)
        return torch.cat([evecs, pad], dim=1)

    def forward(self, ops):
        S = self._take_or_pad_evecs(ops["evecs"], self.k_spec)

        # concatenazione identica al tuo AE
        Z_in = torch.cat([self.Z_per_vertex, S], dim=1)

        V_rec = self.decoder(
            Z_in,
            ops["mass"],
            ops["L"],
            ops["evals"],
            ops["evecs"],
            faces=ops["faces"],
            gradX=ops["gradX"],
            gradY=ops["gradY"],
        )

        return V_rec


# ============================================================
# TRAINING SCRIPT
# ============================================================
def collate_skip(batch):
    return [s for s in batch if s is not None]


def main():
    # ============================================================
    # MULTIPROCESSING
    # ============================================================
    try:
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # ============================================================
    # CONFIG
    # ============================================================
    DATA_DIR = "../../../datasets/GT_ready/npz_data_cropped_23470_with_ops/"
    DIST_PATH = (
        "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/"
        "gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/"
        "normalized_matrix_distances.npz"
    )

    OUT_DIR = "decoder_only_experiment"
    os.makedirs(OUT_DIR, exist_ok=True)

    LATENT_DIM = 256
    WIDTH = 128
    N_BLOCKS = 4
    K_SPEC = 16

    EPOCHS = 50
    LR = 1e-4
    BATCH_SIZE = 8

    N_WORKERS = 0
    PIN_MEMORY = False
    VAL_SPLIT = 0.1
    CHECKPOINT_EVERY = 5

    # debug / preview
    DEBUG_EVERY = 15          # stampa log extra ogni N batch
    SAVE_PREVIEW_EVERY = 100  # salva una ricostruzione .npz ogni N batch (0 = disabilitato)

    # geometric loss weights
    W_L1 = 0.3
    W_NORMAL = 1.0
    W_LAPLACIAN = 0.7

    # latent regularization (qui tenuti a 0, ma li lasciamo nel caso li riattivi)
    LAMBDA_STRESS = 0.0
    LAMBDA_SCALE = 0.0
    WARMUP_EPOCHS = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    # ============================================================
    # DATASET
    # ============================================================
    dataset = GTReadyDataset(DATA_DIR)
    dataset = patch_dataset_with_get_by_name(dataset)
    dataset.files = dataset.files[:1000]   # subset per esperimento decoder-only

    print(f"🧩 Dataset: {len(dataset.files)} meshes")

    example = dataset[0]
    NUM_VERTS = example["verts"].shape[0]

    n_samples = len(dataset)
    n_val = int(n_samples * VAL_SPLIT)
    n_train = n_samples - n_val
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

    # ============================================================
    # MODEL
    # ============================================================
    model = DecoderOnlyPerVertex(
        num_vertices=NUM_VERTS,
        latent_dim=LATENT_DIM,
        width=WIDTH,
        n_blocks=N_BLOCKS,
        k_spec=K_SPEC,
    ).to(device)

    criterion = GeometricLoss(
        w_l1=W_L1,
        w_normal=W_NORMAL,
        w_laplacian=W_LAPLACIAN,
        device=device,
    )

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7
    )

    # ============================================================
    # DIST MATRICES (qui non usate, ma lasciate per compatibilità futura)
    # ============================================================
    print("📂 Loading GT distance matrix…")
    D_pack = np.load(DIST_PATH, allow_pickle=True)
    D_orig = D_pack["D_orig"].astype(np.float64)
    D_orig /= np.max(D_orig[D_orig > 0])
    name_to_idx = build_name_index_map(D_pack["names"])

    # ============================================================
    # LOGGING
    # ============================================================
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(OUT_DIR, "runs", run_name))

    log_csv = os.path.join(OUT_DIR, "train_log.csv")
    with open(log_csv, "w") as f:
        f.write(
            "epoch,train_loss,val_loss,train_l1,val_l1,"
            "train_n,val_n,train_lapl,val_lapl,"
            "train_cos,val_cos,train_pear,val_pear,"
            "lr\n"
        )

    # ============================================================
    # TRAINING
    # ============================================================
    print("\n=== START TRAINING DECODER-ONLY ===")

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
            optimizer.zero_grad(set_to_none=True)

            geo_sum = None
            l1_sum = 0.0
            normal_sum = 0.0
            lapl_sum = 0.0
            geo_count = 0

            last_V = None
            last_Vrec = None
            last_faces = None

            for sample in batch_list:
                V = sample["verts"].to(device)
                ops = {
                    "mass": sample["mass"].to(device),
                    "L": sample["L"].to(device),
                    "evals": sample["evals"].to(device),
                    "evecs": sample["evecs"].to(device),
                    "faces": sample["faces"].to(device),
                    "gradX": sample["gradX"].to(device),
                    "gradY": sample["gradY"].to(device),
                }

                V_rec = model(ops)
                geo_loss, breakdown = criterion(V_rec, V, ops["faces"], ops["L"])

                if not torch.isfinite(geo_loss):
                    continue

                if geo_sum is None:
                    geo_sum = geo_loss
                else:
                    geo_sum = geo_sum + geo_loss

                geo_count += 1
                l1_sum += float(breakdown["loss_l1"])
                normal_sum += float(breakdown["loss_normal"])
                lapl_sum += float(breakdown["loss_laplacian"])

                # cosine / pearson
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

                # salviamo l'ultimo sample del batch per eventuale preview
                last_V = V
                last_Vrec = V_rec
                last_faces = ops["faces"]

            if geo_count == 0:
                continue

            geo_loss_batch = geo_sum / geo_count
            avg_l1 = l1_sum / geo_count
            avg_norm = normal_sum / geo_count
            avg_lapl = lapl_sum / geo_count

            # ================= DEBUG BATCH-LEVEL =================
            if batch_idx % DEBUG_EVERY == 0:
                with torch.no_grad():
                    z_mean = model.Z_per_vertex.mean().item()
                    z_std = model.Z_per_vertex.std().item()
                print(
                    f"\n--- DEBUG epoch {epoch+1} batch {batch_idx} ---\n"
                    f"geo_loss_batch={geo_loss_batch.item():.6f} | "
                    f"L1={avg_l1:.4f} | N={avg_norm:.4f} | Lap={avg_lapl:.4f}\n"
                    f"Z_per_vertex: mean={z_mean:.5f}, std={z_std:.5f}\n"
                    "---------------------------------------------"
                )

            # ================= PREVIEW SALVATAGGIO =================
            if (
                SAVE_PREVIEW_EVERY > 0
                and batch_idx % SAVE_PREVIEW_EVERY == 0
                and last_V is not None
                and last_Vrec is not None
                and last_faces is not None
            ):
                preview_path = os.path.join(
                    OUT_DIR,
                    f"preview_epoch{epoch+1}_batch{batch_idx}.npz",
                )
                np.savez(
                    preview_path,
                    V_gt=last_V.detach().cpu().numpy(),
                    V_pred=last_Vrec.detach().cpu().numpy(),
                    faces=last_faces.detach().cpu().numpy(),
                )
                print(f"💾 Saved preview reconstruction → {preview_path}")

            # BACKWARD
            geo_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss_total += float(geo_loss_batch.item())
            epoch_loss_l1 += avg_l1
            epoch_loss_normal += avg_norm
            epoch_loss_lapl += avg_lapl
            valid_batches += 1

            pbar.set_postfix(geo=f"{geo_loss_batch.item():.4f}")

        if valid_batches == 0:
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

        # ============================================================
        # VALIDATION
        # ============================================================
        model.eval()
        val_total = val_l1 = val_n = val_lapl = 0.0
        val_cos_sum = val_pear_sum = 0.0
        val_corr_count = 0
        n_val_s = 0

        with torch.no_grad():
            for sample_list in tqdm(
                val_loader, desc="Validation", leave=False, dynamic_ncols=True
            ):
                if len(sample_list) == 0:
                    continue
                sample = sample_list[0]

                V = sample["verts"].to(device)
                ops = {
                    "mass": sample["mass"].to(device),
                    "L": sample["L"].to(device),
                    "evals": sample["evals"].to(device),
                    "evecs": sample["evecs"].to(device),
                    "faces": sample["faces"].to(device),
                    "gradX": sample["gradX"].to(device),
                    "gradY": sample["gradY"].to(device),
                }

                V_rec = model(ops)
                loss, breakdown = criterion(V_rec, V, ops["faces"], ops["L"])

                if not torch.isfinite(loss):
                    continue

                val_total += float(breakdown["loss_total"])
                val_l1 += float(breakdown["loss_l1"])
                val_n += float(breakdown["loss_normal"])
                val_lapl += float(breakdown["loss_laplacian"])
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

        val_cos_avg = (
            val_cos_sum / val_corr_count if val_corr_count > 0 else float("nan")
        )
        val_pear_avg = (
            val_pear_sum / val_corr_count if val_corr_count > 0 else float("nan")
        )

        if math.isfinite(val_loss):
            scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]

        # ============================================================
        # PRINT LOG
        # ============================================================
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

        # ============================================================
        # CSV LOGGING
        # ============================================================
        with open(log_csv, "a") as f:
            f.write(
                f"{epoch+1},{train_loss:.6f},{val_loss:.6f},"
                f"{train_l1:.6f},{val_l1_avg:.6f},"
                f"{train_n:.6f},{val_n_avg:.6f},"
                f"{train_lapl:.6f},{val_lapl_avg:.6f},"
                f"{train_cos_avg:.6f},{val_cos_avg:.6f},"
                f"{lr_now:.1e}\n"
            )

        # ============================================================
        # CHECKPOINT
        # ============================================================
        if (epoch + 1) % CHECKPOINT_EVERY == 0 or (epoch + 1) == EPOCHS:
            ckpt = os.path.join(OUT_DIR, f"decoder_only_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"💾 Saved: {ckpt}")

        # ============================================================
        # TENSORBOARD
        # ============================================================
        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Loss/val", val_loss, epoch + 1)
        writer.add_scalar("LR", lr_now, epoch + 1)

    writer.close()
    print("\n✅ Training Complete")


if __name__ == "__main__":
    main()
