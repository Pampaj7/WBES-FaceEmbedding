#!/usr/bin/env python3
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset
from geometric_loss import GeometricLoss
from helper import patch_dataset_with_get_by_name, collate_skip

from diffusion_autoencoder import DiffusionEncoderOnly

# ==========================================================
# DiffusionNet decoder only (conditioned on Z_per_vertex)
# ==========================================================
try:
    import diffusion_net
    DiffusionNet = diffusion_net.layers.DiffusionNet
except Exception:
    from diffusion_net import DiffusionNet


class EncoderFrozenWithDecoder(nn.Module):
    """
    Stage 2 model:
    - Frozen DiffusionEncoderOnly (pre-trained, stress-based)
    - New DiffusionNet decoder trained with geometric loss only
    - Input to decoder: [Z_per_vertex, spectral features]
    """

    def __init__(
        self,
        encoder_ckpt_path: str,
        latent_dim: int = 256,
        width: int = 128,
        n_blocks: int = 4,
        k_spec: int = 16,
        dropout_encoder: float = 0.1,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.k_spec = k_spec
        self.device = device

        # --------------------------
        # Load pre-trained encoder
        # --------------------------
        self.encoder = DiffusionEncoderOnly(
            latent_dim=latent_dim,
            width=width,
            n_blocks=n_blocks,
            dropout=dropout_encoder,
        )

        print(f"📂 Loading encoder checkpoint from: {encoder_ckpt_path}")
        state = torch.load(encoder_ckpt_path, map_location=device)
        self.encoder.load_state_dict(state)

        # Freeze encoder parameters
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.encoder.to(device)
        self.encoder.eval()  # no dropout / no noise

        # --------------------------
        # Decoder: DiffusionNet
        # --------------------------
        cin_decoder = latent_dim + k_spec

        self.decoder = DiffusionNet(
            C_in=cin_decoder,
            C_out=3,
            C_width=width,
            N_block=n_blocks,
            with_gradient_features=True,
            dropout=0.0,
        ).to(device)

    @staticmethod
    def take_or_pad_evecs(evecs: torch.Tensor, k: int) -> torch.Tensor:
        """
        Ensure we always have exactly k spectral components.
        """
        n, kvar = evecs.shape
        if kvar >= k:
            return evecs[:, :k]
        pad = torch.zeros(n, k - kvar, device=evecs.device, dtype=evecs.dtype)
        return torch.cat([evecs, pad], dim=1)

    def forward(self, sample: dict) -> torch.Tensor:
        """
        sample: dict with keys:
            - verts, mass, L, evals, evecs, faces, gradX, gradY
        """
        V = sample["verts"].to(self.device)
        mass = sample["mass"].to(self.device)
        L = sample["L"].to(self.device)
        evals = sample["evals"].to(self.device)
        evecs = sample["evecs"].to(self.device)
        faces = sample["faces"].to(self.device)
        gradX = sample["gradX"].to(self.device)
        gradY = sample["gradY"].to(self.device)

        # 1) Frozen encoder: per-vertex latent field
        with torch.no_grad():
            Z_per_vertex, _ = self.encoder(
                V,
                mass,
                L,
                evals,
                evecs,
                faces,
                gradX,
                gradY,
                return_per_vertex=True,
                add_noise=False,  # deterministic in Stage 2
            )  # (N_verts, latent_dim)

        # 2) Spectral features
        S = self.take_or_pad_evecs(evecs, self.k_spec)  # (N_verts, k_spec)

        # 3) Concatenate latents + spectral features
        Z_in = torch.cat([Z_per_vertex, S], dim=1)  # (N_verts, latent_dim + k_spec)

        # 4) Decoder predicts vertices
        V_rec = self.decoder(
            Z_in,
            mass,
            L,
            evals,
            evecs,
            faces=faces,
            gradX=gradX,
            gradY=gradY,
        )
        return V_rec


# ==========================================================
# TRAINING LOOP FOR STAGE 2
# ==========================================================
def main():

    # ---------------- CONFIG ----------------
    DATA_DIR = "../../../datasets/GT_ready/npz_data_cropped_23470_with_ops/"
    ENCODER_CKPT = "encoder_only/encoder_only_epoch50.pth"  # adjust if needed
    OUT_DIR = "stage2_frozen"
    os.makedirs(OUT_DIR, exist_ok=True)

    LATENT_DIM = 256
    WIDTH = 128
    N_BLOCKS = 4
    K_SPEC = 16

    EPOCHS = 50
    LR = 1e-4
    BATCH_SIZE = 8
    VAL_SPLIT = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Stage 2 (Frozen Encoder) on {device}")

    # ---------------- DATASET ----------------
    dataset = GTReadyDataset(DATA_DIR)
    dataset = patch_dataset_with_get_by_name(dataset)

    print(f"🧩 Dataset: {len(dataset)} meshes")

    n_samples = len(dataset)
    n_val = int(n_samples * VAL_SPLIT)
    n_train = n_samples - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_skip,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_skip,
    )

    # ---------------- MODEL ----------------
    model = EncoderFrozenWithDecoder(
        encoder_ckpt_path=ENCODER_CKPT,
        latent_dim=LATENT_DIM,
        width=WIDTH,
        n_blocks=N_BLOCKS,
        k_spec=K_SPEC,
        dropout_encoder=0.1,
        device=device,
    )

    # Train only decoder parameters
    optimizer = optim.Adam(
        model.decoder.parameters(), lr=LR, weight_decay=1e-6
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7
    )

    criterion = GeometricLoss(
        w_l1=0.3,
        w_normal=1.0,
        w_laplacian=0.7,
        device=device,
    )

    # ---------------- LOGGING ----------------
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(OUT_DIR, "runs", run_name))

    log_csv = os.path.join(OUT_DIR, "train_log.csv")
    with open(log_csv, "w") as f:
        f.write(
            "epoch,train_loss,val_loss,"
            "train_l1,val_l1,"
            "train_normal,val_normal,"
            "train_lapl,val_lapl,"
            "lr\n"
        )

    print("\n=== START STAGE 2 TRAINING (Frozen Encoder) ===")

    # ---------------- TRAIN LOOP ----------------
    for epoch in range(EPOCHS):
        model.decoder.train()

        sum_loss = 0.0
        train_l1_sum = 0.0
        train_normal_sum = 0.0
        train_lapl_sum = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch_list in pbar:
            optimizer.zero_grad(set_to_none=True)

            batch_loss = 0.0
            batch_l1 = 0.0
            batch_normal = 0.0
            batch_lapl = 0.0
            count = 0

            for sample in batch_list:
                # Forward: encoder (no grad) + decoder (trainable)
                V_gt = sample["verts"].to(device)

                V_rec = model(sample)  # model handles device move internally

                loss, breakdown = criterion(
                    V_rec, V_gt, sample["faces"].to(device), sample["L"].to(device)
                )

                if not torch.isfinite(loss):
                    continue

                batch_loss += loss
                batch_l1 += float(breakdown["loss_l1"])
                batch_normal += float(breakdown["loss_normal"])
                batch_lapl += float(breakdown["loss_laplacian"])
                count += 1

            if count == 0:
                continue

            loss_final = batch_loss / count
            avg_l1 = batch_l1 / count
            avg_normal = batch_normal / count
            avg_lapl = batch_lapl / count

            loss_final.backward()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
            optimizer.step()

            sum_loss += loss_final.item()
            train_l1_sum += avg_l1
            train_normal_sum += avg_normal
            train_lapl_sum += avg_lapl
            n_batches += 1

            pbar.set_postfix(loss=f"{loss_final.item():.4f}")

        if n_batches == 0:
            continue

        train_loss = sum_loss / n_batches
        train_l1 = train_l1_sum / n_batches
        train_normal = train_normal_sum / n_batches
        train_lapl = train_lapl_sum / n_batches

        # ---------------- VALIDATION ----------------
        model.decoder.eval()
        val_sum = 0.0
        val_l1_sum = 0.0
        val_normal_sum = 0.0
        val_lapl_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for sample_list in val_loader:
                if len(sample_list) == 0:
                    continue
                sample = sample_list[0]

                V_gt = sample["verts"].to(device)
                V_rec = model(sample)

                loss, breakdown = criterion(
                    V_rec, V_gt, sample["faces"].to(device), sample["L"].to(device)
                )

                if not torch.isfinite(loss):
                    continue

                val_sum += loss.item()
                val_l1_sum += float(breakdown["loss_l1"])
                val_normal_sum += float(breakdown["loss_normal"])
                val_lapl_sum += float(breakdown["loss_laplacian"])
                val_batches += 1

        val_loss = val_sum / max(1, val_batches)
        val_l1 = val_l1_sum / max(1, val_batches)
        val_normal = val_normal_sum / max(1, val_batches)
        val_lapl = val_lapl_sum / max(1, val_batches)

        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"Train L1/N/Lap: {train_l1:.6f}/{train_normal:.6f}/{train_lapl:.6f}")
        print(f"Val   L1/N/Lap: {val_l1:.6f}/{val_normal:.6f}/{val_lapl:.6f}")
        print(f"LR: {lr_now:.1e}")

        writer.add_scalar("loss/train", train_loss, epoch + 1)
        writer.add_scalar("loss/val", val_loss, epoch + 1)
        writer.add_scalar("loss_l1/train", train_l1, epoch + 1)
        writer.add_scalar("loss_l1/val", val_l1, epoch + 1)
        writer.add_scalar("loss_normal/train", train_normal, epoch + 1)
        writer.add_scalar("loss_normal/val", val_normal, epoch + 1)
        writer.add_scalar("loss_lapl/train", train_lapl, epoch + 1)
        writer.add_scalar("loss_lapl/val", val_lapl, epoch + 1)
        writer.add_scalar("lr", lr_now, epoch + 1)

        # CSV log
        with open(log_csv, "a") as f:
            f.write(
                f"{epoch+1},"
                f"{train_loss:.6f},{val_loss:.6f},"
                f"{train_l1:.6f},{val_l1:.6f},"
                f"{train_normal:.6f},{val_normal:.6f},"
                f"{train_lapl:.6f},{val_lapl:.6f},"
                f"{lr_now:.1e}\n"
            )

        # Checkpoint decoder
        ck = os.path.join(OUT_DIR, f"stage2_decoder_epoch{epoch+1}.pth")
        torch.save(model.decoder.state_dict(), ck)
        print(f"💾 Saved {ck}")

    print("\n🎉 DONE Stage 2 (Frozen Encoder)!")


if __name__ == "__main__":
    main()
