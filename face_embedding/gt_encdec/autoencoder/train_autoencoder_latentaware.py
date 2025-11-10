# train_autoencoder_latentaware.py
# Latent-Aware Diffusion Autoencoder — Batch Logging, Schedules, Latent Autocontrol
# Author: Leonardo Pampaloni — 2025

import os
import numpy as np
from datetime import datetime
import scipy.stats as st
import zipfile

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset
from diffusion_autoencoder import DiffusionAutoencoder
from geometric_loss import GeometricLoss
from latent_loss import varcov_loss, smooth_loss, stress_loss


# =========================
# Utils
# =========================
def collate_skip(batch):
    """Drop None samples the dataset may yield."""
    return [s for s in batch if s is not None]


def sample_distance_submatrix(D_full: np.ndarray, rowcol_idx: np.ndarray) -> torch.Tensor:
    """Return D_full[np.ix_(idx, idx)] as float32 torch tensor."""
    return torch.tensor(D_full[np.ix_(rowcol_idx, rowcol_idx)], dtype=torch.float32)


def build_name_index_map(names_from_npz):
    """Map filename → index (ignoring .npz suffix)."""
    mapping = {}
    for i, nm in enumerate(names_from_npz):
        nm = str(nm)
        base = nm[:-4] if nm.endswith(".npz") else nm
        mapping[base] = i
    return mapping


# =========================
# Schedules
# =========================
def linear_decay(epoch, start, end, total_epochs):
    t = min(max(epoch, 0), total_epochs)
    return start + (end - start) * (t / float(total_epochs))


def ramp_up(epoch, target, warmup_epochs):
    if warmup_epochs <= 0:
        return target
    return target * min(1.0, (epoch + 1) / float(warmup_epochs))


def capped_growth(epoch, start, grow, max_val):
    return min(max_val, start * (grow ** epoch))


# =========================
# Triplet loss (GT-structured)
# =========================
def triplet_loss(Z, D_gt, margin=0.2):
    """
    Simple triplet margin loss using GT distances to pick (pos, neg).
    Z: [B, d] normalized embeddings, D_gt: [B, B] in [0, 1].
    """
    n = Z.shape[0]
    if n < 3:
        return torch.tensor(0.0, device=Z.device)
    with torch.no_grad():
        eye = torch.eye(n, device=Z.device)
        pos_idx = torch.argmin(D_gt + eye * 1e9, dim=1)  # closest GT
        neg_idx = torch.argmax(D_gt, dim=1)              # farthest GT
    dist_lat = torch.cdist(Z, Z, p=2)
    ap = dist_lat[torch.arange(n), pos_idx]
    an = dist_lat[torch.arange(n), neg_idx]
    loss = torch.clamp(ap - an + margin, min=0.0).mean()
    return loss


# =========================
# Latent identity validation
# =========================
def latent_identity_check(model, dataset, D_orig, name_to_idx, device, n_samples=100):
    """Return pearson/spearman/R² + slope + MAE between GT and latent distances."""
    model.eval()
    if len(dataset) == 0:
        return None

    idxs = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    Z_list, id_list = [], []

    with torch.no_grad():
        for i in idxs:
            s = dataset[i]
            V = s["verts"].to(device)
            mass, evals, evecs = s["mass"].to(device), s["evals"].to(device), s["evecs"].to(device)
            faces, L = s["faces"].to(device), s["L"].to(device)
            gX, gY = s["gradX"].to(device), s["gradY"].to(device)
            out = model(V, mass, L, evals, evecs, faces, gX, gY)

            if isinstance(out, (tuple, list)):
                Zg = out[-1]
            else:
                continue
            if Zg.dim() == 1:
                Zg = Zg.unsqueeze(0)

            Z_list.append(Zg.cpu())
            base = s["name"][:-4] if s["name"].endswith(".npz") else s["name"]
            if base in name_to_idx:
                id_list.append(name_to_idx[base])

    if len(Z_list) < 2 or len(id_list) < 2:
        return None

    Z = torch.cat(Z_list, dim=0)  # [m, d]
    Z = Z - Z.mean(dim=0, keepdim=True)
    Z = Z / (Z.norm(dim=1, keepdim=True) + 1e-8)

    D_lat = torch.cdist(Z, Z, p=2).cpu().numpy()
    D_lat = (D_lat - D_lat.min()) / (D_lat.max() - D_lat.min() + 1e-8)

    idx_array = np.array(id_list)
    D_gt = D_orig[np.ix_(idx_array, idx_array)]
    D_gt = (D_gt - D_gt.min()) / (D_gt.max() - D_gt.min() + 1e-8)

    mask = np.triu_indices_from(D_gt, k=1)
    x, y = D_gt[mask], D_lat[mask]

    pear = st.pearsonr(x, y)[0]
    spear = st.spearmanr(x, y)[0]
    r2 = np.corrcoef(x, y)[0, 1] ** 2

    # slope via polyfit (no sklearn dependency)
    slope = float(np.polyfit(x, y, 1)[0])
    mae = float(np.mean(np.abs(x - y)))

    print(f"   🔎 Latent Identity → ρ_P={pear:.3f}, ρ_S={spear:.3f}, R²={r2:.3f}, slope={slope:.3f}, MAE={mae:.4f}")
    return {"pearson": pear, "spearman": spear, "r2": r2, "slope": slope, "mae": mae}


# =========================
# Main
# =========================
def main():
    torch.multiprocessing.set_start_method("spawn", force=True)

    DATA_DIR = "../../../datasets/GT_ready/npz_data/"
    DIST_PATH = "latent_analysis/dist_matrices_fields/D_orig_gt_normalized.npz"
    OUT_DIR = "./results_diffusionAE_latentaware/"
    os.makedirs(OUT_DIR, exist_ok=True)

    LATENT_DIM = 256
    WIDTH = 128
    N_BLOCKS = 4
    EPOCHS = 50
    LR = 1e-4
    BATCH_SIZE = 4
    VAL_SPLIT = 0.1
    CHECKPOINT_EVERY = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # geo components
    W_L1, W_NORMAL, W_LAPLACIAN = 0.3, 1.0, 0.7

    # schedules and weights
    GEO_W_START, GEO_W_END, GEO_W_EPOCHS = 1.0, 0.1, 8
    RANK_W_START, RANK_W_MAX, RANK_W_GROW = 1e-3, 0.2, 1.6
    LAMBDA_SMOOTH = 0.005
    TRIPLET_WEIGHT = 0.50

    # latent scheduling with autocontrol
    latent_base = 0.5
    LAT_W_WARMUP = 5
    TARGET_LATENT_FRAC = 0.70  # desired ratio L_lat / geo

    print(f"🚀 Training Latent-Aware Diffusion AE on {DEVICE}")
    print(f"⚙️ Geo loss (L1={W_L1}, N={W_NORMAL}, Lap={W_LAPLACIAN}) | Latent base={latent_base}")

    # ---------- Dataset ----------
    dataset = GTReadyDataset(DATA_DIR)

    valid_files = []
    skipped_csv = os.path.join(OUT_DIR, "skipped_files.csv")
    with open(skipped_csv, "w") as f:
        f.write("filename,error\n")

    for f in dataset.files[:5000]:
        path = f if os.path.isabs(f) else os.path.join(DATA_DIR, f)
        try:
            with np.load(path) as data:
                _ = data.files
            valid_files.append(f)
        except (zipfile.BadZipFile, OSError, EOFError, ValueError) as e:
            with open(skipped_csv, "a") as ff:
                ff.write(f"{f},{e}\n")
            print(f"[WARN] Skipping corrupted file: {f} ({e})")

    dataset.files = valid_files
    print(f"📊 Valid NPZ count: {len(dataset.files)}")
    if len(dataset.files) == 0:
        raise RuntimeError("❌ No valid .npz files found. Cannot start training.")

    n_val = int(len(dataset) * VAL_SPLIT)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_skip)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_skip)

    # ---------- Model/Loss/Opt ----------
    model = DiffusionAutoencoder(latent_dim=LATENT_DIM, width=WIDTH, n_blocks=N_BLOCKS).to(DEVICE)
    geom_loss = GeometricLoss(W_L1, W_NORMAL, W_LAPLACIAN, device=DEVICE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7)

    # ---------- GT distance matrix ----------
    print("📂 Loading GT distance matrix (D_orig)...")
    D_pack = np.load(DIST_PATH)
    D_orig = D_pack["D_orig"]
    norm_factor = np.max(D_orig[D_orig > 0]) if np.any(D_orig > 0) else 1.0
    D_orig = D_orig / norm_factor
    print(f"   → Normalized by factor {norm_factor:.4f}")

    name_to_idx = build_name_index_map([str(x) for x in D_pack["names"]])

    # ---------- Logging ----------
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(OUT_DIR, "runs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    log_csv = os.path.join(OUT_DIR, "train_log.csv")
    log_batches_csv = os.path.join(OUT_DIR, "train_batches.csv")
    with open(log_csv, "w") as f:
        f.write("epoch,total,geo,lat,rank,var,triplet,smooth,pearson,spearman,r2,slope,mae,lr,geo_w,lat_w,rank_w\n")
    with open(log_batches_csv, "w") as f:
        f.write("epoch,batch,geo,rank,var,triplet,smooth,L_lat,total,geo_w,latent_w,rank_w,lambda_smooth,triplet_w\n")

    # =========================
    # Training
    # =========================
    for epoch in range(EPOCHS):
        model.train()

        geo_weight = linear_decay(epoch, GEO_W_START, GEO_W_END, GEO_W_EPOCHS)
        latent_w = ramp_up(epoch, latent_base, LAT_W_WARMUP)
        rank_weight = capped_growth(epoch, RANK_W_START, RANK_W_GROW, RANK_W_MAX)

        print(f"\n🟢 [Epoch {epoch+1}/{EPOCHS}] Training...")
        print(f"   ⚖️ Weights: geo={geo_weight:.3f} | latent={latent_w:.3f} | rank={rank_weight:.4f} | λ_smooth={LAMBDA_SMOOTH:.3f} | triplet={TRIPLET_WEIGHT:.2f}")

        total, geo, lat, rank, var, trip, smooth = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Train {epoch+1}", ncols=120)
        for b_idx, batch_list in enumerate(pbar):
            if len(batch_list) == 0:
                continue
            optimizer.zero_grad(set_to_none=True)

            geo_loss_accum, Zg_list, batch_names, smooth_accum = 0.0, [], [], 0.0

            for sample in batch_list:
                V = sample["verts"].to(DEVICE)
                mass, evals, evecs = sample["mass"].to(DEVICE), sample["evals"].to(DEVICE), sample["evecs"].to(DEVICE)
                faces, L = sample["faces"].to(DEVICE), sample["L"].to(DEVICE)
                gX, gY = sample["gradX"].to(DEVICE), sample["gradY"].to(DEVICE)

                out = model(V, mass, L, evals, evecs, faces, gX, gY)
                if isinstance(out, (tuple, list)) and len(out) == 3:
                    V_rec, Z_field, Z_global = out
                else:
                    V_rec, Z_global = out if isinstance(out, (tuple, list)) else (out, None)
                    Z_field = None

                L_geo, _ = geom_loss(V_rec, V, faces, L)
                geo_loss_accum += L_geo

                if Z_global is not None and Z_global.dim() == 1:
                    Z_global = Z_global.unsqueeze(0)
                if Z_global is not None:
                    Zg_list.append(Z_global)
                batch_names.append(sample.get("name", ""))

                if Z_field is not None:
                    smooth_accum += smooth_loss(Z_field, L) / max(Z_field.shape[0], 1)

            if len(Zg_list) == 0:
                continue

            Zg_batch = torch.cat(Zg_list, dim=0)             # [B, d]
            Zg_batch = Zg_batch - Zg_batch.mean(dim=0, keepdim=True)
            Zg_batch = Zg_batch / (Zg_batch.norm(dim=1, keepdim=True) + 1e-8)

            idx_batch = [name_to_idx[nm[:-4] if nm.endswith(".npz") else nm]
                         for nm in batch_names if (nm[:-4] if nm.endswith(".npz") else nm) in name_to_idx]

            if len(idx_batch) < 2:
                continue
            D_batch = sample_distance_submatrix(D_orig, np.asarray(idx_batch)).to(DEVICE)

            L_rank = stress_loss(Zg_batch, D_batch)
            L_var = varcov_loss(Zg_batch)
            L_trip = triplet_loss(Zg_batch, D_batch, margin=0.2)
            L_smooth = smooth_accum / max(len(batch_list), 1)

            L_lat = rank_weight * L_rank + L_var + LAMBDA_SMOOTH * L_smooth + TRIPLET_WEIGHT * L_trip
            L_geo_batch = geo_loss_accum / len(batch_list)
            L_total = geo_weight * L_geo_batch + latent_w * L_lat

            if not torch.isfinite(L_total):
                continue

            L_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # accumulate
            total += float(L_total)
            geo += float(L_geo_batch)
            lat += float(L_lat)
            rank += float(L_rank)
            var += float(L_var)
            trip += float(L_trip)
            smooth += float(L_smooth)
            n_batches += 1

            # CSV per-batch
            with open(log_batches_csv, "a") as f:
                f.write(
                    f"{epoch+1},{b_idx},"
                    f"{float(L_geo_batch):.6f},{float(L_rank):.6f},{float(L_var):.6f},"
                    f"{float(L_trip):.6f},{float(L_smooth):.6f},{float(L_lat):.6f},"
                    f"{float(L_total):.6f},{geo_weight:.3f},{latent_w:.3f},{rank_weight:.4f},{LAMBDA_SMOOTH:.3f},{TRIPLET_WEIGHT:.2f}\n"
                )

            # pretty print every 25 batches
            if b_idx % 25 == 0:
                print(
                    f"\n🧩 Batch {b_idx:03d}\n"
                    f"   ├─ geo={geo/n_batches:.4f}\n"
                    f"   ├─ latent={lat/n_batches:.4f}\n"
                    f"   ├─ rank={rank/n_batches:.4f}\n"
                    f"   ├─ var={var/n_batches:.4f}\n"
                    f"   ├─ triplet={trip/n_batches:.4f}\n"
                    f"   ├─ smooth={smooth/n_batches:.4f}\n"
                    f"   ├─ total={total/n_batches:.4f}\n"
                    f"   └─ weights: geo_w={geo_weight:.3f}, latent_w={latent_w:.3f}, rank_w={rank_weight:.4f}"
                )

        # latent autocontrol: keep latent/geo near target
        if n_batches > 0:
            geo_avg = geo / n_batches
            lat_avg = lat / n_batches
            ratio = lat_avg / (geo_avg + 1e-8)
            if ratio < TARGET_LATENT_FRAC * 0.95:
                latent_base *= 1.05
            elif ratio > TARGET_LATENT_FRAC * 1.05:
                latent_base *= 0.98
            latent_base = float(np.clip(latent_base, 0.2, 1.5))

        # end-of-epoch logs
        train_total = total / max(n_batches, 1)
        train_geo = geo / max(n_batches, 1)
        train_lat = lat / max(n_batches, 1)
        train_rank = rank / max(n_batches, 1)
        train_var = var / max(n_batches, 1)
        train_trip = trip / max(n_batches, 1)
        train_smooth = smooth / max(n_batches, 1)

        print(f"✅ [Epoch {epoch+1}] Train → total={train_total:.4f} | geo={train_geo:.4f} | "
              f"latent={train_lat:.4f} | rank={train_rank:.4f} | var={train_var:.4f} | triplet={train_trip:.4f}")

        # validation identity
        stats = latent_identity_check(model, val_set, D_orig, name_to_idx, DEVICE) or {}
        scheduler.step(train_total)
        current_lr = optimizer.param_groups[0]["lr"]

        with open(log_csv, "a") as f:
            f.write(
                f"{epoch+1},{train_total:.6f},{train_geo:.6f},{train_lat:.6f},"
                f"{train_rank:.6f},{train_var:.6f},{train_trip:.6f},{train_smooth:.6f},"
                f"{stats.get('pearson', np.nan):.4f},{stats.get('spearman', np.nan):.4f},{stats.get('r2', np.nan):.4f},"
                f"{stats.get('slope', np.nan):.4f},{stats.get('mae', np.nan):.6f},{current_lr:.1e},"
                f"{geo_weight:.3f},{latent_w:.3f},{rank_weight:.4f}\n"
            )

        if (epoch + 1) % CHECKPOINT_EVERY == 0 or (epoch + 1) == EPOCHS:
            ckpt = os.path.join(OUT_DIR, f"latentaware_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"💾 Saved checkpoint: {ckpt}")

    writer.close()
    print("✅ Training complete.")


if __name__ == "__main__":
    main()
