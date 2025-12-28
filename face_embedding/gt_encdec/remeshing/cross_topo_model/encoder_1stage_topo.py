#!/usr/bin/env python3
import os
import re
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import sys

sys.path.append(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/gt_encdec/autoencoder"
)

from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset
from diffusion_autoencoder import DiffusionEncoderOnly

try:
    import diffusion_net
    DiffusionNet = diffusion_net.layers.DiffusionNet
except Exception:
    from diffusion_net import DiffusionNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from latent_loss import stress_loss, scale_loss  # scale_loss non serve se fai slope
from latent_loss import stress_loss, smooth_loss  # smooth_loss usa L e Z_field

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = (
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/REMESH/"
    "npz_data_topo_500_withops"
)

DIST_PATH = (
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/"
    "gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/"
    "normalized_matrix_distances.npz"
)

OUT_DIR = "encoder_stage1_multitopo_second_try"
os.makedirs(OUT_DIR, exist_ok=True)

# Model
LATENT_DIM = 256
WIDTH = 128
N_BLOCKS = 4
DROPOUT = 0.1

# Training
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-6
BATCH_SUBJECTS = 4   # batch in "numero soggetti", NON mesh
GRAD_CLIP = 1.0

# Loss weights
LAMBDA_STRESS = 0.3   # preserva struttura globale (tra soggetti)
LAMBDA_ID     = 0.1   # forza stesso soggetto (topologie diverse) vicino
LAMBDA_METRIC = 0.1   # inizia conservativo

# Varianti attese nel dataset (se ci sono meno non importa)
VARIANT_RE = re.compile(r"^(id\d+)_.*_(original|remesh|crop|noisy)\.npz$")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# EVAL (latent monitoring)
# ============================================================
EVAL_SUBJECTS = 80          # subset fisso per valutazione
EVAL_EVERY = 1              # ogni N epoche
EVAL_SEED = 12345


# ============================================================
# DATA GROUPING: subject -> indices of its variants
# ============================================================
def build_subject_map(dataset):
    subj_to_idxs = {}
    for idx, fname in enumerate(dataset.files):
        m = VARIANT_RE.match(fname)
        if not m:
            # fallback: subject = first token before "_" (compatibile)
            subj = fname.split("_")[0]
        else:
            subj = m.group(1)
        subj_to_idxs.setdefault(subj, []).append(idx)
    return subj_to_idxs


@torch.no_grad()
def eval_latent_structure(
    model,
    dataset,
    subj_map,
    eval_subjects,
    name_to_idx,
    D_orig,
    device,
):
    model.eval()

    subj_mean = {}
    intra_vals = []

    # --------------------------------------------------
    # Encode subjects (all variants → mean embedding)
    # --------------------------------------------------
    for subj in eval_subjects:
        idxs = subj_map[subj]
        Zs = []

        for idx in idxs:
            sample = dataset[idx]

            V = sample["verts"].to(device)
            mass = sample["mass"].to(device)
            L = sample["L"].to(device)
            evals = sample["evals"].to(device)
            evecs = sample["evecs"].to(device)
            faces = sample["faces"].to(device)
            gradX = sample["gradX"].to(device)
            gradY = sample["gradY"].to(device)

            Zg = model(
                V, mass, L, evals, evecs,
                faces, gradX, gradY,
                return_per_vertex=False,
                add_noise=False
            ).squeeze(0)

            Zs.append(Zg)

        Zs = torch.stack(Zs, dim=0)        # (K,D)
        zm = Zs.mean(dim=0)                # (D,)

        subj_mean[subj] = zm
        intra_vals.append(((Zs - zm) ** 2).mean().item())

    # --------------------------------------------------
    # INTER-subject distances vs GT
    # --------------------------------------------------
    kept = [s for s in eval_subjects if s in name_to_idx]
    if len(kept) < 3:
        return None

    Zmat = torch.stack([subj_mean[s] for s in kept], dim=0)  # (S,D)
    idx = np.array([name_to_idx[s] for s in kept], dtype=int)

    D_gt = torch.tensor(
        D_orig[np.ix_(idx, idx)],
        device=device,
        dtype=Zmat.dtype
    )

    D_lat = torch.cdist(Zmat, Zmat)

    iu = torch.triu_indices(D_gt.size(0), D_gt.size(1), offset=1)
    gt = D_gt[iu[0], iu[1]].cpu().numpy()
    lat = D_lat[iu[0], iu[1]].cpu().numpy()

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    pearson = np.corrcoef(gt, lat)[0, 1]

    spearman = np.corrcoef(
        gt.argsort().argsort(),
        lat.argsort().argsort()
    )[0, 1]

    A = np.vstack([gt, np.ones_like(gt)]).T
    slope, intercept = np.linalg.lstsq(A, lat, rcond=None)[0]

    ss_res = ((lat - (slope * gt + intercept)) ** 2).sum()
    ss_tot = ((lat - lat.mean()) ** 2).sum() + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    intra_vals_np = np.array(intra_vals)

    intra_mean   = float(intra_vals_np.mean())
    intra_median = float(np.median(intra_vals_np))
    intra_p90    = float(np.percentile(intra_vals_np, 90))
    intra_max    = float(intra_vals_np.max())

    return {
        "pearson": float(pearson),
        "spearman": float(spearman),
        "r2": float(r2),
        "slope": float(slope),
        "intercept": float(intercept),
        "intra_mean": intra_mean,
        "intra_median": intra_median,
        "intra_p90": intra_p90,
        "intra_max": intra_max,
    }

# ============================================================
# MAIN
# ============================================================
def main():
    print(f"🚀 Stage-1 Encoder training (multi-topology) on {DEVICE}")

    # ----------------------------
    # Dataset
    # ----------------------------
    dataset = GTReadyDataset(DATA_DIR)
    subj_map = build_subject_map(dataset)
    subjects = sorted(map(str, subj_map.keys()))

    # ----------------------------
    # Train / Eval split (BY SUBJECT)
    # ----------------------------
    rng = np.random.default_rng(EVAL_SEED)

    subjects = np.array(subjects, dtype=str)
    n_total = len(subjects)
    n_eval = int(0.2 * n_total)   # 20% eval

    eval_subjects = rng.choice(subjects, size=n_eval, replace=False)
    train_subjects = np.array(
        [s for s in subjects if s not in set(eval_subjects)],
        dtype=str
    )

    # forza liste ordinate (riproducibilità)
    eval_subjects = sorted(eval_subjects.tolist())
    train_subjects = sorted(train_subjects.tolist())

    # ----------------------------
    # HARD CHECKS (NO LEAKAGE)
    # ----------------------------
    overlap = set(train_subjects) & set(eval_subjects)
    print(f"❌ OVERLAP train/eval: {len(overlap)}")
    if overlap:
        print("Example overlap:", list(overlap)[:5])
        raise RuntimeError("Train/Eval leakage detected!")

    print(f"Train subjects: {len(train_subjects)}")
    print(f"Eval subjects : {len(eval_subjects)}")
    print(f"Total subjects: {len(subjects)} | Meshes: {len(dataset.files)}")
    print(f"Example subject variants: {len(subj_map[train_subjects[0]])}")

    # ----------------------------
    # Load GT distance matrix (between subjects)
    # ----------------------------
    print("📂 Loading GT distance matrix...")
    D_pack = np.load(DIST_PATH, allow_pickle=True)
    D_orig = D_pack["D_orig"].astype(np.float64)
    D_orig /= np.max(D_orig[D_orig > 0])

    # names può essere tipo "id0000_GTready" oppure "id0000"
    # Noi mappiamo usando il prefisso idXXXX (robusto)
    names = [str(n) for n in D_pack["names"]]
    name_to_idx = {}
    for i, n in enumerate(names):
        # prova a estrarre idXXXX
        m = re.search(r"(id\d{4})", n)
        if m:
            name_to_idx[m.group(1)] = i

    if len(name_to_idx) == 0:
        raise RuntimeError("name_to_idx is empty: couldn't parse subject ids from D_pack['names'].")

    # ----------------------------
    # Model
    # ----------------------------
    model = DiffusionEncoderOnly(
        latent_dim=LATENT_DIM,
        width=WIDTH,
        n_blocks=N_BLOCKS,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3)

    # ----------------------------
    # Logging
    # ----------------------------
    log_csv = os.path.join(OUT_DIR, "train_log.csv")

    with open(log_csv, "w") as f:
        f.write(
            "epoch,loss,stress,id,lr,"
            "pearson,spearman,r2,fit_slope,fit_intercept,"
            "intra_mean,intra_median,intra_p90,intra_max\n"
        )


    # =========================================================
    # TRAINING LOOP
    # =========================================================
    # Sanity accumulators (debug)
    SANITY_EVERY = 1   # log ogni N step (1 = sempre)

    for epoch in range(EPOCHS):
        model.train()
        rng = np.random.default_rng(epoch + 123)

        # shuffle subjects each epoch
        subjects_shuf = rng.permutation(train_subjects)

        epoch_loss = 0.0
        n_steps = 0

        pbar = tqdm(range(0, len(subjects_shuf), BATCH_SUBJECTS),
                    desc=f"Epoch {epoch+1}/{EPOCHS}")

        for start in pbar:
            batch_subjects = subjects_shuf[start:start + BATCH_SUBJECTS]
            if len(batch_subjects) < 2:
                continue  # stress richiede almeno 2 soggetti

            optimizer.zero_grad(set_to_none=True)

            # Per ogni soggetto: lista embeddings per-variant (global) + smooth field
            subj_Zglobals = {}   # subj -> list[(1,D)]
            subj_Zfields  = []   # list[Z_field (N,D)] per mesh (per smooth)
            subj_Ls       = []   # list[L] per mesh

            # -------------- forward su TUTTE le varianti dei soggetti nel batch
            for subj in batch_subjects:
                idxs = subj_map[subj]
                Zg_list = []

                for idx in idxs:
                    sample = dataset[idx]

                    V = sample["verts"].to(DEVICE)
                    mass = sample["mass"].to(DEVICE)
                    L = sample["L"].to(DEVICE)
                    evals = sample["evals"].to(DEVICE)
                    evecs = sample["evecs"].to(DEVICE)
                    faces = sample["faces"].to(DEVICE)
                    gradX = sample["gradX"].to(DEVICE)
                    gradY = sample["gradY"].to(DEVICE)

                    # QUI È IL PUNTO:
                    # calcoliamo PER-VERTEX e poi global,
                    # ma la loss identity la facciamo sul global.
                    Z_field, Z_global = model(
                        V, mass, L, evals, evecs,
                        faces, gradX, gradY,
                        return_per_vertex=True,
                        add_noise=True
                    )

                    Zg_list.append(Z_global)   # (1,D)
                    subj_Zfields.append(Z_field)
                    subj_Ls.append(L)

                subj_Zglobals[subj] = Zg_list

            # -------------------------------------------------------
            # (A) Identity loss: within-subject collapse (cross-topology)
            #     Minimize variance of Z_global across variants
            # -------------------------------------------------------
            loss_id = torch.tensor(0.0, device=DEVICE)
            count_id = 0

            for subj, Zg_list in subj_Zglobals.items():
                if len(Zg_list) < 2:
                    continue
                Zs = torch.cat(Zg_list, dim=0)            # (K,D)
                Zm = Zs.mean(dim=0, keepdim=True)         # (1,D)
                loss_id = loss_id + ((Zs - Zm) ** 2).mean()
                count_id += 1

            if count_id > 0:
                loss_id = loss_id / count_id

            # -------------------------------------------------------
            # (B) Stress loss: preserve inter-subject structure
            #     Use ONE embedding per subject: mean across variants
            # -------------------------------------------------------
            subj_means = []
            subj_gt_idx = []

            for subj in batch_subjects:
                if subj not in subj_Zglobals:
                    continue
                Zs = torch.cat(subj_Zglobals[subj], dim=0)     # (K,D)
                Zm = Zs.mean(dim=0, keepdim=True)              # (1,D)
                subj_means.append(Zm)

                if subj in name_to_idx:
                    subj_gt_idx.append(name_to_idx[subj])
                else:
                    # se manca, droppiamo il soggetto per stress (evita casino)
                    subj_gt_idx.append(None)

            # filtra quelli che hanno GT idx
            keep = [i for i, gi in enumerate(subj_gt_idx) if gi is not None]
            if len(keep) >= 2:
                Z_batch = torch.cat([subj_means[i] for i in keep], dim=0)  # (B,D)
                idx_np = np.array([subj_gt_idx[i] for i in keep], dtype=int)
                # ========================================================
                # SANITY CHECKS (LATENT SPACE HEALTH)
                # ========================================================
                with torch.no_grad():
                    # 1) Norm of latent vectors (collapse / explosion check)
                    z_norms = Z_batch.norm(dim=1)
                    z_norm_mean = z_norms.mean().item()
                    z_norm_std  = z_norms.std().item()

                    # 2) Intra-subject distances (identity strength)
                    intra_dists = []
                    for subj, Zg_list in subj_Zglobals.items():
                        if len(Zg_list) >= 2:
                            Zs = torch.cat(Zg_list, dim=0)  # (K,D)
                            d = torch.pdist(Zs, p=2)
                            intra_dists.append(d.mean())

                    intra_mean = (
                        torch.stack(intra_dists).mean().item()
                        if len(intra_dists) > 0 else 0.0
                    )

                D_batch = torch.tensor(
                    D_orig[np.ix_(idx_np, idx_np)],
                    dtype=Z_batch.dtype,
                    device=Z_batch.device
                )

                loss_stress = stress_loss(Z_batch, D_batch)
            else:
                loss_stress = torch.tensor(0.0, device=DEVICE)




            # -------------------------------------------------------
            # TOTAL
            # -------------------------------------------------------
            loss = (
                LAMBDA_STRESS * loss_stress +
                LAMBDA_ID     * loss_id 
            )


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            n_steps += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                stress=f"{loss_stress.item():.4f}",
                id=f"{loss_id.item():.4f}",
                zmean=f"{z_norm_mean:.2f}",
                zstd=f"{z_norm_std:.2f}",
                intra=f"{intra_mean:.2f}",
            )



        # epoch end
        epoch_loss = epoch_loss / max(1, n_steps)
        scheduler.step(epoch_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"\n🧠 Epoch {epoch+1}/{EPOCHS}")
        print(f"   Loss: {epoch_loss:.6f} | LR: {lr_now:.1e}")



        # =========================================================
        # EVAL + CSV LOG (single row per epoch)
        # =========================================================
        metrics = None
        if (epoch + 1) % EVAL_EVERY == 0:
            metrics = eval_latent_structure(
                model,
                dataset,
                subj_map,
                eval_subjects,
                name_to_idx,
                D_orig,
                DEVICE,
            )

            if metrics is not None:
                print(
                    f"📊 Eval | "
                    f"Spear={metrics['spearman']:.3f} | "
                    f"Pear={metrics['pearson']:.3f} | "
                    f"R2={metrics['r2']:.3f} | "
                    f"Slope={metrics['slope']:.2f} | "
                    f"Intra={metrics['intra_mean']:.4f}"
                )

        with open(log_csv, "a") as f:
            if metrics is not None:
                f.write(
                    f"{epoch+1},"
                    f"{epoch_loss:.6f},"
                    f"{loss_stress.item():.6f},"
                    f"{loss_id.item():.6f},"
                    f"{lr_now:.2e},"
                    f"{metrics['pearson']:.4f},"
                    f"{metrics['spearman']:.4f},"
                    f"{metrics['r2']:.4f},"
                    f"{metrics['slope']:.4f},"
                    f"{metrics['intercept']:.4f},"
                    f"{metrics['intra_mean']:.6f},"
                    f"{metrics['intra_median']:.6f},"
                    f"{metrics['intra_p90']:.6f},"
                    f"{metrics['intra_max']:.6f}\n"
                )


            
        if ((epoch + 1) % 5 == 0) or (epoch + 1 == EPOCHS):
            ckpt = os.path.join(OUT_DIR, f"encoder_stage1_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"💾 Saved checkpoint: {ckpt}")

    print("\n✅ DONE.")


if __name__ == "__main__":
    main()
