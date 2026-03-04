#!/usr/bin/env python3
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

sys.path.append(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/gt_encdec/autoencoder"
)

from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset
from diffusion_autoencoder import DiffusionEncoderOnly
from latent_loss import stress_loss

try:
    import diffusion_net
    DiffusionNet = diffusion_net.layers.DiffusionNet
except Exception:
    from diffusion_net import DiffusionNet


# ============================================================
# CONFIG
# ============================================================
DATA_DIR  = "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/REMESH/npz_data_topo_500_withops"
DIST_PATH = (
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/"
    "gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz"
)
OUT_DIR = "encoder_pos_invariant_results"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
LATENT_DIM = 256
WIDTH      = 128
N_BLOCKS   = 4
DROPOUT    = 0.1

# Training
EPOCHS         = 50
LR             = 1e-4
WEIGHT_DECAY   = 1e-6
BATCH_SUBJECTS = 24
GRAD_CLIP      = 1.0

# Loss weights
LAMBDA_STRESS = 2.0
LAMBDA_ID     = 0.1
LAMBDA_SE3    = 0.2
K_AUG         = 4

# Eval
EVAL_EVERY = 1
EVAL_SEED  = 12345

VARIANT_RE = re.compile(r"^(id\d+)_.*_(original|remesh|crop|noisy)\.npz$")


# ============================================================
# HELPERS
# ============================================================
def build_subject_map(dataset):
    subj_to_idxs = {}
    for idx, fname in enumerate(dataset.files):
        m = VARIANT_RE.match(fname)
        subj = m.group(1) if m else fname.split("_")[0]
        subj_to_idxs.setdefault(subj, []).append(idx)
    return subj_to_idxs


def load_sample(sample, device):
    keys = ["verts", "mass", "L", "evals", "evecs", "faces", "gradX", "gradY"]
    return [sample[k].to(device) for k in keys]


def random_se3_transform(V, rot_strength=1.0, trans_strength=0.5):
    axis  = torch.randn(3, device=V.device)
    axis  = axis / axis.norm()
    angle = rot_strength * torch.rand(1, device=V.device) * 2 * torch.pi
    K = torch.tensor([
        [0,       -axis[2],  axis[1]],
        [axis[2],  0,       -axis[0]],
        [-axis[1], axis[0],  0      ],
    ], device=V.device)
    R = torch.eye(3, device=V.device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
    return V @ R.T + trans_strength * torch.randn(1, 3, device=V.device)


# ============================================================
# EVAL
# ============================================================
@torch.no_grad()
def eval_latent_structure(model, dataset, subj_map, eval_subjects, name_to_idx, D_orig, device):
    model.eval()

    subj_mean  = {}
    intra_vals = []

    for subj in eval_subjects:
        Zs = []
        for idx in subj_map[subj]:
            V, mass, L, evals, evecs, faces, gradX, gradY = load_sample(dataset[idx], device)
            Zg = model(
                V, mass, L, evals, evecs, faces, gradX, gradY,
                return_per_vertex=False, add_noise=False
            ).squeeze(0)
            Zs.append(Zg)

        Zs = torch.stack(Zs, dim=0)
        zm = Zs.mean(dim=0)
        subj_mean[subj] = zm
        intra_vals.append(((Zs - zm) ** 2).mean().item())

    kept = [s for s in eval_subjects if s in name_to_idx]
    if len(kept) < 3:
        return None

    Zmat     = torch.stack([subj_mean[s] for s in kept], dim=0)
    idx      = np.array([name_to_idx[s] for s in kept], dtype=int)
    D_gt     = torch.tensor(D_orig[np.ix_(idx, idx)], device=device, dtype=Zmat.dtype)

    # L2-normalize before distance computation
    Zmat_norm = F.normalize(Zmat, dim=1)
    D_lat     = torch.cdist(Zmat_norm, Zmat_norm)

    iu  = torch.triu_indices(D_gt.size(0), D_gt.size(1), offset=1)
    gt  = D_gt[iu[0], iu[1]].cpu().numpy()
    lat = D_lat[iu[0], iu[1]].cpu().numpy()

    pearson  = float(np.corrcoef(gt, lat)[0, 1])
    spearman = float(np.corrcoef(gt.argsort().argsort(), lat.argsort().argsort())[0, 1])

    A                = np.vstack([gt, np.ones_like(gt)]).T
    slope, intercept = np.linalg.lstsq(A, lat, rcond=None)[0]
    ss_res           = ((lat - (slope * gt + intercept)) ** 2).sum()
    ss_tot           = ((lat - lat.mean()) ** 2).sum() + 1e-12
    r2               = float(1.0 - ss_res / ss_tot)

    intra = np.array(intra_vals)
    return {
        "pearson":      pearson,
        "spearman":     spearman,
        "r2":           r2,
        "slope":        float(slope),
        "intercept":    float(intercept),
        "intra_mean":   float(intra.mean()),
        "intra_median": float(np.median(intra)),
        "intra_p90":    float(np.percentile(intra, 90)),
        "intra_max":    float(intra.max()),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Device: {DEVICE}")

    dataset    = GTReadyDataset(DATA_DIR)
    subj_map   = build_subject_map(dataset)
    subjects   = np.array(sorted(map(str, subj_map.keys())), dtype=str)

    rng            = np.random.default_rng(EVAL_SEED)
    n_eval         = int(0.2 * len(subjects))
    eval_subjects  = sorted(rng.choice(subjects, size=n_eval, replace=False).tolist())
    train_subjects = sorted([s for s in subjects if s not in set(eval_subjects)])

    assert not (set(train_subjects) & set(eval_subjects)), "Train/eval leakage detected!"
    print(f"Train: {len(train_subjects)} | Eval: {len(eval_subjects)} | Meshes: {len(dataset.files)}")

    D_pack     = np.load(DIST_PATH, allow_pickle=True)
    D_orig     = D_pack["D_orig"].astype(np.float64)
    D_orig    /= np.max(D_orig[D_orig > 0])
    name_to_idx = {}
    for i, n in enumerate([str(x) for x in D_pack["names"]]):
        m = re.search(r"(id\d{4})", n)
        if m:
            name_to_idx[m.group(1)] = i
    assert name_to_idx, "Could not parse subject ids from distance matrix."

    model = DiffusionEncoderOnly(
        latent_dim=LATENT_DIM, width=WIDTH, n_blocks=N_BLOCKS, dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=7)

    log_csv = os.path.join(OUT_DIR, "train_log.csv")
    with open(log_csv, "w") as f:
        f.write("epoch,loss,stress,id,lr,"
                "pearson,spearman,r2,fit_slope,fit_intercept,"
                "intra_mean,intra_median,intra_p90,intra_max\n")

    def model_fwd(*inputs):
        return model(*inputs, return_per_vertex=True, add_noise=True)

    for epoch in range(EPOCHS):
        model.train()
        rng           = np.random.default_rng(epoch + 123)
        subjects_shuf = rng.permutation(train_subjects)
        epoch_loss    = 0.0
        n_steps       = 0
        z_norm_mean   = z_norm_std = intra_mean = 0.0

        pbar = tqdm(range(0, len(subjects_shuf), BATCH_SUBJECTS), desc=f"Epoch {epoch+1}/{EPOCHS}")

        for start in pbar:
            batch_subjects = subjects_shuf[start:start + BATCH_SUBJECTS]
            if len(batch_subjects) < 2:
                continue

            optimizer.zero_grad(set_to_none=True)

            subj_Zglobals = {}
            loss_se3      = torch.tensor(0.0, device=DEVICE)
            count_se3     = 0

            for subj in batch_subjects:
                Zg_list = []
                for idx in subj_map[subj]:
                    V, mass, L, evals, evecs, faces, gradX, gradY = load_sample(dataset[idx], DEVICE)

                    V_ref = random_se3_transform(V) if model.training else V

                    # Checkpointed reference pass (tracked for backprop)
                    _, Z_global_ref = checkpoint(
                        model_fwd,
                        V_ref, mass, L, evals, evecs, faces, gradX, gradY,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )

                    # SE(3) augmentations: no_grad, compare against detached reference
                    Z_ref_detached  = Z_global_ref.detach()
                    loss_se3_local  = 0.0
                    with torch.no_grad():
                        for _ in range(K_AUG - 1):
                            V_aug = random_se3_transform(V) if model.training else V
                            _, Z_aug = model_fwd(V_aug, mass, L, evals, evecs, faces, gradX, gradY)
                            loss_se3_local += F.mse_loss(Z_aug, Z_ref_detached)

                    if K_AUG > 1:
                        loss_se3  = loss_se3 + loss_se3_local / (K_AUG - 1)
                        count_se3 += 1

                    Zg_list.append(Z_global_ref)
                subj_Zglobals[subj] = Zg_list

            if count_se3 > 0:
                loss_se3 = loss_se3 / count_se3  # normalize over subjects

            # Identity loss: minimize intra-subject variance across topology variants
            loss_id  = torch.tensor(0.0, device=DEVICE)
            count_id = 0
            for Zg_list in subj_Zglobals.values():
                if len(Zg_list) < 2:
                    continue
                Zs       = torch.cat(Zg_list, dim=0)
                loss_id += ((Zs - Zs.mean(dim=0, keepdim=True)) ** 2).mean()
                count_id += 1
            if count_id > 0:
                loss_id = loss_id / count_id

            # Stress loss: preserve inter-subject distance structure
            subj_means  = []
            subj_gt_idx = []
            for subj in batch_subjects:
                if subj not in subj_Zglobals:
                    continue
                Zs = torch.cat(subj_Zglobals[subj], dim=0)
                subj_means.append(Zs.mean(dim=0, keepdim=True))
                subj_gt_idx.append(name_to_idx.get(subj, None))

            keep        = [i for i, gi in enumerate(subj_gt_idx) if gi is not None]
            loss_stress = torch.tensor(0.0, device=DEVICE)

            if len(keep) >= 2:
                Z_batch  = torch.cat([subj_means[i] for i in keep], dim=0)
                idx_np   = np.array([subj_gt_idx[i] for i in keep], dtype=int)

                with torch.no_grad():
                    z_norms     = Z_batch.norm(dim=1)
                    z_norm_mean = z_norms.mean().item()
                    z_norm_std  = z_norms.std().item()
                    intra_dists = [
                        torch.pdist(torch.cat(Zg_list, dim=0), p=2).mean()
                        for Zg_list in subj_Zglobals.values()
                        if len(Zg_list) >= 2
                    ]
                    intra_mean = torch.stack(intra_dists).mean().item() if intra_dists else 0.0

                # L2-normalize: aligns latent distances [0,2] with GT [0,1]
                Z_batch_norm = F.normalize(Z_batch, dim=1)
                D_batch      = torch.tensor(
                    D_orig[np.ix_(idx_np, idx_np)],
                    dtype=Z_batch.dtype, device=Z_batch.device
                )
                loss_stress = stress_loss(Z_batch_norm, D_batch)

            loss = LAMBDA_STRESS * loss_stress + LAMBDA_ID * loss_id + LAMBDA_SE3 * loss_se3
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            n_steps    += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                stress=f"{loss_stress.item():.4f}",
                id=f"{loss_id.item():.4f}",
                se3=f"{loss_se3.item():.4f}",
                zmean=f"{z_norm_mean:.2f}",
                zstd=f"{z_norm_std:.2f}",
                intra=f"{intra_mean:.4f}",
            )

        epoch_loss /= max(1, n_steps)
        scheduler.step(epoch_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.6f} | LR: {lr_now:.1e}")

        metrics = None
        if (epoch + 1) % EVAL_EVERY == 0:
            metrics = eval_latent_structure(
                model, dataset, subj_map, eval_subjects, name_to_idx, D_orig, DEVICE
            )
            if metrics is not None:
                print(
                    f"Eval | Pear={metrics['pearson']:.3f} | Spear={metrics['spearman']:.3f} | "
                    f"R2={metrics['r2']:.3f} | Slope={metrics['slope']:.2f} | "
                    f"Intra={metrics['intra_mean']:.4f}"
                )

        with open(log_csv, "a") as f:
            if metrics is not None:
                f.write(
                    f"{epoch+1},{epoch_loss:.6f},{loss_stress.item():.6f},{loss_id.item():.6f},"
                    f"{lr_now:.2e},{metrics['pearson']:.4f},{metrics['spearman']:.4f},"
                    f"{metrics['r2']:.4f},{metrics['slope']:.4f},{metrics['intercept']:.4f},"
                    f"{metrics['intra_mean']:.6f},{metrics['intra_median']:.6f},"
                    f"{metrics['intra_p90']:.6f},{metrics['intra_max']:.6f}\n"
                )

        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
            ckpt = os.path.join(OUT_DIR, f"encoder_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"Checkpoint: {ckpt}")

    print("Done.")


if __name__ == "__main__":
    main()
