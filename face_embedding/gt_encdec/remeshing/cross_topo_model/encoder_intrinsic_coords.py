#!/usr/bin/env python3
import os
import re
import sys
import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.append(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/gt_encdec/autoencoder"
)

from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset
from diffusion_autoencoder import DiffusionEncoderOnlyIntrinsec as DiffusionEncoderOnly


# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/REMESH/npz_data_topo_500_withops"
DIST_PATH = (
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/"
    "gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz"
)

OUT_DIR = "encoder_intrinsic"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = False  # must stay False with sparse ops in DiffusionNet

# Model
LATENT_DIM = 128
WIDTH = 64
N_BLOCKS = 4
DROPOUT = 0.1
N_HKS = 16
N_WKS = 16

# Training
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-6
BATCH_SUBJECTS = 9
GRAD_CLIP = 1.0

# Loss weights
LAMBDA_DISTORT = 2.0
LAMBDA_ID = 0.1

# VICReg-style regularizers (keep secondary)
LAMBDA_VAR = 0.1
LAMBDA_COV = 0.01
VICREG_EPS = 1e-4
VICREG_GAMMA = 1.0 / math.sqrt(LATENT_DIM)

# Distortion loss numerics (paper-style normalization uses eps on sum(dH)) [page:1]
DIST_EPS_SUM_DH = 1e-6
DIST_MIN_GT = 1e-4     # keep low; paper's anti-collapse relies on ratio exploding when dH->0 [page:1]
DIST_SCALE_CLAMP = 1e4 # safety clamp

# Eval
EVAL_EVERY = 1
EVAL_SEED = 12345
EVAL_DIAG_SUBJECTS = 48

# Logging frequency
STEP_DIAG_EVERY = 0

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


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    d = x.size(0)
    return x.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()


def vicreg_var_loss(Z: torch.Tensor, gamma: float, eps: float) -> torch.Tensor:
    std = torch.sqrt(Z.var(dim=0, unbiased=False) + eps)
    return torch.relu(gamma - std).mean()


def vicreg_cov_loss(Z: torch.Tensor) -> torch.Tensor:
    B, D = Z.shape
    if B < 2:
        return torch.tensor(0.0, device=Z.device, dtype=Z.dtype)
    Zc = Z - Z.mean(dim=0, keepdim=True)
    cov = (Zc.T @ Zc) / (B - 1)
    return (off_diagonal(cov) ** 2).mean()


def distortion_ratio_loss_normalized(
    Z: torch.Tensor,
    D_gt: torch.Tensor,
    min_gt: float = 1e-4,
    eps_sum_dh: float = 1e-6,
    scale_clamp: float = 1e4,
    return_diag: bool = False,
):
    """
    Implements the paper-style normalized ratio:
      rho(i,j) = (dH(i,j) / dS(i,j)) * (sum dS / (sum dH + eps))
    and L_rho = mean_{i<j} (rho - 1)^2, with masking on small dS. [page:1]

    dH: cosine dissimilarity = 1 - <z_i,z_j> / (||z_i|| ||z_j||). [page:1]
    """
    ZN = F.normalize(Z, dim=1)
    S = (ZN @ ZN.T).clamp(-1.0, 1.0)
    dH = (1.0 - S).clamp_min(0.0)  # [0,2]

    iu = torch.triu_indices(D_gt.size(0), D_gt.size(1), offset=1, device=D_gt.device)
    dH_ = dH[iu[0], iu[1]]
    dS_ = D_gt[iu[0], iu[1]]

    mask = dS_ > min_gt
    if mask.sum().item() == 0:
        out = torch.tensor(0.0, device=Z.device, dtype=Z.dtype)
        if return_diag:
            return out, {"n_pairs": 0, "scale": 0.0, "dH_mean": 0.0, "dS_mean": 0.0}
        return out

    dH_m = dH_[mask]
    dS_m = dS_[mask]

    sum_dH = dH_m.sum()
    sum_dS = dS_m.sum()

    scale = (sum_dS / (sum_dH + eps_sum_dh)).clamp(max=scale_clamp)
    rho = (dH_m / dS_m) * scale

    loss = ((rho - 1.0) ** 2).mean()

    if return_diag:
        diag = {
            "n_pairs": int(mask.sum().item()),
            "scale": float(scale.item()),
            "dH_mean": float(dH_m.mean().item()),
            "dS_mean": float(dS_m.mean().item()),
            "rho_mean": float(rho.mean().item()),
        }
        return loss, diag

    return loss


@torch.no_grad()
def grad_norm(model) -> float:
    tot = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if g.is_sparse:
            g = g.coalesce().values()
        tot += g.float().norm().item() ** 2
    return float(tot ** 0.5)


@torch.no_grad()
def batch_latent_health(Z: torch.Tensor) -> dict:
    zn = Z.norm(dim=1)
    ZN = F.normalize(Z, dim=1)
    sim = ZN @ ZN.T
    iu = torch.triu_indices(sim.size(0), sim.size(1), 1, device=Z.device)
    sim_vals = sim[iu[0], iu[1]] if iu.numel() > 0 else torch.tensor([0.0], device=Z.device)

    std_dim = torch.sqrt(Z.var(dim=0, unbiased=False).clamp_min(0) + 1e-8)

    return {
        "z_norm_mean": zn.mean().item(),
        "z_norm_std": zn.std().item(),
        "z_std_dim_min": std_dim.min().item(),
        "z_std_dim_mean": std_dim.mean().item(),
        "cos_sim_mean": sim_vals.mean().item(),
        "cos_sim_std": sim_vals.std().item(),
    }


# ============================================================
# EVAL
# ============================================================
@torch.no_grad()
def eval_latent_structure(model, dataset, subj_map, eval_subjects, name_to_idx, D_orig, device):
    model.eval()

    subj_mean = {}
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

    Zmat = torch.stack([subj_mean[s] for s in kept], dim=0)
    idx = np.array([name_to_idx[s] for s in kept], dtype=int)

    D_gt = torch.tensor(D_orig[np.ix_(idx, idx)], device=device, dtype=Zmat.dtype)

    # cosine dissimilarity to match training dH [page:1]
    ZN = F.normalize(Zmat, dim=1)
    S = (ZN @ ZN.T).clamp(-1.0, 1.0)
    D_lat = (1.0 - S).clamp_min(0.0)

    iu = torch.triu_indices(D_gt.size(0), D_gt.size(1), offset=1, device=device)
    gt = D_gt[iu[0], iu[1]].detach().cpu().numpy()
    lat = D_lat[iu[0], iu[1]].detach().cpu().numpy()

    pearson = float(np.corrcoef(gt, lat)[0, 1])
    spearman = float(np.corrcoef(gt.argsort().argsort(), lat.argsort().argsort())[0, 1])

    A = np.vstack([gt, np.ones_like(gt)]).T
    slope, intercept = np.linalg.lstsq(A, lat, rcond=None)[0]

    ss_res = ((lat - (slope * gt + intercept)) ** 2).sum()
    ss_tot = ((lat - lat.mean()) ** 2).sum() + 1e-12
    r2 = float(1.0 - ss_res / ss_tot)

    intra = np.array(intra_vals)
    return {
        "pearson": pearson,
        "spearman": spearman,
        "r2": r2,
        "slope": float(slope),
        "intercept": float(intercept),
        "intra_mean": float(intra.mean()),
        "intra_median": float(np.median(intra)),
        "intra_p90": float(np.percentile(intra, 90)),
        "intra_max": float(intra.max()),
    }


@torch.no_grad()
def end_epoch_diagnostics(model, dataset, subj_map, subjects, device, max_subjects=48):
    model.eval()
    subs = subjects[:max_subjects]

    Z = []
    for subj in subs:
        idx0 = subj_map[subj][0]
        V, mass, L, evals, evecs, faces, gradX, gradY = load_sample(dataset[idx0], device)
        z = model(
            V, mass, L, evals, evecs, faces, gradX, gradY,
            return_per_vertex=False, add_noise=False
        ).squeeze(0)
        Z.append(z)

    Z = torch.stack(Z, dim=0)
    h_raw = batch_latent_health(Z)
    h_norm = batch_latent_health(F.normalize(Z, dim=1))
    return {
        **{f"raw_{k}": v for k, v in h_raw.items()},
        **{f"nrm_{k}": v for k, v in h_norm.items()},
    }


# ============================================================
# MAIN
# ============================================================
def main():
    torch.backends.cuda.matmul.allow_tf32 = True if DEVICE.type == "cuda" else False
    torch.backends.cudnn.allow_tf32 = True if DEVICE.type == "cuda" else False

    print(f"Device: {DEVICE} | AMP: {USE_AMP}")

    dataset = GTReadyDataset(DATA_DIR)
    subj_map = build_subject_map(dataset)
    subjects = np.array(sorted(map(str, subj_map.keys())), dtype=str)

    rng = np.random.default_rng(EVAL_SEED)
    n_eval = int(0.2 * len(subjects))
    eval_subjects = sorted(rng.choice(subjects, size=n_eval, replace=False).tolist())
    train_subjects = sorted([s for s in subjects if s not in set(eval_subjects)])

    assert not (set(train_subjects) & set(eval_subjects)), "Train/eval leakage detected!"
    print(f"Train: {len(train_subjects)} | Eval: {len(eval_subjects)} | Meshes: {len(dataset.files)}")

    D_pack = np.load(DIST_PATH, allow_pickle=True)
    D_orig = D_pack["D_orig"].astype(np.float32)
    D_orig /= np.max(D_orig[D_orig > 0])

    name_to_idx = {}
    for i, n in enumerate([str(x) for x in D_pack["names"]]):
        m = re.search(r"(id\d{4})", n)
        if m:
            name_to_idx[m.group(1)] = i
    assert name_to_idx, "Could not parse subject ids from distance matrix."

    model = DiffusionEncoderOnly(
        latent_dim=LATENT_DIM,
        width=WIDTH,
        n_blocks=N_BLOCKS,
        dropout=DROPOUT,
        n_hks=N_HKS,
        n_wks=N_WKS,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5)

    log_csv = os.path.join(OUT_DIR, "train_log.csv")
    with open(log_csv, "w") as f:
        f.write(
            "epoch,loss,distort,id,var,cov,gnorm,lr,"
            "pearson,spearman,r2,fit_slope,fit_intercept,"
            "intra_mean,intra_median,intra_p90,intra_max,"
            "dist_pairs,dist_scale,dist_dH_mean,dist_dS_mean,dist_rho_mean,"
            "raw_norm_mean,raw_norm_std,raw_std_dim_min,raw_std_dim_mean,raw_cos_mean,raw_cos_std,"
            "nrm_norm_mean,nrm_norm_std,nrm_std_dim_min,nrm_std_dim_mean,nrm_cos_mean,nrm_cos_std\n"
        )

    for epoch in range(EPOCHS):
        model.train()
        rng = np.random.default_rng(epoch + 123)
        subjects_shuf = rng.permutation(train_subjects)

        epoch_loss = 0.0
        epoch_dist = 0.0
        epoch_id = 0.0
        epoch_var = 0.0
        epoch_cov = 0.0
        epoch_gn = 0.0
        n_steps = 0

        # distortion diagnostics (averaged)
        dist_pairs = 0.0
        dist_scale = 0.0
        dist_dH_mean = 0.0
        dist_dS_mean = 0.0
        dist_rho_mean = 0.0

        pbar = tqdm(range(0, len(subjects_shuf), BATCH_SUBJECTS), desc=f"Epoch {epoch+1}/{EPOCHS}")

        for start in pbar:
            batch_subjects = subjects_shuf[start:start + BATCH_SUBJECTS]
            if len(batch_subjects) < 2:
                continue

            optimizer.zero_grad(set_to_none=True)

            # collect per-variant embeddings (IMPORTANT change)
            Z_mesh_list = []
            mesh_gt_idx = []

            # also keep per-subject lists for identity loss
            subj_Zglobals = {}

            for subj in batch_subjects:
                Zg_list = []
                subj_idx = name_to_idx.get(subj, None)

                for idx in subj_map[subj]:
                    V, mass, L, evals, evecs, faces, gradX, gradY = load_sample(dataset[idx], DEVICE)
                    _, Z_global = model(
                        V, mass, L, evals, evecs, faces, gradX, gradY,
                        return_per_vertex=True, add_noise=True
                    )

                    Zg_list.append(Z_global)

                    if subj_idx is not None:
                        Z_mesh_list.append(Z_global.squeeze(0))
                        mesh_gt_idx.append(subj_idx)

                subj_Zglobals[subj] = Zg_list

            # (A) Identity loss: intra-subject variance across topology variants
            loss_id = torch.tensor(0.0, device=DEVICE)
            count_id = 0
            for Zg_list in subj_Zglobals.values():
                if len(Zg_list) < 2:
                    continue
                Zs = torch.cat(Zg_list, dim=0)
                loss_id = loss_id + ((Zs - Zs.mean(dim=0, keepdim=True)) ** 2).mean()
                count_id += 1
            if count_id > 0:
                loss_id = loss_id / count_id

            loss_dist = torch.tensor(0.0, device=DEVICE)
            loss_var = torch.tensor(0.0, device=DEVICE)
            loss_cov = torch.tensor(0.0, device=DEVICE)
            dist_diag = {"n_pairs": 0, "scale": 0.0, "dH_mean": 0.0, "dS_mean": 0.0, "rho_mean": 0.0}

            if len(Z_mesh_list) >= 3:
                Z_mesh = torch.stack(Z_mesh_list, dim=0)  # (M,D)
                idx_np = np.array(mesh_gt_idx, dtype=int)

                D_mesh = torch.tensor(
                    D_orig[np.ix_(idx_np, idx_np)],
                    dtype=Z_mesh.dtype,
                    device=Z_mesh.device
                )

                loss_dist, dist_diag = distortion_ratio_loss_normalized(
                    Z_mesh,
                    D_mesh,
                    min_gt=DIST_MIN_GT,
                    eps_sum_dh=DIST_EPS_SUM_DH,
                    scale_clamp=DIST_SCALE_CLAMP,
                    return_diag=True,
                )

                Zb = F.normalize(Z_mesh, dim=1)
                loss_var = vicreg_var_loss(Zb, gamma=VICREG_GAMMA, eps=VICREG_EPS)
                loss_cov = vicreg_cov_loss(Zb)

            loss = (
                LAMBDA_DISTORT * loss_dist +
                LAMBDA_ID * loss_id +
                LAMBDA_VAR * loss_var +
                LAMBDA_COV * loss_cov
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            gn = grad_norm(model)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_dist += loss_dist.item()
            epoch_id += loss_id.item()
            epoch_var += loss_var.item()
            epoch_cov += loss_cov.item()
            epoch_gn += gn

            dist_pairs += dist_diag["n_pairs"]
            dist_scale += dist_diag["scale"]
            dist_dH_mean += dist_diag["dH_mean"]
            dist_dS_mean += dist_diag["dS_mean"]
            dist_rho_mean += dist_diag["rho_mean"]

            n_steps += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                distort=f"{loss_dist.item():.4f}",
                id=f"{loss_id.item():.4f}",
                var=f"{loss_var.item():.4f}",
                cov=f"{loss_cov.item():.4f}",
                gnorm=f"{gn:.2e}",
                scale=f"{dist_diag['scale']:.1f}",
                pairs=f"{dist_diag['n_pairs']}",
            )

        # end epoch
        denom = max(1, n_steps)
        epoch_loss /= denom
        epoch_dist /= denom
        epoch_id /= denom
        epoch_var /= denom
        epoch_cov /= denom
        epoch_gn /= denom

        dist_pairs /= denom
        dist_scale /= denom
        dist_dH_mean /= denom
        dist_dS_mean /= denom
        dist_rho_mean /= denom

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
                    f"R2={metrics['r2']:.3f} | Slope={metrics['slope']:.3f} | "
                    f"Intra={metrics['intra_mean']:.6f}"
                )

        diag = end_epoch_diagnostics(
            model, dataset, subj_map, eval_subjects, DEVICE, max_subjects=EVAL_DIAG_SUBJECTS
        )
        print(
            f"[epoch health] raw ||Z||={diag['raw_z_norm_mean']:.3f}±{diag['raw_z_norm_std']:.3f} | "
            f"raw std_dim min/mean={diag['raw_z_std_dim_min']:.4f}/{diag['raw_z_std_dim_mean']:.4f} | "
            f"raw cos={diag['raw_cos_sim_mean']:.3f}±{diag['raw_cos_sim_std']:.3f}"
        )
        print(
            f"[epoch health] nrm ||Z||={diag['nrm_z_norm_mean']:.3f}±{diag['nrm_z_norm_std']:.3f} | "
            f"nrm std_dim min/mean={diag['nrm_z_std_dim_min']:.4f}/{diag['nrm_z_std_dim_mean']:.4f} | "
            f"nrm cos={diag['nrm_cos_sim_mean']:.3f}±{diag['nrm_cos_sim_std']:.3f}"
        )

        with open(log_csv, "a") as f:
            if metrics is None:
                metrics = {
                    "pearson": float("nan"),
                    "spearman": float("nan"),
                    "r2": float("nan"),
                    "slope": float("nan"),
                    "intercept": float("nan"),
                    "intra_mean": float("nan"),
                    "intra_median": float("nan"),
                    "intra_p90": float("nan"),
                    "intra_max": float("nan"),
                }

            f.write(
                f"{epoch+1},"
                f"{epoch_loss:.6f},{epoch_dist:.6f},{epoch_id:.6f},{epoch_var:.6f},{epoch_cov:.6f},{epoch_gn:.3e},{lr_now:.2e},"
                f"{metrics['pearson']:.4f},{metrics['spearman']:.4f},{metrics['r2']:.4f},{metrics['slope']:.4f},{metrics['intercept']:.4f},"
                f"{metrics['intra_mean']:.6f},{metrics['intra_median']:.6f},{metrics['intra_p90']:.6f},{metrics['intra_max']:.6f},"
                f"{dist_pairs:.1f},{dist_scale:.6f},{dist_dH_mean:.6f},{dist_dS_mean:.6f},{dist_rho_mean:.6f},"
                f"{diag['raw_z_norm_mean']:.6f},{diag['raw_z_norm_std']:.6f},{diag['raw_z_std_dim_min']:.6f},{diag['raw_z_std_dim_mean']:.6f},{diag['raw_cos_sim_mean']:.6f},{diag['raw_cos_sim_std']:.6f},"
                f"{diag['nrm_z_norm_mean']:.6f},{diag['nrm_z_norm_std']:.6f},{diag['nrm_z_std_dim_min']:.6f},{diag['nrm_z_std_dim_mean']:.6f},{diag['nrm_cos_sim_mean']:.6f},{diag['nrm_cos_sim_std']:.6f}\n"
            )

        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
            ckpt = os.path.join(OUT_DIR, f"encoder_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"Checkpoint: {ckpt}")

    print("Done.")


if __name__ == "__main__":
    main()
