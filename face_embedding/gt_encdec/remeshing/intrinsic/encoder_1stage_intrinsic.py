#!/usr/bin/env python3
import os
import re
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import sys
from typing import Dict, List, Sequence, Tuple

sys.path.append(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/gt_encdec/autoencoder"
)

from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset
from diffusion_autoencoder import DiffusionEncoderOnlyIntrinsec, DiffusionEncoderXYZSpectrum
from latent_loss import stress_loss
from intrinsic_utils import SUBJECT_RE_ANY, build_subject_map, sample_mesh_indices

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ============================================================
# ARGPARSE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_xyz", action="store_true")
    parser.add_argument("--n_hks", type=int, default=0)
    parser.add_argument("--n_wks", type=int, default=0)

    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_subjects", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--max_meshes_per_subject_train", type=int, default=0,
                        help="0 = use all variants per subject in training")
    parser.add_argument("--max_meshes_per_subject_eval", type=int, default=0,
                        help="0 = use all variants per subject in evaluation")

    parser.add_argument("--cache_samples", dest="cache_samples", action="store_true")
    parser.add_argument("--no_cache_samples", dest="cache_samples", action="store_false")
    parser.set_defaults(cache_samples=True)

    parser.add_argument("--cache_device_tensors", action="store_true",
                        help="Cache tensors directly on DEVICE (faster, more VRAM).")
    parser.add_argument("--preload_cache", action="store_true",
                        help="Preload all samples into CPU cache before epoch 1.")

    return parser.parse_args()


# ============================================================
# CONFIG PATHS
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

OUT_DIR = "encoder_stage1_intrinsic_controlled"
os.makedirs(OUT_DIR, exist_ok=True)

class SampleAccessor:
    """
    Two-level cache:
    - CPU cache: avoids re-loading/parsing NPZ every epoch
    - DEVICE cache (optional): avoids repeated CPU->GPU transfers
    """

    def __init__(self, dataset, cache_samples: bool = True, cache_device_tensors: bool = False):
        self.dataset = dataset
        self.cache_samples = cache_samples
        self.cache_device_tensors = cache_device_tensors
        self._cpu_cache: Dict[int, dict] = {}
        self._device_cache: Dict[int, Tuple[torch.Tensor, ...]] = {}

    def get_cpu(self, idx: int) -> dict:
        if not self.cache_samples:
            return self.dataset[idx]
        out = self._cpu_cache.get(idx)
        if out is None:
            out = self.dataset[idx]
            self._cpu_cache[idx] = out
        return out

    def get_tensors(self, idx: int) -> Tuple[torch.Tensor, ...]:
        if self.cache_device_tensors:
            out = self._device_cache.get(idx)
            if out is not None:
                return out

        sample = self.get_cpu(idx)
        out = (
            sample["verts"].to(DEVICE),
            sample["mass"].to(DEVICE),
            sample["L"].to(DEVICE),
            sample["evals"].to(DEVICE),
            sample["evecs"].to(DEVICE),
            sample["faces"].to(DEVICE),
            sample["gradX"].to(DEVICE),
            sample["gradY"].to(DEVICE),
        )

        if self.cache_device_tensors:
            self._device_cache[idx] = out
        return out

    def preload_cpu(self, indices: Sequence[int]) -> None:
        if not self.cache_samples:
            return
        for idx in tqdm(indices, desc="Preload CPU cache", dynamic_ncols=True):
            _ = self.get_cpu(int(idx))

    @property
    def cpu_cache_size(self) -> int:
        return len(self._cpu_cache)

    @property
    def device_cache_size(self) -> int:
        return len(self._device_cache)


# ============================================================
# EVAL
# ============================================================

@torch.inference_mode()
def eval_latent_structure(
    model,
    sample_accessor: SampleAccessor,
    subj_map,
    eval_subjects,
    name_to_idx,
    D_orig,
    max_meshes_per_subject_eval: int = 0,
):
    model.eval()

    subj_mean: Dict[str, torch.Tensor] = {}
    intra_vals: List[float] = []

    for subj in eval_subjects:
        idxs = sample_mesh_indices(
            subj_map[subj],
            max_meshes=max_meshes_per_subject_eval,
            rng=None,
        )
        if len(idxs) == 0:
            continue

        z_list: List[torch.Tensor] = []

        for idx in idxs:
            V, mass, L, evals, evecs, faces, gradX, gradY = sample_accessor.get_tensors(int(idx))

            Zg = model(
                V, mass, L, evals, evecs,
                faces, gradX, gradY,
                return_per_vertex=False,
                add_noise=False
            ).squeeze(0)

            z_list.append(Zg)

        if len(z_list) == 0:
            continue

        Zs = torch.stack(z_list, dim=0)
        zm = Zs.mean(dim=0)

        subj_mean[subj] = zm
        intra_vals.append(((Zs - zm) ** 2).mean().item())

    kept = [s for s in eval_subjects if s in name_to_idx and s in subj_mean]
    if len(kept) < 3:
        return None

    Zmat = torch.stack([subj_mean[s] for s in kept], dim=0)
    idx = np.array([name_to_idx[s] for s in kept], dtype=int)

    D_gt = torch.tensor(
        D_orig[np.ix_(idx, idx)],
        device=DEVICE,
        dtype=Zmat.dtype
    )

    # Upper triangular indices
    iu = torch.triu_indices(D_gt.size(0), D_gt.size(1), offset=1)
    gt = D_gt[iu[0], iu[1]].cpu().numpy()

    Z_np = Zmat.cpu().numpy()

    # -----------------------
    # 1) L2 standard
    # -----------------------
    D_l2 = torch.cdist(Zmat, Zmat)
    lat_l2 = D_l2[iu[0], iu[1]].cpu().numpy()

    pearson_l2 = np.corrcoef(gt, lat_l2)[0, 1]
    spearman_l2 = np.corrcoef(gt.argsort().argsort(),
                            lat_l2.argsort().argsort())[0, 1]

    # -----------------------
    # 2) Cosine distance
    # -----------------------
    Z_norm = Zmat / (Zmat.norm(dim=1, keepdim=True) + 1e-12)
    D_cos = 1.0 - (Z_norm @ Z_norm.T)
    lat_cos = D_cos[iu[0], iu[1]].cpu().numpy()

    pearson_cos = np.corrcoef(gt, lat_cos)[0, 1]
    spearman_cos = np.corrcoef(gt.argsort().argsort(),
                            lat_cos.argsort().argsort())[0, 1]

    # -----------------------
    # 3) L2 after unit norm
    # -----------------------
    D_l2_unit = torch.cdist(Z_norm, Z_norm)
    lat_l2_unit = D_l2_unit[iu[0], iu[1]].cpu().numpy()

    pearson_l2_unit = np.corrcoef(gt, lat_l2_unit)[0, 1]
    spearman_l2_unit = np.corrcoef(gt.argsort().argsort(),
                                lat_l2_unit.argsort().argsort())[0, 1]

    # -----------------------
    # 4) L2 after z-score per dimension
    # -----------------------
    Z_z = (Z_np - Z_np.mean(axis=0)) / (Z_np.std(axis=0) + 1e-12)
    Z_z = torch.tensor(Z_z, device=Zmat.device, dtype=Zmat.dtype)

    D_l2_z = torch.cdist(Z_z, Z_z)
    lat_l2_z = D_l2_z[iu[0], iu[1]].cpu().numpy()

    pearson_l2_z = np.corrcoef(gt, lat_l2_z)[0, 1]
    spearman_l2_z = np.corrcoef(gt.argsort().argsort(),
                                lat_l2_z.argsort().argsort())[0, 1]

    return {
        "pearson_l2": float(pearson_l2),
        "spearman_l2": float(spearman_l2),
        "pearson_cos": float(pearson_cos),
        "spearman_cos": float(spearman_cos),
        "pearson_l2_unit": float(pearson_l2_unit),
        "spearman_l2_unit": float(spearman_l2_unit),
        "pearson_l2_z": float(pearson_l2_z),
        "spearman_l2_z": float(spearman_l2_z),
        "intra_mean": float(np.mean(intra_vals)),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    print(f"🚀 Stage-1 Intrinsic Controlled | device={DEVICE}")
    print(f"Input → XYZ={args.use_xyz} | HKS={args.n_hks} | WKS={args.n_wks}")
    print(
        f"Speed opts → cache_samples={args.cache_samples} "
        f"cache_device_tensors={args.cache_device_tensors} "
        f"eval_every={args.eval_every} "
        f"train_mesh_cap={args.max_meshes_per_subject_train or 'all'} "
        f"eval_mesh_cap={args.max_meshes_per_subject_eval or 'all'}"
    )

    dataset = GTReadyDataset(DATA_DIR)
    sample_accessor = SampleAccessor(
        dataset=dataset,
        cache_samples=args.cache_samples,
        cache_device_tensors=args.cache_device_tensors,
    )

    subj_map = build_subject_map(dataset.files, subject_re=SUBJECT_RE_ANY)
    subjects = sorted(subj_map.keys())

    rng = np.random.default_rng(12345)
    subjects = np.array(subjects)
    n_eval = int(0.2 * len(subjects))

    eval_subjects = sorted(rng.choice(subjects, n_eval, replace=False))
    train_subjects = sorted([s for s in subjects if s not in eval_subjects])

    D_pack = np.load(DIST_PATH, allow_pickle=True)
    D_orig = D_pack["D_orig"].astype(np.float64)
    D_orig /= np.max(D_orig[D_orig > 0])

    names = [str(n) for n in D_pack["names"]]
    name_to_idx = {}
    for i, n in enumerate(names):
        m = re.search(r"(id\d{4})", n)
        if m:
            name_to_idx[m.group(1)] = i

    model = DiffusionEncoderOnlyIntrinsec(
        latent_dim=args.latent_dim,
        width=args.width,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        use_xyz=args.use_xyz,
        n_hks=args.n_hks,
        n_wks=args.n_wks,
    ).to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3)

    log_csv = os.path.join(OUT_DIR, "train_log.csv")

    with open(log_csv, "w") as f:
        f.write("epoch,loss,stress,id,pearson,spearman,intra\n")

    if args.preload_cache:
        all_indices = sorted({idx for idxs in subj_map.values() for idx in idxs})
        sample_accessor.preload_cpu(all_indices)
        print(f"CPU cache preloaded: {sample_accessor.cpu_cache_size} samples")

    last_metrics = {
        "spearman_l2": float("nan"),
        "spearman_cos": float("nan"),
        "spearman_l2_unit": float("nan"),
        "spearman_l2_z": float("nan"),
        "intra_mean": float("nan"),
    }

    for epoch in range(args.epochs):

        model.train()
        rng = np.random.default_rng(epoch + 999)
        subjects_shuf = rng.permutation(train_subjects)

        epoch_loss = 0.0
        epoch_stress = 0.0
        epoch_id = 0.0
        n_steps = 0

        pbar = tqdm(range(0, len(subjects_shuf), args.batch_subjects),
                    desc=f"Epoch {epoch+1}/{args.epochs}",
                    dynamic_ncols=True)

        for start in pbar:
            batch_subjects = subjects_shuf[start:start + args.batch_subjects]
            if len(batch_subjects) < 2:
                continue

            optimizer.zero_grad(set_to_none=True)

            # Store per-subject stats once, reuse for both identity and stress losses.
            subj_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

            for subj in batch_subjects:
                idxs = sample_mesh_indices(
                    subj_map[subj],
                    max_meshes=args.max_meshes_per_subject_train,
                    rng=rng,
                )
                if len(idxs) == 0:
                    continue
                z_list: List[torch.Tensor] = []

                for idx in idxs:
                    V, mass, L, evals, evecs, faces, gradX, gradY = sample_accessor.get_tensors(int(idx))

                    Zg = model(
                        V, mass, L, evals, evecs,
                        faces, gradX, gradY,
                        return_per_vertex=False,
                        add_noise=True
                    ).squeeze(0)
                    z_list.append(Zg)

                if len(z_list) == 0:
                    continue
                Zs = torch.stack(z_list, dim=0)
                Zm = Zs.mean(dim=0)
                subj_stats[subj] = (Zs, Zm)

            if len(subj_stats) < 2:
                continue

            # Identity loss
            loss_id = torch.tensor(0.0, device=DEVICE)
            count_id = 0

            for Zs, Zm in subj_stats.values():
                if Zs.shape[0] < 2:
                    continue
                loss_id += ((Zs - Zm.unsqueeze(0)) ** 2).mean()
                count_id += 1

            if count_id > 0:
                loss_id /= count_id

            # Stress loss
            subj_means: List[torch.Tensor] = []
            subj_gt_idx: List[int] = []

            for subj in batch_subjects:
                packed = subj_stats.get(subj)
                if packed is None:
                    continue
                gt_i = name_to_idx.get(subj)
                if gt_i is None:
                    continue
                _, Zm = packed
                subj_means.append(Zm)
                subj_gt_idx.append(gt_i)

            if len(subj_means) >= 2:
                Z_batch = torch.stack(subj_means, dim=0)
                idx_np = np.array(subj_gt_idx, dtype=int)

                D_batch = torch.tensor(
                    D_orig[np.ix_(idx_np, idx_np)],
                    device=DEVICE,
                    dtype=Z_batch.dtype
                )

                loss_stress = stress_loss(Z_batch, D_batch)
            else:
                loss_stress = torch.tensor(0.0, device=DEVICE)

            loss = 0.3 * loss_stress + 0.1 * loss_id

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_item = float(loss.item())
            loss_stress_item = float(loss_stress.item())
            loss_id_item = float(loss_id.item())
            epoch_loss += loss_item
            epoch_stress += loss_stress_item
            epoch_id += loss_id_item
            n_steps += 1

            pbar.set_postfix(
                loss=f"{loss_item:.4f}",
                stress=f"{loss_stress_item:.4f}",
                ident=f"{loss_id_item:.4f}",
            )

        epoch_loss /= max(1, n_steps)
        epoch_stress /= max(1, n_steps)
        epoch_id /= max(1, n_steps)
        scheduler.step(epoch_loss)

        do_eval = args.eval_every > 0 and (
            (epoch + 1) % args.eval_every == 0 or (epoch + 1) == args.epochs
        )
        if do_eval:
            metrics = eval_latent_structure(
                model=model,
                sample_accessor=sample_accessor,
                subj_map=subj_map,
                eval_subjects=eval_subjects,
                name_to_idx=name_to_idx,
                D_orig=D_orig,
                max_meshes_per_subject_eval=args.max_meshes_per_subject_eval,
            )
            if metrics is not None:
                last_metrics = metrics
        metrics = last_metrics

        print(f"\nEpoch {epoch+1}")
        print(
            {
                "loss": epoch_loss,
                "stress": epoch_stress,
                "id": epoch_id,
                "spearman_l2": metrics["spearman_l2"],
                "spearman_cos": metrics["spearman_cos"],
                "spearman_l2_unit": metrics["spearman_l2_unit"],
                "spearman_l2_z": metrics["spearman_l2_z"],
                "cpu_cache": sample_accessor.cpu_cache_size,
                "device_cache": sample_accessor.device_cache_size,
            }
        )

        with open(log_csv, "a") as f:
            f.write(
                "epoch,loss,stress,id,"
                "spearman_l2,spearman_cos,"
                "spearman_l2_unit,spearman_l2_z,"
                "intra\n"
            )

    print("✅ DONE.")


if __name__ == "__main__":
    main()
