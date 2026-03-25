#!/usr/bin/env python3
import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List

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
from diffusion_autoencoder import DiffusionEncoderXYZSpectrum, DiffusionEncoderOnlyIntrinsec
from latent_loss import stress_loss
from intrinsic_utils import (
    SUBJECT_RE_ANY,
    build_subject_map,
    load_gt_distance_matrix,
    pairwise_rank_loss,
    preflight_gt_alignment_baseline,
    preflight_spectrum_sanity,
    sample_mesh_indices,
    seed_everything,
    slugify_token,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    # NOTE: deprecation warning in recent PyTorch; still works for now.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ============================================================
# RUN DIR (config-based)
# ============================================================

RUNS_ROOT = Path("runs_diffusion_xyz_spectrum")

# ============================================================
# ARGPARSE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_xyz", action="store_true")
    parser.add_argument("--use_spectrum", action="store_true")
    parser.add_argument(
    "--model",
    type=str,
    default="xyz_spectrum",
    choices=["xyz_spectrum", "hkswks"],
    help="Which encoder to train",
    )

    # Args usati solo da xyz_spectrum (li hai già, li lasci)
    # --use_spectrum --k_spec --log_input

    # Args usati solo da hkswks
    parser.add_argument("--n_hks", type=int, default=16)
    parser.add_argument("--n_wks", type=int, default=16)
    parser.add_argument("--eig_k", type=int, default=300)
    parser.add_argument("--pool_mode", type=str, default="meanmax", choices=["mean", "meanmax"])
    
    parser.add_argument("--k_spec", type=int, default=100)
    parser.add_argument("--log_input", action="store_true")
    parser.add_argument("--eps", type=float, default=1e-8)

    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_subjects", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--lambda_stress", type=float, default=0.3)
    parser.add_argument("--lambda_id", type=float, default=0.1)

    parser.add_argument("--max_meshes_per_subject_train", type=int, default=0)
    parser.add_argument("--max_meshes_per_subject_eval", type=int, default=0)

    parser.add_argument("--lambda_rank", type=float, default=0.3)
    parser.add_argument("--rank_margin", type=float, default=0.05)
    parser.add_argument("--rank_pairs", type=int, default=2048)
    parser.add_argument("--rank_tau", type=float, default=0.02)
    parser.add_argument("--rank_hard_frac", type=float, default=0.7)
    
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--eval_every", type=int, default=1)

    # Preflight checks (fail-fast)
    parser.add_argument("--preflight", action="store_true", help="Run preflight checks before training")
    parser.add_argument("--preflight_samples", type=int, default=500, help="How many meshes to sample for preflight")
    parser.add_argument("--preflight_subjects", type=int, default=300, help="How many subjects for GT-alignment baseline")
    parser.add_argument("--preflight_eps_range", type=float, default=1e-3, help="Range threshold to consider a channel 'almost constant' across meshes")
    parser.add_argument("--preflight_dead_frac_warn", type=float, default=0.30, help="Warn if fraction of constant channels exceeds this")
    parser.add_argument("--preflight_dead_frac_stop", type=float, default=0.50, help="Stop if fraction of constant channels exceeds this")
    
    return parser.parse_args()

# ============================================================
# CONFIG PATHS
# ============================================================

DATA_DIR = "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/REMESH/npz_data_topo_500_withops"

DIST_PATH = (
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/"
    "gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/"
    "normalized_matrix_distances.npz"
)

def make_run_dir(args: argparse.Namespace) -> Path:
    """
    Create a run directory based on a stable fingerprint of the args.
    Saves config.json and creates checkpoints/ folder.
    """
    fp = {
        "use_xyz": bool(args.use_xyz),
        "use_spectrum": bool(args.use_spectrum),
        "k_spec": int(args.k_spec),
        "log_input": bool(args.log_input),
        "eps": float(args.eps),

        "latent_dim": int(args.latent_dim),
        "width": int(args.width),
        "n_blocks": int(args.n_blocks),
        "dropout": float(args.dropout),

        "epochs": int(args.epochs),
        "batch_subjects": int(args.batch_subjects),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),

        "lambda_stress": float(args.lambda_stress),
        "lambda_id": float(args.lambda_id),

        "max_meshes_per_subject_train": int(args.max_meshes_per_subject_train),
        "max_meshes_per_subject_eval": int(args.max_meshes_per_subject_eval),

        "seed": int(args.seed),
        "save_every": int(args.save_every),
        "eval_every": int(args.eval_every),
    }

    fp_json = json.dumps(fp, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha1(fp_json.encode("utf-8")).hexdigest()[:8]

    name = (
        f"xyz{int(fp['use_xyz'])}_spec{int(fp['use_spectrum'])}"
        f"_k{fp['k_spec']}_log{int(fp['log_input'])}"
        f"_z{fp['latent_dim']}_w{fp['width']}_b{fp['n_blocks']}"
        f"_bs{fp['batch_subjects']}_lr{fp['lr']:.1e}_wd{fp['weight_decay']:.1e}"
        f"_ls{fp['lambda_stress']:.2g}_li{fp['lambda_id']:.2g}"
        f"_seed{fp['seed']}"
        f"__{h}"
    )

    run_dir = RUNS_ROOT / slugify_token(name)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    # Save config
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "fingerprint": fp}, f, indent=2, sort_keys=True)

    # Optional timestamp file (doesn't affect folder name)
    (run_dir / "run_started.txt").write_text(datetime.now().isoformat(), encoding="utf-8")

    return run_dir

@torch.inference_mode()
def eval_latent_structure(
    model,
    dataset,
    subj_map,
    eval_subjects,
    name_to_idx,
    D_orig,
    max_meshes_per_subject_eval=0,
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
        if not idxs:
            continue

        z_list: List[torch.Tensor] = []
        for idx in idxs:
            sample = dataset[int(idx)]
            Zg = model(
                sample["verts"].to(DEVICE),
                sample["mass"].to(DEVICE),
                sample["L"].to(DEVICE),
                sample["evals"].to(DEVICE),
                sample["evecs"].to(DEVICE),
                sample["faces"].to(DEVICE),
                sample["gradX"].to(DEVICE),
                sample["gradY"].to(DEVICE),
                return_per_vertex=False,
                add_noise=False,
            ).squeeze(0)
            z_list.append(Zg)

        if not z_list:
            continue

        Zs = torch.stack(z_list, dim=0)
        zm = Zs.mean(dim=0)
        subj_mean[subj] = zm
        intra_vals.append(float(((Zs - zm) ** 2).mean().item()))

    kept = [s for s in eval_subjects if s in name_to_idx and s in subj_mean]
    if len(kept) < 3:
        return None

    Zmat = torch.stack([subj_mean[s] for s in kept], dim=0)
    idx = np.array([name_to_idx[s] for s in kept], dtype=int)

    D_gt = torch.tensor(D_orig[np.ix_(idx, idx)], device=DEVICE, dtype=Zmat.dtype)

    iu = torch.triu_indices(D_gt.size(0), D_gt.size(1), offset=1)
    gt = D_gt[iu[0], iu[1]].detach().cpu().numpy()

    # 1) L2
    D_l2 = torch.cdist(Zmat, Zmat, p=2)
    lat_l2 = D_l2[iu[0], iu[1]].detach().cpu().numpy()
    pearson_l2 = np.corrcoef(gt, lat_l2)[0, 1]
    spearman_l2 = np.corrcoef(gt.argsort().argsort(), lat_l2.argsort().argsort())[0, 1]

    # 2) Cosine distance
    Z_norm = Zmat / (Zmat.norm(dim=1, keepdim=True) + 1e-12)
    D_cos = 1.0 - (Z_norm @ Z_norm.T)
    lat_cos = D_cos[iu[0], iu[1]].detach().cpu().numpy()
    pearson_cos = np.corrcoef(gt, lat_cos)[0, 1]
    spearman_cos = np.corrcoef(gt.argsort().argsort(), lat_cos.argsort().argsort())[0, 1]

    # 3) L2 after unit norm
    D_l2_unit = torch.cdist(Z_norm, Z_norm, p=2)
    lat_l2_unit = D_l2_unit[iu[0], iu[1]].detach().cpu().numpy()
    pearson_l2_unit = np.corrcoef(gt, lat_l2_unit)[0, 1]
    spearman_l2_unit = np.corrcoef(gt.argsort().argsort(), lat_l2_unit.argsort().argsort())[0, 1]

    # 4) L2 after z-score per dim
    Z_np = Zmat.detach().cpu().numpy()
    Z_z = (Z_np - Z_np.mean(axis=0, keepdims=True)) / (Z_np.std(axis=0, keepdims=True) + 1e-12)
    Z_z = torch.tensor(Z_z, device=DEVICE, dtype=Zmat.dtype)
    D_l2_z = torch.cdist(Z_z, Z_z, p=2)
    lat_l2_z = D_l2_z[iu[0], iu[1]].detach().cpu().numpy()
    pearson_l2_z = np.corrcoef(gt, lat_l2_z)[0, 1]
    spearman_l2_z = np.corrcoef(gt.argsort().argsort(), lat_l2_z.argsort().argsort())[0, 1]

    return {
        "pearson_l2": float(pearson_l2),
        "spearman_l2": float(spearman_l2),
        "pearson_cos": float(pearson_cos),
        "spearman_cos": float(spearman_cos),
        "pearson_l2_unit": float(pearson_l2_unit),
        "spearman_l2_unit": float(spearman_l2_unit),
        "pearson_l2_z": float(pearson_l2_z),
        "spearman_l2_z": float(spearman_l2_z),
        "intra_mean": float(np.mean(intra_vals)) if intra_vals else float("nan"),
        "n_eval": int(len(kept)),
    }

def main():
    args = parse_args()
    seed_everything(args.seed)

    # ------------------------------------------------------------
    # Basic run setup
    # ------------------------------------------------------------
    run_dir = make_run_dir(args)
    log_csv = run_dir / "train_log.csv"

    print(f"Device={DEVICE}")
    print(f"Run dir: {run_dir}")
    print(f"Model={args.model}")

    # Model-specific sanity (inputs)
    if args.model == "xyz_spectrum":
        if not (args.use_xyz or args.use_spectrum):
            raise ValueError("For --model xyz_spectrum enable --use_xyz and/or --use_spectrum")
        print(
            f"Inputs: XYZ={args.use_xyz} Spectrum={args.use_spectrum} "
            f"k={args.k_spec} log={args.log_input}"
        )
    elif args.model == "hkswks":
        if not (args.use_xyz or args.n_hks > 0 or args.n_wks > 0):
            raise ValueError("For --model hkswks enable --use_xyz and/or set --n_hks/--n_wks > 0")
        print(
            f"Inputs: XYZ={args.use_xyz} HKS={args.n_hks} WKS={args.n_wks} "
            f"eig_k={args.eig_k} pool={args.pool_mode}"
        )
    else:
        raise RuntimeError(f"Unknown model: {args.model}")

    # ------------------------------------------------------------
    # Dataset + GT
    # ------------------------------------------------------------
    dataset = GTReadyDataset(DATA_DIR)
    subj_map = build_subject_map(dataset.files, subject_re=SUBJECT_RE_ANY)
    D_orig, name_to_idx = load_gt_distance_matrix(DIST_PATH, dtype=np.float64)

    subjects = sorted([s for s in subj_map.keys() if s in name_to_idx])
    if len(subjects) < 6:
        raise RuntimeError(f"Need at least 6 subjects overlapping GT matrix, found {len(subjects)}")

    rng_split = np.random.default_rng(args.seed)
    n_eval = max(3, int(0.2 * len(subjects)))
    n_eval = min(n_eval, len(subjects) - 3)
    eval_subjects = sorted(rng_split.choice(subjects, n_eval, replace=False).tolist())
    train_subjects = sorted([s for s in subjects if s not in set(eval_subjects)])

    # ------------------------------------------------------------
    # Preflight (only meaningful for xyz_spectrum for now)
    # ------------------------------------------------------------
    if args.preflight:
        if args.model != "xyz_spectrum":
            print("\n⚠️  Preflight currently supports only --model xyz_spectrum. Skipping preflight.\n")
        else:
            print("\n🧪 Running preflight checks...")

            sanity = preflight_spectrum_sanity(
                dataset=dataset,
                run_dir=run_dir,
                k_spec=args.k_spec,
                log_input=args.log_input,
                eps=args.eps,
                n_meshes=args.preflight_samples,
                seed=args.seed,
                eps_range=args.preflight_eps_range,
                dead_frac_warn=args.preflight_dead_frac_warn,
                dead_frac_stop=args.preflight_dead_frac_stop,
            )
            print(
                "Preflight spectrum sanity:",
                {k: sanity[k] for k in ["status", "almost_constant_fraction", "almost_constant_channels", "range_p99_p1_median"] if k in sanity},
            )

            if sanity.get("status") == "stop":
                print("\n⛔ PREFLIGHT STOP:", sanity.get("reason", "Unknown reason"))
                print(f"See: {sanity.get('json_path','(missing)')}")
                raise SystemExit(2)

            align = preflight_gt_alignment_baseline(
                dataset=dataset,
                subj_map=subj_map,
                subjects=subjects,
                name_to_idx=name_to_idx,
                D_orig=D_orig,
                run_dir=run_dir,
                k_spec=args.k_spec,
                log_input=args.log_input,
                eps=args.eps,
                n_subjects=args.preflight_subjects,
                seed=args.seed + 123,
            )
            print(
                "Preflight GT alignment (spectrum-only):",
                {k: align.get(k) for k in ["status", "spearman_l2", "spearman_cos", "pearson_l2", "pearson_cos", "n_subjects_used"]},
            )

            if align.get("status") == "ok":
                rho = align.get("spearman_l2", float("nan"))
                if isinstance(rho, (float, int)) and not np.isnan(rho) and abs(rho) < 0.05:
                    print("\n⚠️  PREFLIGHT WARNING: spectrum-only baseline has ~0 correlation with D_orig.")
                    print("   This often means your target is extrinsic or spectrum scaling carries little signal.\n")

    # ------------------------------------------------------------
    # Build model (switch by CLI)
    # ------------------------------------------------------------
    if args.model == "xyz_spectrum":
        model = DiffusionEncoderXYZSpectrum(
            latent_dim=args.latent_dim,
            width=args.width,
            n_blocks=args.n_blocks,
            dropout=args.dropout,
            use_xyz=args.use_xyz,
            use_spectrum=args.use_spectrum,
            k_spec=args.k_spec,
            log_input=args.log_input,
            eps=args.eps,
        ).to(DEVICE)

    elif args.model == "hkswks":
        model = DiffusionEncoderOnlyIntrinsec(
            latent_dim=args.latent_dim,
            width=args.width,
            n_blocks=args.n_blocks,
            dropout=args.dropout,
            use_xyz=args.use_xyz,
            n_hks=args.n_hks,
            n_wks=args.n_wks,
            eig_k=args.eig_k,
            eps=args.eps,
            pool_mode=args.pool_mode,
        ).to(DEVICE)

    else:
        raise RuntimeError(f"Unknown model: {args.model}")

    # ------------------------------------------------------------
    # Optim
    # ------------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3)

    # ------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------
    with open(log_csv, "w", encoding="utf-8") as f:
        f.write(
            "epoch,loss,stress,rank,lr,"
            "spearman_l2,spearman_cos,spearman_l2_unit,spearman_l2_z,"
            "pearson_l2,pearson_cos,pearson_l2_unit,pearson_l2_z,"
            "intra,n_eval\n"
        )

    last_metrics = {
        "spearman_l2": float("nan"),
        "spearman_cos": float("nan"),
        "spearman_l2_unit": float("nan"),
        "spearman_l2_z": float("nan"),
        "pearson_l2": float("nan"),
        "pearson_cos": float("nan"),
        "pearson_l2_unit": float("nan"),
        "pearson_l2_z": float("nan"),
        "intra_mean": float("nan"),
        "n_eval": 0,
    }

    best_spearman = -1e9
    best_epoch = -1
    best_ckpt_path = run_dir / "checkpoints" / "best_by_spearman.pth"

    # ------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = np.random.default_rng(args.seed + 999 + epoch)
        subjects_shuf = rng.permutation(train_subjects)

        epoch_loss = 0.0
        epoch_stress = 0.0
        epoch_rank = 0.0
        n_steps = 0

        pbar = tqdm(
            range(0, len(subjects_shuf), args.batch_subjects),
            desc=f"Epoch {epoch}/{args.epochs}",
            dynamic_ncols=True,
        )

        for start in pbar:
            batch_subjects = subjects_shuf[start : start + args.batch_subjects]
            if len(batch_subjects) < 2:
                continue

            optimizer.zero_grad(set_to_none=True)

            subj_means: List[torch.Tensor] = []
            subj_gt_idx: List[int] = []

            for subj in batch_subjects:
                idxs = sample_mesh_indices(
                    subj_map[subj],
                    max_meshes=args.max_meshes_per_subject_train,
                    rng=rng,
                )
                if not idxs:
                    continue

                z_list: List[torch.Tensor] = []
                for idx in idxs:
                    sample = dataset[int(idx)]
                    Zg = model(
                        sample["verts"].to(DEVICE),
                        sample["mass"].to(DEVICE),
                        sample["L"].to(DEVICE),
                        sample["evals"].to(DEVICE),
                        sample["evecs"].to(DEVICE),
                        sample["faces"].to(DEVICE),
                        sample["gradX"].to(DEVICE),
                        sample["gradY"].to(DEVICE),
                        return_per_vertex=False,
                        add_noise=True,
                    ).squeeze(0)
                    z_list.append(Zg)

                if not z_list:
                    continue

                Zs = torch.stack(z_list, dim=0)
                subj_means.append(Zs.mean(dim=0))
                subj_gt_idx.append(name_to_idx[subj])

            if len(subj_means) < 2:
                continue

            Z_batch = torch.stack(subj_means, dim=0)
            idx_np = np.array(subj_gt_idx, dtype=int)

            D_batch = torch.tensor(
                D_orig[np.ix_(idx_np, idx_np)],
                device=DEVICE,
                dtype=Z_batch.dtype,
            )

            loss_stress = stress_loss(Z_batch, D_batch)

            D_lat = torch.cdist(Z_batch, Z_batch, p=2)
            loss_rank = pairwise_rank_loss(
                D_lat,
                D_batch,
                n_pairs=args.rank_pairs,
                margin=args.rank_margin,
                tau=args.rank_tau,
                hard_frac=args.rank_hard_frac,
            )

            loss = args.lambda_stress * loss_stress + args.lambda_rank * loss_rank

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            loss_item = float(loss.item())
            stress_item = float(loss_stress.item())
            rank_item = float(loss_rank.item())

            epoch_loss += loss_item
            epoch_stress += stress_item
            epoch_rank += rank_item
            n_steps += 1

            pbar.set_postfix(loss=f"{loss_item:.4f}", stress=f"{stress_item:.4f}", rank=f"{rank_item:.4f}")

        if n_steps == 0:
            raise RuntimeError("No valid optimization step in epoch. Check split/settings.")

        epoch_loss /= n_steps
        epoch_stress /= n_steps
        epoch_rank /= n_steps

        scheduler.step(epoch_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        do_eval = (args.eval_every > 0) and (epoch % args.eval_every == 0 or epoch == args.epochs)
        if do_eval:
            metrics = eval_latent_structure(
                model=model,
                dataset=dataset,
                subj_map=subj_map,
                eval_subjects=eval_subjects,
                name_to_idx=name_to_idx,
                D_orig=D_orig,
                max_meshes_per_subject_eval=args.max_meshes_per_subject_eval,
            )
            if metrics is not None:
                last_metrics = metrics

        metrics = last_metrics

        print(f"\nEpoch {epoch}")
        print(
            {
                "loss": epoch_loss,
                "stress": epoch_stress,
                "rank": epoch_rank,
                "lr": lr_now,
                "spearman_l2": metrics["spearman_l2"],
                "spearman_cos": metrics["spearman_cos"],
                "spearman_l2_unit": metrics["spearman_l2_unit"],
                "spearman_l2_z": metrics["spearman_l2_z"],
                "n_eval": metrics["n_eval"],
            }
        )

        # Save best checkpoint by spearman_l2_z
        if do_eval:
            score = float(metrics.get("spearman_l2_z", float("nan")))
            if not np.isnan(score) and score > best_spearman:
                best_spearman = score
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": vars(args),
                        "best_spearman_l2_z": best_spearman,
                    },
                    best_ckpt_path,
                )
                (run_dir / "best_by_spearman.txt").write_text(
                    f"best_epoch={best_epoch}\nbest_spearman_l2_z={best_spearman}\n",
                    encoding="utf-8",
                )
                print(f"🏁 New best spearman_l2_z={best_spearman:.4f} @ epoch {best_epoch} -> {best_ckpt_path}")

        with open(log_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{epoch_loss:.6f},{epoch_stress:.6f},{epoch_rank:.6f},{lr_now:.2e},"
                f"{metrics['spearman_l2']:.6f},{metrics['spearman_cos']:.6f},"
                f"{metrics['spearman_l2_unit']:.6f},{metrics['spearman_l2_z']:.6f},"
                f"{metrics['pearson_l2']:.6f},{metrics['pearson_cos']:.6f},"
                f"{metrics['pearson_l2_unit']:.6f},{metrics['pearson_l2_z']:.6f},"
                f"{metrics['intra_mean']:.6e},{int(metrics['n_eval'])}\n"
            )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = run_dir / "checkpoints" / f"epoch{epoch:03d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                },
                ckpt,
            )

    print("✅ DONE.")
    print(f"Saved in: {run_dir}")
    print(f"Best spearman_l2_z={best_spearman:.4f} at epoch {best_epoch} -> {best_ckpt_path}")


if __name__ == "__main__":
    main()
