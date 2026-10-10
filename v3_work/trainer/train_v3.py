#!/usr/bin/env python
"""Trainer v3: la catena v2 appiattita in un solo programma, con i flag del piano massivo.

Con i flag v3 al default (sotto) e la ricetta v1 del run su scala, riproduce v2: stessi batch, stesse mesh,
stesso rumore, stessa loss e stessi gradienti (tests/test_equivalence.py, eseguito e riportato in
aau/runs/evidence/trainer_v3/summary.md). I default della ricetta sono quelli del run su scala
(aau/data_scale/recipe_v1.sh + train_scale.sbatch), e i nomi dei flag sono quelli di v1/train_steps, cosi'
la riga di lancio di v2 funziona tale e quale.

Flag v3 (default = v2):
  --forward sequential|packed      una forward per mesh (v2) o tutte le mesh del passo in una (model_v3.py)
  --bucket-ratio R                 packed: max/min dei vertici in un bucket della parte spettrale (1.5)
  --loss v2|log|log+inv|log+inv+nbr  losses_v3.py; pesi --lambda-scale --lambda-inv --lambda-nbr ...
  --sampler v2|balanced            sampler_v3.py; --domain-alpha a: p_d ~ n_d^a (0 = uniforme fra domini)
  --batch-domains single|mixed     batch a dominio singolo (v2) o misti (serve una GT definita fra domini)
  --pooling meanmax|area_meanmax|area_attn   (--attn-heads)
  --input-norm maxabs|sqrt_area|global  frame dei vertici prima del rumore (data_v3.reframe_sqrt_area; global:
                                   mm nel frame canonico / --global-unit-mm, global_v3.py, con --scale-table)
  --head embed|factorized          z = (s, u): log centroid size + forma (factorized_v3.py, --size-table,
                                   --lambda-size, --scale-aug lo,hi); --forward sequential o groups
  --width / --n_blocks             taglia (flag v1)
  --ema-decay D                    EMA dei pesi (0 = spenta); i checkpoint *_ema.pth hanno i pesi EMA
  --compact-cache                  cache senza perdita: int32 per facce e indici, niente L (-~28% RAM)
  --lr-constant                    lr costante senza scheduler (alternativo a --lr-steps / --plateau-patience)
  --resume auto|none|<file>        ripresa da checkpoints/last.pth (scritto a ogni epoca)
  --max-hours H                    si ferma pulito (last.pth e checkpoint finale) oltre H ore
DDP: lanciato con torchrun (``--nproc-per-node N``, elastico con ``--max-restarts``), ogni rank tiene in RAM
solo il suo shard di soggetti (stratificato per dominio), fa ``--steps-per-epoch`` batch per epoca dal suo
shard; DistributedDataParallel media i gradienti. Un passo = un batch per rank.

    aau/run.sh v3_work/trainer/train_v3.py --total-steps ... --data-spec ... (righe come train_steps.py)
    torchrun --standalone --nproc-per-node 2 --max-restarts 2 v3_work/trainer/train_v3.py ...
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import common  # noqa: E402,F401
from common import dist_info, domain_of, log0, split_name  # noqa: E402
import data_v3 as dv  # noqa: E402
import factorized_v3 as fz  # noqa: E402
import sampler_v3 as sv  # noqa: E402
from losses_v3 import LOSSES, StepBatch, compute_loss  # noqa: E402
from model_v3 import POOLINGS, StepEmbedder, build_model_v3  # noqa: E402

import robustness.train_runner as tr  # noqa: E402  (v1 congelato, sola lettura)
from intrinsic_utils import SUBJECT_RE_ANY, build_subject_map, seed_everything, slugify_token  # noqa: E402
from robustness.data_utils import (  # noqa: E402
    build_eval_plan, build_sample_eval_records, infer_topology_label_from_name, preload_eval_samples,
    rebuild_subject_split)
from robustness.eval_utils import (  # noqa: E402
    SubjectEvalContext, build_pair_eval_context, build_sigma_grid, evaluate_subject_robustness_grid,
    summarize_eval_plan)
from robustness.noise import PerturbationParams, parse_noise_mode_weights, parse_noise_modes  # noqa: E402

STATE = {"steps": 0, "t_start": time.time(), "stop": False}


# --- argomenti -------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    B = argparse.BooleanOptionalAction
    # v1 (default = ricetta del run su scala, aau/data_scale/recipe_v1.sh)
    p.add_argument("--data_dir", default="")
    p.add_argument("--dist_npz", default="", help="file GT (D_orig + names); vuoto = quello di --gt")
    p.add_argument("--gt", default="maxabs", choices=["maxabs", "unified"],
                   help="target: maxabs = GT del run su scala (datasets/SCALE_ALL, NaN fra domini); unified = GT "
                        "unificata (datasets/UNIFIED_GT/train, invariante alla similarita', definita fra domini)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--runs_root", required=True)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_subjects", type=int, default=5)
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--n_blocks", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--model", default="xyz_dn", choices=["xyz_dn"])
    p.add_argument("--k_spec", type=int, default=0)
    p.add_argument("--log_spec", action=B, default=False)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--use_xyz", action=B, default=True)
    p.add_argument("--use_spectrum", action=B, default=True)
    p.add_argument("--n_hks", type=int, default=0)
    p.add_argument("--n_wks", type=int, default=0)
    p.add_argument("--eig_k", type=int, default=300)
    p.add_argument("--pool_mode", default="meanmax", choices=["mean", "meanmax"])
    p.add_argument("--xyz_feature_dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--max_subjects_eval_train", type=int, default=16)
    p.add_argument("--preload_eval_samples_train", action=B, default=True)
    p.add_argument("--preload_eval_workers_train", type=int, default=2)
    p.add_argument("--max_meshes_per_subject_train", type=int, default=6)
    p.add_argument("--max_meshes_per_subject_eval", type=int, default=6)
    p.add_argument("--train_level", default="mixed", choices=["mixed"])
    p.add_argument("--train_pair_mode", default="cross_topology", choices=["all", "within_topology", "cross_topology"])
    p.add_argument("--lambda_subject", type=float, default=1.0)
    p.add_argument("--lambda_mesh", type=float, default=1.0)
    p.add_argument("--use_id_loss", action=B, default=True)
    p.add_argument("--lambda_id", type=float, default=0.25)
    p.add_argument("--lambda_rank", type=float, default=0.5)
    p.add_argument("--rank_margin", type=float, default=0.05)
    p.add_argument("--rank_pairs", type=int, default=1024)
    p.add_argument("--rank_tau", type=float, default=0.02)
    p.add_argument("--rank_hard_frac", type=float, default=0.7)
    p.add_argument("--train_latent_noise", action=B, default=True)
    p.add_argument("--p_noise", type=float, default=0.6)
    p.add_argument("--sigma_min", type=float, default=5e-4)
    p.add_argument("--sigma_max", type=float, default=2e-2)
    p.add_argument("--noise_modes", default="translation,rotation,jitter")
    p.add_argument("--noise_mode_weights", default="translation=4,rotation=2,jitter=1")
    p.add_argument("--outlier_frac", type=float, default=0.02)
    p.add_argument("--outlier_scale", type=float, default=6.0)
    p.add_argument("--rigid_rot_deg", type=float, default=12.0)
    p.add_argument("--rigid_trans_scale", type=float, default=0.03)
    p.add_argument("--rigid_rot_deg_min", type=float, default=0.5)
    p.add_argument("--rigid_trans_scale_min", type=float, default=0.001)
    p.add_argument("--sigma_min_eval", type=float, default=1e-3)
    p.add_argument("--sigma_max_eval", type=float, default=0.1)
    p.add_argument("--n_sigma_eval", type=int, default=6)
    p.add_argument("--eval_mode", default="average", choices=["fixed", "random", "average"])
    p.add_argument("--init_checkpoint", default="")
    # train_steps / train_v2 / train_fast
    p.add_argument("--total-steps", type=int, default=0)
    p.add_argument("--steps-per-epoch", type=int, default=0)
    p.add_argument("--split-json", default="")
    p.add_argument("--frozen-heldout", default="")
    p.add_argument("--data-spec", default="")
    p.add_argument("--stage-root", default="")
    p.add_argument("--store", default="", help="store in mmap degli operatori (tools/build_store.py): niente "
                                               "staging ne' pre-pass; la data-spec resta per gruppi, blocchi e quote")
    p.add_argument("--prepass-proc", type=int, default=16)
    p.add_argument("--prepass-tolerate", type=int, default=0,
                   help="mesh fallite nel pre-pass tollerate per blocco (scartate con avviso); 0 = errore, come v2")
    p.add_argument("--prepass-grad", default="vec", choices=["orig", "vec"],
                   help="vec: build_grad vettorizzato di E9 nel pre-pass (gradX/gradY identici dopo il cast a fp32)")
    p.add_argument("--domain-blocked", action=B, default=True)
    p.add_argument("--eval_domain", default="")
    p.add_argument("--gt-keep-scale", action="store_true")
    p.add_argument("--gt-scale", type=float, default=1.0,
                   help="GT moltiplicata per k al caricamento (es. kappa della GT shape: i margini v2 sono in unita' GT)")
    p.add_argument("--plateau-patience", type=int, default=None)
    p.add_argument("--lr-steps", default="")
    p.add_argument("--lr-constant", action="store_true",
                   help="lr costante (--lr) per tutto il run, nessuno scheduler (la ricetta di C3F/C3M, dove il "
                        "confine di --lr-steps cade oltre la fine); alternativo a --lr-steps e --plateau-patience. "
                        "Senza nessuno dei tre: ReduceLROnPlateau(0.5, patience 8), il default di v2")
    p.add_argument("--pin-cache", action="store_true")
    p.add_argument("--train-threads", type=int, default=0)
    p.add_argument("--cache-residency", default="ram", choices=["ram"])
    p.add_argument("--cache-workers", type=int, default=16)
    p.add_argument("--cache-max-gb", type=float, default=900.0)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--frame", default="current", choices=["current"], help="compatibilita' v2: usare --input-norm")
    # v3
    p.add_argument("--forward", default="sequential", choices=["sequential", "groups", "packed"])
    p.add_argument("--pad-slack", type=float, default=0.05, help="groups: max/min dei vertici in un gruppo - 1")
    p.add_argument("--bucket-ratio", type=float, default=1.5)
    p.add_argument("--loss", default="v2", choices=list(LOSSES))
    p.add_argument("--lambda-scale", type=float, default=1.0)
    p.add_argument("--scale-target", type=float, default=1.0)
    p.add_argument("--lambda-inv", type=float, default=1.0)
    p.add_argument("--lambda-nbr", type=float, default=0.5)
    p.add_argument("--log-huber-delta", type=float, default=0.25)
    p.add_argument("--log-eps", type=float, default=1e-6)
    p.add_argument("--log-eps-gt", type=float, default=1e-4)
    p.add_argument("--log-pair-mode", default="all", choices=["all", "cross_topology"])
    p.add_argument("--w-cross", type=float, default=1.0)
    p.add_argument("--sampler", default="v2", choices=["v2", "balanced"])
    p.add_argument("--domain-alpha", type=float, default=0.0)
    p.add_argument("--batch-domains", default="single", choices=["single", "mixed"])
    p.add_argument("--pooling", default="meanmax", choices=list(POOLINGS))
    p.add_argument("--attn-heads", type=int, default=1)
    p.add_argument("--input-norm", default="maxabs", choices=["maxabs", "sqrt_area", "global"])
    p.add_argument("--scale-table", default="", help="global: tabelle di scala per mesh (tools/build_scale_table.py), "
                                                     "separate da virgola")
    p.add_argument("--global-unit-mm", type=float, default=100.0, help="global: L0, la costante (mm) uguale per tutte")
    p.add_argument("--global-ops", default="areanorm", choices=["areanorm", "mm"],
                   help="global: operatori come serviti (area 1) o massa/autovettori nelle unita' di xyz (equivalenti)")
    p.add_argument("--head", default="embed", choices=["embed", "factorized", "factorized2", "dual"])
    p.add_argument("--dist-npz-shape", default="", help="dual: GT di u (GT-SR); --dist_npz e' quella di z_F (GT-FR)")
    p.add_argument("--gt-scale-shape", type=float, default=1.0, help="dual: come --gt-scale, per la GT di u")
    p.add_argument("--lambda-form", type=float, default=1.0, help="dual: peso della loss su z_F")
    p.add_argument("--lambda-shape", type=float, default=1.0, help="dual: peso della loss su u")
    p.add_argument("--size-table", default="", help="factorized: log centroid size per identita' "
                                                    "(tools/build_factorized_targets.py, size.npz)")
    p.add_argument("--lambda-size", type=float, default=1.0)
    p.add_argument("--size-hidden", type=int, default=64)
    p.add_argument("--size-mask-domains", default="", help="domini esclusi dalla MSE di s, separati da virgola (es. bfm)")
    p.add_argument("--scale-aug", default="", help="lo,hi: fattore di scala log-uniforme per mesh (solo global)")
    p.add_argument("--area", default="off", choices=["off", "on", "robust"],
                   help="per area SIA il centro dell'input SIA il pooling, con la scala sqrt(area) (E3): on = pesi "
                        "per area; robust = pesi robusti al rumore (--area-robust); off = v2 per vertice")
    p.add_argument("--area-weights", default="mass", choices=["mass", "smooth", "winsor"],
                   help="pesi per centro, scala e pooling (lo imposta --area)")
    p.add_argument("--area-robust", default="smooth", choices=["smooth", "winsor"])
    p.add_argument("--area-smooth-k", type=int, default=64, help="smooth: autovettori del passa-basso")
    p.add_argument("--winsor-pct", default="1,99", help="winsor: percentili")
    p.add_argument("--ema-decay", type=float, default=0.0)
    p.add_argument("--gt-vector", action="store_true", help="--dist_npz e' una GT a vettori (VectorGT, provvisoria)")
    p.add_argument("--compact-cache", action="store_true")
    p.add_argument("--fast-data", action="store_true",
                   help="sparsi serviti senza riordino e solo 6 tensori sulla GPU (stessi valori, meno CPU)")
    p.add_argument("--resume", default="auto")
    p.add_argument("--ckpt-minutes", type=float, default=20.0,
                   help="checkpoint di ripresa anche a meta' epoca ogni N minuti (0 = solo a fine epoca)")
    p.add_argument("--max-hours", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=0, help="riga di log ogni N passi (0 = solo per epoca)")
    p.add_argument("--test-crash-at-step", type=int, default=0, help="solo test: il rank piu' alto esce al passo N "
                                                                     "al primo tentativo di torchrun")
    return p


GT_FILES = {"maxabs": common.REPO_ROOT / "datasets/SCALE_ALL/gt_joint_bfm_ict_gnm.npz",
            "unified": common.REPO_ROOT / "datasets/UNIFIED_GT/train/gt_unified_bfm_ict_gnm.npz"}


def check_args(a: argparse.Namespace) -> None:
    if not a.dist_npz:
        a.dist_npz = str(GT_FILES[a.gt])
    if a.area != "off":
        if a.input_norm != "global":     # global: centro per area, ma scala globale (non per mesh)
            a.input_norm = "sqrt_area"
        if a.pooling == "meanmax":
            a.pooling = "area_meanmax"
        a.area_weights = "mass" if a.area == "on" else a.area_robust
    import area_v3
    lo, hi = (float(x) for x in a.winsor_pct.split(","))
    area_v3.CFG.update(k=int(a.area_smooth_k), lo=lo, hi=hi)
    if a.data_spec and not (a.split_json and (a.stage_root or a.store) and a.total_steps > 0):
        raise SystemExit("--data-spec richiede --split-json, --stage-root (o --store) e --total-steps")
    if not a.data_spec and not a.data_dir:
        raise SystemExit("serve --data-spec o --data_dir")
    if sum([bool(a.lr_steps), a.plateau_patience is not None, bool(a.lr_constant)]) > 1:
        raise SystemExit("--lr-steps, --plateau-patience e --lr-constant sono alternativi")
    if a.batch_domains == "mixed" and a.sampler != "balanced":
        raise SystemExit("--batch-domains mixed va con --sampler balanced")
    if a.batch_domains == "mixed" and a.domain_blocked:
        raise SystemExit("--batch-domains mixed richiede --no-domain-blocked")
    if a.loss == "v2" and a.train_level != "mixed":
        raise SystemExit("loss v2 = train_level mixed")
    if (a.input_norm == "global") != bool(a.scale_table):
        raise SystemExit("--input-norm global va con --scale-table (e --scale-table solo con global)")
    if a.scale_aug and a.input_norm != "global":
        raise SystemExit("--scale-aug ha senso solo con --input-norm global")
    if a.head in ("factorized", "factorized2"):
        if not a.size_table:
            raise SystemExit("--head factorized richiede --size-table")
        if a.forward == "packed":     # groups: model_v3._pool_masked e factorized_v3.group_normalize (10 ottobre)
            raise SystemExit("--head factorized: --forward sequential o groups (packed chiama solo pool_proj)")
        if a.pooling == "meanmax":
            raise SystemExit("--head factorized richiede un pooling per area (--area on|robust)")
        import factorized_v3
        factorized_v3.parse_scale_aug(a.scale_aug)
    elif a.size_table:
        raise SystemExit("--size-table solo con --head factorized")
    if (a.head == "dual") != bool(a.dist_npz_shape):
        raise SystemExit("--head dual va con --dist-npz-shape (e --dist-npz-shape solo con dual)")
    if a.head == "dual" and (a.forward != "sequential" or a.pooling == "meanmax"):
        raise SystemExit("--head dual: --forward sequential e un pooling per area (--area on|robust)")


# --- run dir ----------------------------------------------------------------------------------------

V3_DEFAULTS = {"forward": "sequential", "loss": "v2", "sampler": "v2", "batch_domains": "single",
               "pooling": "meanmax", "input_norm": "maxabs", "area_weights": "mass", "ema_decay": 0.0,
               "compact_cache": False, "gt": "maxabs", "head": "embed"}
# flag della modalita' fattorizzata: fuori dall'hash quando sono al default, cosi' il run dir di un run senza di
# loro resta quello di prima (una ripresa con --resume auto ritrova il suo last.pth)
FACTORIZED_DEFAULTS = {"scale_table": "", "global_unit_mm": 100.0, "global_ops": "areanorm", "head": "embed",
                       "size_table": "", "lambda_size": 1.0, "size_hidden": 64, "scale_aug": "", "gt_scale": 1.0,
                       "size_mask_domains": "", "dist_npz_shape": "", "gt_scale_shape": 1.0, "lambda_form": 1.0,
                       "lambda_shape": 1.0}
# flag aggiunti dopo i run validati, fuori dall'hash al default per la stessa ragione
LATER_DEFAULTS = {"lr_constant": False}


def make_run_dir(args: argparse.Namespace) -> Path:
    """Nome alla v1 (livello, coppie, modello, rank, id, z, w, b, bs, pool, rumore, seme) + i flag v3 diversi
    dal default + hash di tutti gli argomenti che cambiano il training."""
    skip = {"runs_root", "device", "resume", "max_hours", "log_every", "test_crash_at_step", "cache_workers",
            "cache_max_gb", "prepass_proc", "train_threads", "stage_root", "pin_cache", "data_dir", "size_init"}
    skip |= {k for k, d in {**FACTORIZED_DEFAULTS, **LATER_DEFAULTS}.items() if getattr(args, k, d) == d}
    fp = {k: v for k, v in sorted(vars(args).items()) if k not in skip}
    h = hashlib.sha1(json.dumps(fp, sort_keys=True, default=str).encode()).hexdigest()[:8]
    v3 = [f"{k.replace('_', '')}-{getattr(args, k)}" for k, d in V3_DEFAULTS.items() if getattr(args, k) != d]
    parts = ["v3", "mixed", tr._pair_mode_tag(args.train_pair_mode), args.model,
             f"rank{args.lambda_rank:.2f}" if args.lambda_rank > 0 else "norank",
             f"id{args.lambda_id:.2f}" if args.use_id_loss else "noid", f"z{args.latent_dim}", f"w{args.width}",
             f"b{args.n_blocks}", f"bs{args.batch_subjects}", f"pool{args.pool_mode}",
             f"noise{tr._format_prob_tag(args.p_noise)}", *v3, f"seed{args.seed}"]
    run_dir = Path(args.runs_root).expanduser().resolve() / f"{slugify_token('_'.join(parts))}__{h}"
    return run_dir


# --- utility DDP ------------------------------------------------------------------------------------

def _dist():
    import torch.distributed as dist
    return dist if dist.is_available() and dist.is_initialized() else None


def barrier() -> None:
    d = _dist()
    if d is not None:
        d.barrier()


def _comm_device() -> torch.device:
    d = _dist()
    return torch.device("cuda", torch.cuda.current_device()) if d is not None and d.get_backend() == "nccl" \
        else torch.device("cpu")


def all_max(x: float) -> float:
    d = _dist()
    if d is None:
        return float(x)
    t = torch.tensor([float(x)], device=_comm_device(), dtype=torch.float64)
    d.all_reduce(t, op=d.ReduceOp.MAX)
    return float(t.item())


def check_sync(model: torch.nn.Module) -> float:
    """DDP: i pesi devono essere IDENTICI su tutti i rank (impronta somma e somma dei quadrati, float64)."""
    d = _dist()
    if d is None:
        return 0.0
    ps = [p.detach().double() for p in model.parameters()]
    t = torch.stack([sum(p.sum() for p in ps), sum((p * p).sum() for p in ps)]).reshape(1, 2).to(_comm_device())
    out = [torch.zeros_like(t) for _ in range(d.get_world_size())]
    d.all_gather(out, t)
    spread = float((torch.cat(out).max(0).values - torch.cat(out).min(0).values).abs().max())
    if spread != 0.0:
        raise RuntimeError(f"DDP: pesi diversi fra i rank (scarto dell'impronta {spread:.3e})")
    return spread


def all_mean(x: float) -> float:
    d = _dist()
    if d is None:
        return float(x)
    t = torch.tensor([float(x)], device=_comm_device(), dtype=torch.float64)
    d.all_reduce(t)
    return float(t.item()) / d.get_world_size()


class ServeOnGet(dict):
    """Cache dell'eval che ricostruisce gli sparsi a ogni lettura. Serve SOLO su CPU: l'eval v1 gira in
    inference_mode e ``.to('cpu')`` non copia, quindi uno sparso creato fuori (al preload) fa fallire
    ``L.unsqueeze`` ("Cannot set version_counter for inference tensor"). Su GPU la copia lo evita e la cache
    resta quella di v2. Stessi valori."""

    def __getitem__(self, k):
        return dv._serve(super().__getitem__(k))


# --- EMA --------------------------------------------------------------------------------------------

class EMA:
    """Media mobile esponenziale dei parametri (i buffer si copiano). ``model`` e' la copia da valutare."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        ema_p = [p for p in self.model.parameters()]
        src_p = [p.detach() for p in model.parameters()]
        torch._foreach_mul_(ema_p, self.decay)
        torch._foreach_add_(ema_p, src_p, alpha=1.0 - self.decay)
        for b_ema, b in zip(self.model.buffers(), model.buffers()):
            b_ema.copy_(b)


# --- dati: soggetti, split, blocchi -----------------------------------------------------------------

def topology_map(files, subj_map, groups: dict | None):
    """train_runner._build_subject_topology_map con i gruppi di etichette di train_steps.install_label_groups."""
    if groups and "by_domain" in groups:
        inv_dom = {d: {lab: g for g, labs in gr.items() for lab in labs} for d, gr in groups["by_domain"].items()}
    elif groups:
        inv_dom = {None: {lab: g for g, labs in groups.items() for lab in labs}}
    else:
        inv_dom = {}
    default = inv_dom.get(None, {})
    out = {}
    for sid, idxs in subj_map.items():
        table = inv_dom.get(domain_of(sid), default) if inv_dom else {}
        topo = {}
        for idx in idxs:
            lab = infer_topology_label_from_name(str(files[int(idx)]), sid) or "unknown"
            topo.setdefault(str(table.get(lab, lab)), []).append(int(idx))
        out[str(sid)] = topo
    return out


def select_online(args, held: list[str], split: dict | None) -> list[str]:
    """Soggetti dell'eval online come in v2: dominio unico (train_v2._single_domain_eval) e, se lo split lo
    dichiara, la lista esplicita (train_steps); altrimenti la scelta v1 (_select_online_eval_subjects)."""
    subjects = [str(s) for s in held]
    if args.domain_blocked:
        counts = {}
        for s in subjects:
            counts[domain_of(s)] = counts.get(domain_of(s), 0) + 1
        dom = args.eval_domain or max(counts, key=lambda d: (counts[d], d))
        if dom not in counts:
            raise ValueError(f"--eval_domain {dom} assente dai soggetti di eval {counts}")
        subjects = [s for s in subjects if domain_of(s) == dom]
        log0(f"[v3] eval online dominio={dom} soggetti={len(subjects)}/{len(held)} tutti={dict(sorted(counts.items()))}")
    if split is not None and split.get("online_eval"):
        fixed = list(split["online_eval"])
        missing = sorted(set(fixed) - set(subjects))
        if missing:
            raise SystemExit(f"eval online esplicito: {missing[:3]} non sono held-out di questo run")
        return fixed
    return tr._select_online_eval_subjects(eval_subjects=subjects, max_subjects_eval_train=args.max_subjects_eval_train,
                                           seed=args.seed)


class Data:
    """Dataset, mappe, GT, split e blocchi; per rank lo shard dei soggetti."""

    def __init__(self, args: argparse.Namespace, rank: int, world: int) -> None:
        self.args, self.rank, self.world = args, rank, world
        spec = json.loads(Path(args.data_spec).read_text()) if args.data_spec else {}
        self.spec = spec
        aug = spec.get("aug") or None
        if aug and not (float(aug.get("rot_deg", 0)) > 0 or float(aug.get("reflect_p", 0)) > 0):
            aug = None
        gframe = None
        if args.input_norm == "global":
            import global_v3
            gframe = global_v3.GlobalFrame(global_v3.ScaleTable(args.scale_table.split(",")), args.global_unit_mm,
                                           args.area_weights, args.global_ops)
            log0(f"[v3] ingresso globale: {len(gframe.table.area)} mesh nelle tabelle di scala, L0 "
                 f"{args.global_unit_mm:g} mm, centro con pesi {args.area_weights}, operatori {args.global_ops}")
        self.transform = dv.ServeTransform(args.input_norm, aug, int(spec.get("aug_seed", 0)),
                                           area_weights=args.area_weights, global_frame=gframe)
        if args.store:
            self.dataset = dv.StoreDataset(dv.MmapStore(args.store), self.transform)
            log0(f"[v3] store {args.store}: {len(self.dataset)} mesh in mmap (nessuno staging)")
        elif args.data_spec:
            sources = dv.collect_sources(spec)
            self.dataset = dv.BlockedDataset(sources, {
                "labels": spec.get("labels"), "convention": spec.get("convention", "areanorm"),
                "prepass_proc": max(1, int(args.prepass_proc) // world), "cache_workers": int(args.cache_workers),
                "cache_max_gb": float(args.cache_max_gb), "pin": bool(args.pin_cache), "compact": bool(args.compact_cache),
                "canon": spec.get("canon") or None, "tar_index": spec.get("tar_index"),
                "view_bytes_json": spec.get("view_bytes_json"), "prepass_grad": args.prepass_grad,
                "tolerate": int(args.prepass_tolerate)}, self.transform)
            log0(f"[v3] data-spec {args.data_spec}: {len(sources)} mesh, staging in {args.stage_root}")
        else:
            self.dataset = None   # costruito dopo lo split (la cache tiene solo shard + eval)
            self._dir_files = dv.GTReadyDatasetNPZ(args.data_dir).files
        files = self.dataset.files if self.dataset is not None else self._dir_files
        self.files = files
        self.subj_map = build_subject_map(files, subject_re=SUBJECT_RE_ANY)
        self.topo_map = topology_map(files, self.subj_map, spec.get("label_groups"))
        self.gt, self.name_to_idx = dv.load_gt(args.dist_npz, keep_scale=args.gt_keep_scale, vector=args.gt_vector)
        if args.gt_scale != 1.0:
            self.gt *= np.float32(args.gt_scale) if self.gt.dtype == np.float32 else args.gt_scale
            log0(f"[v3] GT moltiplicata per {args.gt_scale:g} (--gt-scale)")
        subjects = sorted(s for s in self.subj_map if s in self.name_to_idx)
        self.split = json.loads(Path(args.split_json).read_text()) if args.split_json else None
        if self.split is None:
            self.train, self.held = rebuild_subject_split(subjects=subjects, eval_fraction=0.2, seed=args.seed,
                                                          max_subjects=0)
        else:
            have = set(subjects)
            self.train = sorted(s for s in self.split["train"] if s in have)
            self.held = sorted(s for s in self.split["heldout"] if s in have)
            if set(self.train) & set(self.held):
                raise SystemExit("split-json: soggetti sia in train sia in heldout")
            log0(f"[v3] split esplicito {args.split_json}: train={len(self.train)} heldout={len(self.held)} "
                 f"(fuori dallo split: {len(have) - len(self.train) - len(self.held)})")
        if args.frozen_heldout:
            fz = json.loads(Path(args.frozen_heldout).read_text())
            frozen = set(fz["bfm"]) | set(fz["ict_view"]) | set(fz.get("gnm", []))
            leak = sorted(set(self.train) & frozen)
            if leak:
                raise SystemExit(f"GUARDIA HELD-OUT: {len(leak)} soggetti di test congelati nel training (primo {leak[0]})")
            log0(f"[v3] guardia held-out OK: nessuno dei {len(frozen)} soggetti congelati e' di training")
        self.transform.train_set = set(self.train)
        self.online = select_online(args, self.held, self.split)
        self.extra = {d: list(ids) for d, ids in ((self.split or {}).get("online_eval_extra") or {}).items()}
        bad = sorted({s for ids in self.extra.values() for s in ids} - set(self.held))
        if bad:
            raise SystemExit(f"online_eval_extra: {bad[:3]} non sono held-out")
        self.counts = {}
        for s in self.train:
            self.counts[domain_of(s)] = self.counts.get(domain_of(s), 0) + 1
        self.gt_shape = self.n2i_shape = None
        if args.head == "dual":            # GT di u (GT-SR); data.gt resta quella di z_F (GT-FR)
            self.gt_shape, self.n2i_shape = dv.load_gt(args.dist_npz_shape, keep_scale=args.gt_keep_scale)
            if args.gt_scale_shape != 1.0:
                self.gt_shape *= np.float32(args.gt_scale_shape) if self.gt_shape.dtype == np.float32 else args.gt_scale_shape
            miss = sorted((set(self.train) | set(self.online)) - set(self.n2i_shape))
            if miss:
                raise SystemExit(f"--dist-npz-shape: {len(miss)} soggetti assenti ({miss[:3]})")
        self.log_cs, self.size_init = None, 0.0
        if args.head in ("factorized", "factorized2"):      # log centroid size per identita' (bersaglio di s)
            import factorized_v3    # qui ``fz`` e' il json del held-out congelato (sopra)
            self.log_cs = factorized_v3.load_log_cs(args.size_table)
            need = set(self.train) | set(self.online) | {s for ids in self.extra.values() for s in ids}
            miss = sorted(need - set(self.log_cs))
            if miss:
                raise SystemExit(f"--size-table {args.size_table}: {len(miss)} soggetti senza centroid size ({miss[:3]})")
            self.size_init = float(np.mean([self.log_cs[s] for s in self.train]))
            log0(f"[v3] testa fattorizzata: log centroid size di {len(need)} soggetti, media del training "
                 f"{self.size_init:.4f} (S = {math.exp(self.size_init):.1f} mm)")
        self.blocks = None
        self.pending = None
        self.n_fixed_parts = 0

    # shard del rank
    def my(self, subjects) -> list[str]:
        return dv.shard_subjects(subjects, self.rank, self.world, self.args.seed)

    def setup(self, epochs: int) -> None:
        args = self.args
        eval_ids = sorted(set(self.online) | {s for ids in self.extra.values() for s in ids}) if self.rank == 0 else []
        if args.data_spec:
            K = int(self.spec.get("n_blocks", 1))
            if K > epochs:
                raise SystemExit(f"n_blocks={K} > epoche={epochs}: qualche blocco non verrebbe mai addestrato")
            self.blocks = partition_blocks_logged(self.train, self.spec)
            if eval_ids and not args.store:
                names = self.dataset.names_of(eval_ids)
                root = self.root()
                self.dataset.stage(names, root / "eval")
                ds_, local_, miss_ = self.dataset.load(root / "eval", names)
                self.dataset._parts.append((ds_, local_))
                self.drop_meshes(miss_)
                shutil.rmtree(root / "eval", ignore_errors=True)
                shutil.rmtree(root / "eval_geom", ignore_errors=True)
                self.n_fixed_parts = 1
        else:
            keep = set(self.my(self.train)) | set(eval_ids)
            self.dataset = dv.DirDataset(args.data_dir, self.transform, cache=not args.no_cache,
                                         workers=args.cache_workers, compact=args.compact_cache,
                                         subjects=None if self.world == 1 else keep)

    def drop_meshes(self, names: list[str]) -> None:
        """Mesh che il pre-pass non ha prodotto (--prepass-tolerate): fuori dalle mappe del campionatore; un
        soggetto senza mesh esce dai blocchi."""
        if not names:
            return
        pos = {n: i for i, n in enumerate(self.files)}
        for n in names:
            sid, _ = split_name(n)
            i = pos[n]
            if i in self.subj_map.get(sid, []):
                self.subj_map[sid].remove(i)
            for lab, idxs in list(self.topo_map.get(sid, {}).items()):
                if i in idxs:
                    idxs.remove(i)
                    if not idxs:
                        del self.topo_map[sid][lab]
            if not self.subj_map.get(sid):
                if self.blocks is not None:
                    self.blocks = [[s for s in b if s != sid] for b in self.blocks]
                self.train = [s for s in self.train if s != sid]
                print(f"[v3] AVVISO: {sid} senza mesh, escluso dal training", flush=True)

    def root(self) -> Path:
        r = Path(self.args.stage_root)
        return r if self.world == 1 else r / f"rank{self.rank}"

    def block_of_epoch(self, epoch: int, epochs: int) -> int:
        return (epoch - 1) * len(self.blocks) // epochs

    def switch_block(self, k: int) -> None:
        if self.args.store:        # tutto in mmap: il blocco e' solo il pool del campionatore
            self.dataset.resident_block = k
            log0(f"[v3] blocco {k} ({len(self.my(self.blocks[k]))} soggetti) dallo store")
            return
        ds: dv.BlockedDataset = self.dataset
        root = self.root()
        dest = root / f"block{k:03d}"
        names = ds.names_of(self.my(self.blocks[k]))
        if self.pending and self.pending[0] == k:
            ds.finish(self.pending[1], dest)
        else:
            if self.pending:
                dv.kill_proc(self.pending[1])
            ds.stage(names, dest)
        if len(ds._parts) > self.n_fixed_parts:
            dv.mem_log(f"cambio {ds.resident_block}->{k}: prima di liberare")
            ds.drop_parts_after(self.n_fixed_parts)
            dv.mem_log(f"cambio {ds.resident_block}->{k}: blocco {ds.resident_block} liberato (gc + malloc_trim)")
            shutil.rmtree(root / f"block{ds.resident_block:03d}", ignore_errors=True)
        shutil.rmtree(root / f"block{k:03d}_geom", ignore_errors=True)
        t0 = time.time()
        ds_, local_, miss_ = ds.load(dest, names)
        ds._parts.append((ds_, local_))
        self.drop_meshes(miss_)
        shutil.rmtree(dest, ignore_errors=True)
        dv.mem_log(f"blocco {k} caricato, /tmp liberato")
        ds.resident_block = k
        print(f"[v3] rank {self.rank}: blocco {k} residente ({len(self.my(self.blocks[k]))} soggetti) in "
              f"{time.time() - t0:.0f}s", flush=True)
        nxt = k + 1
        self.pending = None
        if nxt < len(self.blocks):
            self.pending = (nxt, ds.stage(ds.names_of(self.my(self.blocks[nxt])), root / f"block{nxt:03d}", wait=False))


def partition_blocks_logged(train, spec):
    blocks = dv.partition_blocks(train, spec)
    pinned_dom = set(spec.get("pin_domains") or [])
    log0(f"[v3] {len(blocks)} blocchi da {[len(b) for b in blocks][:8]}{'...' if len(blocks) > 8 else ''} soggetti "
         f"(residenti in tutti: {sorted(pinned_dom) or '-'})")
    return blocks


# --- epoca ------------------------------------------------------------------------------------------

def epoch_plans(args, data: Data, epoch: int, S: int, steps: int, B: int, drawcfg, seed_r: int):
    """Piano dell'epoca per questo rank: campionatore v2 (default) o bilanciato."""
    if data.blocks is not None:
        pool = data.my(data.blocks[data.dataset.resident_block])
    else:
        pool = data.my(data.train)
    if args.sampler == "v2":
        if (data.blocks is None or len(data.blocks) == 1) and data.world == 1 and steps == S \
                and S == sv.natural_steps(list(data.train), B, args.domain_blocked):
            subset = list(data.train)          # epoca v1 identica: stessa lista
        else:
            subset = sv.epoch_subset(pool, steps, B, seed_r + 31 + epoch, args.domain_blocked,
                                     (data.spec or {}).get("domain_step_share"))
        return sv.plan_epoch_v2(subset, B, seed_r, epoch, args.domain_blocked, data.topo_map, data.subj_map, drawcfg)
    probs = sv.domain_probs(data.counts, args.domain_alpha)
    return sv.plan_epoch_balanced(pool, steps, B, seed_r, epoch, probs, args.batch_domains == "mixed",
                                  data.topo_map, data.subj_map, drawcfg)


STEP_LOG: list = []   # (passo, epoca, loss) del rank 0, scritti in steps_loss.csv a ogni checkpoint e fine epoca


def train_epoch(args, plans, embedder, net, model, optimizer, ema, data: Data, lr_steps, epoch: int, ddp: bool,
                start_batch: int = 0, partial: dict | None = None, saver=None):
    """_train_epoch_mixed del v1 su un piano gia' estratto (stessa sequenza di operazioni per batch).

    Ripresa a meta' epoca: il piano e' deterministico (seme ed epoca), quindi si salta ai batch da
    ``start_batch`` e gli accumulatori ripartono da ``partial``; ``saver(batch_successivo, stato)`` decide se
    scrivere il checkpoint dopo ogni passo."""
    embedder.train()
    keys = ("loss", "stress", "rank", "id", "subject_stress", "subject_rank", "mesh_stress", "mesh_rank")
    acc = dict(partial["acc"]) if partial else {k: 0.0 for k in keys}
    extra_acc: dict = dict(partial["extra"]) if partial else {}
    n_steps = int(partial["n_steps"]) if partial else 0
    n_meshes = int(partial["n_meshes"]) if partial else 0
    t0 = time.time()
    n_new = 0
    stopped_at = None
    for bi, plan in enumerate(plans):
        if bi < start_batch:
            continue
        if not plan.valid():
            if ddp:
                raise RuntimeError("batch non valido in DDP: i rank perderebbero il passo in comune")
            continue
        optimizer.zero_grad(set_to_none=True)
        log_a = None
        if args.scale_aug:     # log a per mesh, funzione di (seme, rank, passo): la ripresa li riproduce
            log_a = fz.draw_log_scales(args.scale_aug, len(plan.entries), args.seed, data.rank, STATE["steps"] + 1)
            embedder.__dict__["log_scales"] = log_a
        Z = net(plan.entries, plan.sigma, bool(args.train_latent_noise))
        batch = StepBatch(Z=Z, mesh_subjects=[e[0] for e in plan.entries], mesh_topos=[e[2] for e in plan.entries],
                          batch_subjects=list(plan.subjects), gt=data.gt, name_to_idx=data.name_to_idx)
        if args.head in ("factorized", "factorized2"):
            loss, terms = fz.factorized_loss(args, batch, data.log_cs, log_a)
        elif args.head == "dual":
            loss, terms = fz.dual_loss(args, batch, data.gt_shape, data.n2i_shape)
        else:
            loss, terms = compute_loss(args.loss, batch, args)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        n = STATE["steps"] + 1
        for boundary, lr in lr_steps:
            if n > boundary and optimizer.param_groups[0]["lr"] != lr:
                for g in optimizer.param_groups:
                    g["lr"] = lr
                log0(f"[v3] passo {n}: lr -> {lr:g} (oltre il passo {boundary})")
        optimizer.step()
        STATE["steps"] = n
        if ema is not None:
            ema.update(model)
        li = float(loss.item())
        acc["loss"] += li
        for k in keys[1:]:
            acc[k] += float(terms.get(k, 0.0))
        for k, v in terms.items():
            if k not in keys:
                extra_acc[k] = extra_acc.get(k, 0.0) + float(v)
        n_steps += 1
        n_new += 1
        n_meshes += len(plan.entries)
        STEP_LOG.append((n, epoch, li))
        if args.log_every and n % args.log_every == 0:
            dt = time.time() - t0
            log0(f"[v3] passo {n}: loss={li:.4f} " + " ".join(f"{k}={v:.4f}" for k, v in terms.items())
                 + f" sigma={plan.sigma:.2e} ({dt / n_new:.3f} s/passo)")
        if saver is not None and bi + 1 < len(plans):
            saver(bi + 1, {"acc": dict(acc), "extra": dict(extra_acc), "n_steps": n_steps, "n_meshes": n_meshes})
        if args.test_crash_at_step and n == args.test_crash_at_step \
                and os.environ.get("TORCHELASTIC_RESTART_COUNT", "0") == "0" and data.rank == data.world - 1:
            raise RuntimeError(f"crash di prova al passo {n} (rank {data.rank})")
        if not ddp and args.max_hours and time.time() - STATE["t_start"] > args.max_hours * 3600:
            STATE["stop"] = True
            stopped_at = bi + 1 if bi + 1 < len(plans) else None
            log0(f"[v3] --max-hours {args.max_hours}: fermo al passo {n}")
            break
        if n >= int(args.total_steps):
            break
    if n_steps == 0:
        raise RuntimeError("No valid optimization step in epoch. Check split/settings.")
    dt = time.time() - t0
    out = {k: v / n_steps for k, v in acc.items()}
    out.update({f"x_{k}": v / n_steps for k, v in extra_acc.items()})
    out["smooth"] = 0.0
    out["teacher"] = 0.0
    out["_n_steps"] = n_steps
    out["_seconds"] = dt
    out["_meshes"] = n_meshes
    out["_new_steps"] = n_new
    out["_stopped_at"] = stopped_at      # batch da cui riprendere se fermato a meta' epoca (--max-hours)
    out["_partial"] = {"acc": dict(acc), "extra": dict(extra_acc), "n_steps": n_steps, "n_meshes": n_meshes}
    return out


# --- training ---------------------------------------------------------------------------------------

def configure_device(args, local_rank: int, ddp: bool) -> torch.device:
    """train_runner._configure_device (TF32 come in v2) con il device del rank."""
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        return torch.device("cpu")
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    except Exception:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return torch.device("cuda", local_rank) if ddp else torch.device("cuda")


def truncate_csv(path: Path, keep) -> None:
    """Riscrive ``path`` tenendo l'intestazione (se c'e') e le righe per cui ``keep(campi)`` e' vero."""
    if not path.exists():
        return
    lines = path.read_text().splitlines()
    out = []
    for ln in lines:
        fields = ln.split(",")
        try:
            float(fields[0])
        except ValueError:
            out.append(ln)            # intestazione
            continue
        if keep(fields):
            out.append(ln)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text("".join(x + "\n" for x in out))
    os.replace(tmp, path)


def _save(obj: dict, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def run(args: argparse.Namespace) -> None:
    rank, world, local_rank = dist_info()
    ddp = world > 1
    if ddp:
        import torch.distributed as dist
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.set_device(local_rank)
            dist.init_process_group("nccl", timeout=timedelta(hours=3))
        else:   # solo per i test della logica DDP senza GPU
            dist.init_process_group("gloo", timeout=timedelta(hours=3))
    if args.train_threads > 0:
        torch.set_num_threads(int(args.train_threads))
    dv.check_gt_scale(args.dist_npz, args.gt_keep_scale)

    noise_modes = parse_noise_modes(args.noise_modes)
    noise_probs = parse_noise_mode_weights(args.noise_mode_weights, noise_modes)
    perturbation = PerturbationParams.from_namespace(args)
    seed_everything(args.seed)
    device = configure_device(args, local_rank, ddp)
    if device.type == "cpu" and str(args.device).startswith("cuda"):
        log0("[v3] AVVISO: nessuna GPU visibile, si gira su CPU")

    run_dir = make_run_dir(args)
    if rank == 0:
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps({"args": vars(args), "noise_modes": noise_modes,
                                                         "world_size": world}, indent=2, sort_keys=True, default=str))
        tr._write_launch_summary(run_dir, args)
    barrier()
    log0(f"Device={device} world={world}\nRun dir: {run_dir}\nModel: {args.model} pooling={args.pooling} "
         f"input_norm={args.input_norm} forward={args.forward} loss={args.loss} sampler={args.sampler}")

    data = Data(args, rank, world)
    B = int(args.batch_subjects)
    if int(args.total_steps) > 0:
        S = int(args.steps_per_epoch) or sv.natural_steps(data.train, B, args.domain_blocked)
        epochs = math.ceil(int(args.total_steps) / S)
    else:
        S = sv.natural_steps(data.train, B, args.domain_blocked)
        epochs = int(args.epochs)
        args.total_steps = S * epochs
    if int(args.save_every) >= int(args.epochs):
        args.save_every = epochs
    args.epochs = epochs
    log0(f"[v3] T={args.total_steps} passi, S={S} per epoca (per rank), epoche={epochs}, batch globale "
         f"{world}x{B} soggetti")
    data.setup(epochs)

    online_plan = build_eval_plan(subj_map=data.subj_map, eval_subjects=data.online,
                                  max_meshes_per_subject_eval=int(args.max_meshes_per_subject_eval),
                                  seed=int(args.seed) + 91_000) if rank == 0 else {}
    if args.head in ("factorized", "factorized2"):
        args.size_init = data.size_init        # bias iniziale di s: media di log S sul training
    model = build_model_v3(args, device)
    if rank > 0:
        torch.manual_seed(args.seed + 100_003 * rank)
        torch.cuda.manual_seed(args.seed + 100_003 * rank)
    n_par = sum(p.numel() for p in model.parameters())
    log0(f"[v3] parametri: {n_par}")
    if args.init_checkpoint:
        pack = tr._load_checkpoint_bundle(tr._resolve_init_checkpoint(args.init_checkpoint))
        model.load_state_dict(pack["state_dict"] if "state_dict" in pack else pack, strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_steps = sorted((int(b_), float(v)) for b_, v in (x.split(":") for x in args.lr_steps.split(",") if x)) \
        if args.lr_steps else []
    scheduler = None
    if lr_steps:
        log0(f"[v3] lr {args.lr:g}, a passi {lr_steps} (--lr-steps)")
    elif args.lr_constant:
        log0(f"[v3] lr {args.lr:g} costante (--lr-constant)")
    else:
        pp = 8 if args.plateau_patience is None else int(args.plateau_patience)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=pp)
        log0(f"[v3] lr {args.lr:g}, ReduceLROnPlateau(factor 0.5, patience {pp}) sulla loss d'epoca"
             + (" (default v2: nessuno fra --lr-steps, --plateau-patience, --lr-constant)"
                if args.plateau_patience is None else ""))
    ema = EMA(model, args.ema_decay) if (args.ema_decay > 0 and rank == 0) else None

    embedder = StepEmbedder(model, args.forward, args.bucket_ratio, args.pad_slack).to(device)
    embedder.fast_data = bool(args.fast_data)
    dv.FAST["on"] = bool(args.fast_data)
    net = embedder
    if ddp:
        from torch.nn.parallel import DistributedDataParallel
        net = DistributedDataParallel(embedder, device_ids=[local_rank] if device.type == "cuda" else None,
                                      broadcast_buffers=False, find_unused_parameters=False)

    # ripresa: last.pth (pesi, ottimizzatore, scheduler, EMA, passo, epoca e batch dentro l'epoca, accumulatori,
    # best, ultima eval) + rng_rank<r>.pt per ogni rank (generatori CPU e CUDA allo stesso passo)
    start_epoch, best = 1, {"auc": (-1e9, -1), "clean": (-1e9, -1), "xtopo": (-1e9, -1)}
    ck_dir = run_dir / "checkpoints"
    last_path = ck_dir / "last.pth"
    resume_from, resume_pack, start_batch, partial = None, None, 0, None
    if args.resume == "auto" and last_path.exists():
        resume_from = last_path
    elif args.resume not in ("auto", "none"):
        resume_from = Path(args.resume)
    if resume_from is not None:
        pack = torch.load(resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(pack["state_dict"])
        optimizer.load_state_dict(pack["optimizer"])
        if scheduler is not None and pack.get("scheduler"):
            scheduler.load_state_dict(pack["scheduler"])
        if ema is not None and pack.get("ema_state_dict"):
            ema.model.load_state_dict(pack["ema_state_dict"])
        STATE["steps"] = int(pack["step"])
        if pack.get("in_epoch") is not None:          # salvato a meta' epoca: si riprende la stessa epoca
            start_epoch, start_batch, partial = int(pack["epoch"]), int(pack["in_epoch"]), pack["partial"]
        else:
            start_epoch = int(pack["epoch"]) + 1
        best = pack.get("best", best)
        resume_pack = pack
        if rank == 0:   # log coerenti col checkpoint: niente righe doppie se il processo e' morto fra log e salvataggio
            for f in ("train_log.csv", "robustness_grid.csv", "xtopo_mesh_log.csv", "mixed_train_log.csv",
                      "loss_terms.csv", "steps_log.csv"):
                truncate_csv(run_dir / f, lambda row: int(float(row[0])) < start_epoch)
            for f in ("steps_loss.csv", "extra_eval.csv"):
                truncate_csv(run_dir / f, lambda row: int(float(row[0])) <= STATE["steps"])
        log0(f"[v3] RIPRESA da {resume_from}: epoca {start_epoch} dal batch {start_batch}, passo {STATE['steps']}, "
             f"world {world} (il checkpoint era a world {pack.get('world_size')})")
    if data.blocks is not None:
        data.switch_block(data.block_of_epoch(start_epoch, epochs))
    barrier()

    # eval online (solo rank 0), dopo il primo blocco come in v2
    eval_ctx = mesh_ctx = None
    sigma_grid = build_sigma_grid(args.sigma_min_eval, args.sigma_max_eval, args.n_sigma_eval)
    log_csv, robust_csv = run_dir / "train_log.csv", run_dir / "robustness_grid.csv"
    xtopo_csv, mixed_csv = run_dir / "xtopo_mesh_log.csv", run_dir / "mixed_train_log.csv"
    if rank == 0:
        cache = preload_eval_samples(dataset=data.dataset, eval_plan=online_plan,
                                     workers=int(args.preload_eval_workers_train)) \
            if args.preload_eval_samples_train else None
        if cache is not None and device.type == "cpu":
            cache = ServeOnGet(cache)
        summ = summarize_eval_plan(online_plan)
        (run_dir / "online_eval_summary.json").write_text(json.dumps(
            {"n_subjects_selected": len(data.online), "selected_subjects": list(data.online), "plan_summary": summ,
             "preloaded_samples": len(cache or {})}, indent=2, sort_keys=True))
        log0(f"Online eval: subjects={len(data.online)}/{len(data.held)} meshes={summ['total_meshes']} "
             f"preloaded={len(cache or {})}")
        eval_ctx = SubjectEvalContext(dataset=data.dataset, subj_map=data.subj_map, eval_subjects=data.online,
                                      name_to_idx=data.name_to_idx, gt_matrix=data.gt, device=device,
                                      max_meshes_per_subject_eval=args.max_meshes_per_subject_eval,
                                      eval_plan=online_plan, sample_cache=cache)
        records = build_sample_eval_records(dataset=data.dataset, eval_plan=online_plan,
                                            eval_subjects=data.online, sample_cache=cache)
        mesh_ctx = build_pair_eval_context(sample_records=records, name_to_idx=data.name_to_idx, gt_matrix=data.gt,
                                           device=device, pair_mode=str(args.train_pair_mode),
                                           aggregation_level="mesh_pair")
        if not log_csv.exists():
            tr._write_log_headers(log_csv, robust_csv, xtopo_mesh_log_csv=xtopo_csv, mixed_log_csv=mixed_csv)
    extra_ctx: dict = {}

    drawcfg = sv.DrawCfg(p_noise=args.p_noise, sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                         noise_modes=noise_modes, noise_mode_probs=noise_probs,
                         max_meshes=int(args.max_meshes_per_subject_train))
    embedder.bind(data.dataset, perturbation)
    seed_r = int(args.seed) + 1_000_003 * rank
    last_eval = {k: float("nan") for k in ("spearman_clean", "pearson_clean", "auc_r", "gate_mean_clean",
                                           "gate_mean_noisy_max", "spearman_noisy_max", "ratio_noisy_max",
                                           "intra_clean", "xtopo_mesh_clean", "xtopo_mesh_pearson")}
    last_eval.update(n_eval=0, xtopo_mesh_n_pairs=0)

    if resume_pack is not None:
        rng_path = ck_dir / f"rng_rank{rank}.pt"
        rng = torch.load(rng_path, map_location="cpu", weights_only=False) if rng_path.exists() else None
        if rng is not None and int(rng["step"]) == STATE["steps"] and int(rng["world"]) == world:
            torch.set_rng_state(rng["cpu"])
            if rng.get("cuda") is not None and device.type == "cuda":
                torch.cuda.set_rng_state(rng["cuda"])
            log0(f"[v3] generatori ripristinati al passo {STATE['steps']}")
        else:   # world diverso o file mancante: semi deterministici, la traiettoria non e' piu' quella di prima
            torch.manual_seed(args.seed + 100_003 * rank + 17 * start_epoch + STATE["steps"])
            torch.cuda.manual_seed(args.seed + 100_003 * rank + 17 * start_epoch + STATE["steps"])
            log0(f"[v3] generatori NON ripristinati (world o passo diversi): risemenzati")
        last_eval.update(resume_pack.get("last_eval") or {})

    def flush_steps() -> None:
        if rank == 0 and STEP_LOG:
            new = not (run_dir / "steps_loss.csv").exists()
            with open(run_dir / "steps_loss.csv", "a") as f:
                if new:
                    f.write("step,epoch,loss\n")
                f.writelines(f"{a},{b},{c:.8f}\n" for a, b, c in STEP_LOG)
            STEP_LOG.clear()

    def save_last(epoch_: int, in_epoch, partial_) -> None:
        """Checkpoint di ripresa: generatori di OGNI rank, poi last.pth dal rank 0 (scritture atomiche)."""
        _save({"step": STATE["steps"], "world": world, "cpu": torch.get_rng_state(),
               "cuda": torch.cuda.get_rng_state() if device.type == "cuda" else None}, ck_dir / f"rng_rank{rank}.pt")
        barrier()
        if rank == 0:
            flush_steps()
            save_ckpt(last_path, epoch_, model, optimizer, ema, args,
                      {"step": STATE["steps"], "best": best, "world_size": world, "in_epoch": in_epoch,
                       "partial": partial_, "last_eval": dict(last_eval),
                       "scheduler": scheduler.state_dict() if scheduler is not None else None},
                      with_ema_file=False)
        barrier()

    ck_clock = {"t": time.time()}

    def saver(epoch_: int):
        def fn(next_batch: int, partial_: dict) -> None:
            due = args.ckpt_minutes > 0 and time.time() - ck_clock["t"] >= args.ckpt_minutes * 60
            if ddp:
                due = all_max(float(due)) > 0
            if due:
                save_last(epoch_, next_batch, partial_)
                ck_clock["t"] = time.time()
                log0(f"[v3] checkpoint di ripresa: epoca {epoch_}, batch {next_batch}, passo {STATE['steps']}")
        return fn

    for epoch in range(start_epoch, epochs + 1):
        if STATE["stop"] or STATE["steps"] >= int(args.total_steps):
            break
        if data.blocks is not None:
            k = data.block_of_epoch(epoch, epochs)
            if k != data.dataset.resident_block:
                data.switch_block(k)
                barrier()
        sb, part = (start_batch, partial) if epoch == start_epoch else (0, None)
        # passi dell'epoca: quelli del piano intero (anche se si riprende a meta')
        steps = min(S, int(args.total_steps) - STATE["steps"] + (part["n_steps"] if part else 0))
        plans = epoch_plans(args, data, epoch, S, steps, B, drawcfg, seed_r)
        before = STATE["steps"] - (part["n_steps"] if part else 0)
        stats = train_epoch(args, plans, embedder, net, model, optimizer, ema, data, lr_steps, epoch, ddp,
                            start_batch=sb, partial=part, saver=saver(epoch))
        done = STATE["steps"] - before
        stats_loss = all_mean(stats["loss"])
        if ddp:
            check_sync(model)
            if args.max_hours and all_max(time.time() - STATE["t_start"]) > args.max_hours * 3600:
                STATE["stop"] = True
                log0(f"[v3] --max-hours {args.max_hours}: fermo a fine epoca {epoch}")
        nn_ = max(stats["_new_steps"], 1)
        log0(f"\n[v3] epoca {epoch}: {done} passi ({stats['_new_steps']} in questo processo) in {stats['_seconds']:.0f}s "
             f"({stats['_seconds'] / nn_:.3f} s/passo)")
        if done != steps and not STATE["stop"]:
            log0(f"[v3] AVVISO epoca {epoch}: {done} passi eseguiti, attesi {steps}")
        if scheduler is not None:
            scheduler.step(stats_loss)
        lr_now = float(optimizer.param_groups[0]["lr"])

        final = epoch == epochs or STATE["stop"] or STATE["steps"] >= int(args.total_steps)
        do_eval = args.eval_every > 0 and (epoch % args.eval_every == 0 or final)
        if rank == 0:
            eval_model = ema.model if ema is not None else model
            if do_eval:
                fz.set_output(eval_model, "eval")  # fattorizzati: u (forma); dual: z_F contro data.gt (FR)
                pack = evaluate_subject_robustness_grid(model=eval_model, eval_ctx=eval_ctx, sigma_grid=sigma_grid,
                                                        noise_modes=noise_modes, params=perturbation,
                                                        seed=args.seed + 50_000 + epoch, eval_mode=args.eval_mode)
                last_eval.update({k: float(pack[k]) for k in ("spearman_clean", "pearson_clean", "auc_r",
                                                              "gate_mean_clean", "gate_mean_noisy_max",
                                                              "spearman_noisy_max", "ratio_noisy_max")})
                last_eval.update(intra_clean=float(pack["clean"]["intra_mean"]), n_eval=int(pack["n_eval"]))
                tr._append_eval_outputs(run_dir=run_dir, robust_csv=robust_csv, epoch=epoch, eval_pack=pack)
                eval_extra(args, data, eval_ctx, extra_ctx, eval_model, sigma_grid, noise_modes, perturbation,
                           epoch, run_dir)
                me = tr._evaluate_mesh_pair_context_clean(model=eval_model, dataset=data.dataset, pair_ctx=mesh_ctx,
                                                          sample_cache=eval_ctx.sample_cache, device=device)
                last_eval.update(xtopo_mesh_clean=float(me["spearman"]), xtopo_mesh_pearson=float(me["pearson"]),
                                 xtopo_mesh_n_pairs=int(me["n_pairs"]))
                for key, metric in (("auc", "auc_r"), ("clean", "spearman_clean"), ("xtopo", "xtopo_mesh_clean")):
                    v = float(last_eval[metric])
                    if np.isfinite(v) and v > best[key][0]:
                        best[key] = (v, epoch)
                        name = {"auc": "best_by_auc", "clean": "best_by_clean", "xtopo": "best_by_xtopo_mesh_clean"}[key]
                        save_ckpt(run_dir / "checkpoints" / f"{name}.pth", epoch, model, optimizer, ema, args,
                                  {f"best_{metric}": v, "sigma_grid": sigma_grid})
                        (run_dir / f"{name}.txt").write_text(f"best_epoch={epoch}\nbest_{metric}={v}\n")
                fz.set_output(eval_model, "full")
            print(f"Epoch {epoch:03d} | loss={stats_loss:.4f} stress={stats['stress']:.4f} rank={stats['rank']:.4f} "
                  f"id={stats['id']:.4f} smooth=0.0000 teacher=0.0000 lr={lr_now:.2e} "
                  f"sp_clean={last_eval['spearman_clean']:.4f} aucR={last_eval['auc_r']:.4f} "
                  f"xtopo_mesh={last_eval['xtopo_mesh_clean']:.4f} n_eval={int(last_eval['n_eval'])}"
                  + "".join(f" {k[2:]}={v:.4f}" for k, v in stats.items() if k.startswith("x_")), flush=True)
            with open(log_csv, "a") as f:
                f.write(f"{epoch},{stats_loss:.6f},{stats['stress']:.6f},{stats['rank']:.6f},{stats['id']:.6f},"
                        f"0.000000,0.000000,{lr_now:.2e},{last_eval['spearman_clean']:.6f},"
                        f"{last_eval['pearson_clean']:.6f},{last_eval['auc_r']:.6f},{last_eval['gate_mean_clean']:.6f},"
                        f"{last_eval['gate_mean_noisy_max']:.6f},{last_eval['spearman_noisy_max']:.6f},"
                        f"{last_eval['ratio_noisy_max']:.6f},{last_eval['intra_clean']:.6e},{int(last_eval['n_eval'])}\n")
            with open(xtopo_csv, "a") as f:
                f.write(f"{epoch},{last_eval['xtopo_mesh_clean']:.6f},{last_eval['xtopo_mesh_pearson']:.6f},"
                        f"{int(last_eval['xtopo_mesh_n_pairs'])}\n")
            with open(mixed_csv, "a") as f:
                f.write(f"{epoch},{stats['subject_stress']:.6f},{stats['subject_rank']:.6f},{stats['mesh_stress']:.6f},"
                        f"{stats['mesh_rank']:.6f},{args.lambda_subject:.6f},{args.lambda_mesh:.6f},"
                        f"{args.train_pair_mode}\n")
            xs = {k[2:]: v for k, v in stats.items() if k.startswith("x_")}
            if xs:   # termini delle loss log (grad, scale, inv, nbr, b, d_mean), media dell'epoca sul rank 0
                new = not (run_dir / "loss_terms.csv").exists()
                with open(run_dir / "loss_terms.csv", "a") as f:
                    if new:
                        f.write("epoch,step," + ",".join(sorted(xs)) + "\n")
                    f.write(f"{epoch},{STATE['steps']}," + ",".join(f"{xs[k]:.6f}" for k in sorted(xs)) + "\n")
            if not (run_dir / "steps_log.csv").exists():
                (run_dir / "steps_log.csv").write_text("epoch,step,done,seconds,meshes,world\n")
            with open(run_dir / "steps_log.csv", "a") as f:
                f.write(f"{epoch},{STATE['steps']},{done},{stats['_seconds']:.1f},{stats['_meshes']},{world}\n")
            if epoch % args.save_every == 0 or final:
                save_ckpt(run_dir / "checkpoints" / f"epoch{epoch:03d}.pth", epoch, model, optimizer, ema, args, {})
        barrier()
        if stats["_stopped_at"] is not None:  # fermato a meta' epoca: si riprendera' da li'
            save_last(epoch, stats["_stopped_at"], stats["_partial"])
        else:                                  # fine epoca, dopo l'eval: la ripresa parte dall'epoca successiva
            save_last(epoch, None, None)
        ck_clock["t"] = time.time()

    if data.pending:
        dv.kill_proc(data.pending[1])
    log0("DONE")
    log0(f"Saved in: {run_dir}")
    log0(f"[v3] passi eseguiti: {STATE['steps']} (richiesti {args.total_steps})")
    for key, (v, e) in best.items():
        log0(f"Best {key}={v:.4f} at epoch {e}")
    if ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


def save_ckpt(path: Path, epoch: int, model, optimizer, ema, args, extra: dict, with_ema_file: bool = True) -> None:
    """Chiavi v1 (epoch, state_dict, optimizer, args) + EMA; con EMA anche <nome>_ema.pth, i cui pesi sono
    quelli EMA nella chiave state_dict (i tool di eval la leggono senza sapere dell'EMA)."""
    obj = {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
           "args": vars(args), "trainer": "v3", **extra}
    if ema is not None:
        obj["ema_state_dict"] = ema.model.state_dict()
        obj["ema_decay"] = ema.decay
    _save(obj, path)
    if ema is not None and with_ema_file:
        _save({"epoch": epoch, "state_dict": ema.model.state_dict(), "args": vars(args), "trainer": "v3",
               "weights": f"ema {ema.decay}", **{k: v for k, v in extra.items() if k.startswith("best")}},
              path.with_name(path.stem + "_ema.pth"))


def eval_extra(args, data: Data, eval_ctx, extra_ctx: dict, model, sigma_grid, noise_modes, perturbation, epoch,
               run_dir: Path) -> None:
    """train_steps: la stessa eval online su ogni insieme ``online_eval_extra`` (es. 16 GNM), solo registrata."""
    import dataclasses
    for dom, ids in data.extra.items():
        if dom not in extra_ctx:
            plan = build_eval_plan(subj_map=data.subj_map, eval_subjects=ids,
                                   max_meshes_per_subject_eval=int(args.max_meshes_per_subject_eval),
                                   seed=int(args.seed) + 91_000)
            cache = preload_eval_samples(dataset=data.dataset, eval_plan=plan, workers=2)
            if eval_ctx.device.type == "cpu":
                cache = ServeOnGet(cache)
            extra_ctx[dom] = dataclasses.replace(eval_ctx, eval_subjects=list(ids), eval_plan=plan, sample_cache=cache)
        r = evaluate_subject_robustness_grid(model=model, eval_ctx=extra_ctx[dom], sigma_grid=sigma_grid,
                                             noise_modes=noise_modes, params=perturbation,
                                             seed=args.seed + 50_000 + epoch, eval_mode=args.eval_mode)
        vals = {k: float(r[k]) for k in ("spearman_clean", "pearson_clean", "auc_r")}
        print(f"[v3] eval extra {dom} ({len(ids)} soggetti, passo {STATE['steps']}): "
              + " ".join(f"{k}={v:.4f}" for k, v in vals.items()), flush=True)
        csv = run_dir / "extra_eval.csv"
        new = not csv.exists()
        with open(csv, "a") as fh:
            if new:
                fh.write("step,domain,n_subjects,spearman_clean,pearson_clean,auc_r\n")
            fh.write(f"{STATE['steps']},{dom},{len(ids)},{vals['spearman_clean']:.6f},{vals['pearson_clean']:.6f},"
                     f"{vals['auc_r']:.6f}\n")


def _on_term(signum, frame):
    """SIGTERM (torchrun che riavvia, scancel): chiude i pre-pass in corso, che altrimenti resterebbero orfani."""
    dv.kill_all()
    sys.exit(128 + signum)


def main() -> None:
    import faulthandler
    import signal
    faulthandler.enable()          # stack Python anche su SIGSEGV (segfault del rank 1 al cambio di blocco su CPU)
    signal.signal(signal.SIGTERM, _on_term)
    args = build_parser().parse_args()
    check_args(args)
    try:
        run(args)
    finally:
        dv.kill_all()


if __name__ == "__main__":
    main()
