#!/usr/bin/env python3
"""Split di training ricostruiti, soggetti di eval per cella e viste a symlink per la
tabella cross-3DMM di WS2 (modello BFM-only / ICT-only / congiunto x dominio BFM / ICT).

Il problema che risolve: gli script di ranking rifanno il loro split sui soggetti che
TROVANO nella data dir, ma ICT-only e congiunto non hanno lo split che quegli script
ricostruirebbero. Il training (``robustness/train_runner.py:1557``) chiama
``rebuild_subject_split(subjects, eval_fraction=0.2, seed, max_subjects=0)`` su TUTTI i
soggetti della sua data dir, quindi:

  - ICT-only (job 1019531, ``datasets/ICT/train_ready``): 1000 held-out su 5000, scelti a
    caso fra id10000-id14999. ``split_heldout.txt`` (id14500-id14999) NON e' lo split del
    training: ~80% di ``eval_view_heldout`` era nel suo training.
  - congiunto (job 1019532, ``datasets/JOINT_BFM_ICT``): 1100 held-out su 5500, 108 BFM e
    992 ICT (dal log), di nuovo a caso sull'unione.

Lo split si ricostruisce con le stesse funzioni del training, importate e non riscritte,
e si VERIFICA contro quello che il training ha scritto: conteggi held-out per dominio e i
16 soggetti dell'eval online (``online_eval_summary.json``), che il training estrae dagli
held-out con ``_select_online_eval_subjects`` (per il congiunto dopo il filtro di dominio
di ``train_v2._single_domain_eval``). Se non coincidono lo script si ferma.

Soggetti valutati per cella (sempre e solo held-out del modello valutato):
  - dominio BFM, BFM-only e ICT-only: i 100 held-out del BFM-only (seed 1234), gli stessi
    della sua eval esistente; ICT-only non ha mai visto un soggetto BFM.
  - dominio BFM, congiunto: i suoi 108 held-out BFM.
  - dominio ICT, BFM-only: i 100 dello zero-shot esistente (pool id14500-id14999).
  - dominio ICT, ICT-only e congiunto: held-out del modello INTERSECATI con lo stesso pool
    id14500-id14999. Il pool e' quello che ha le espressioni casuali
    (``expressions_random_withops``), quindi cella ICT e riga espressioni guardano gli
    stessi soggetti, e la popolazione e' la stessa dello zero-shot.

Per ogni cella scrive una vista PIATTA con le sole mesh dei soggetti scelti; gli sbatch la
valutano con ``--subject_split all --max_subjects 0``, cioe' tutti e soli quei soggetti.
Il controllo leak (|valutati inter training| == 0 per ogni cella) e' un assert, e finisce
in ``splits.json`` insieme alle liste, che il summarizer rilegge per rifare lo stesso
controllo sui ``selected_subjects`` scritti dagli script di eval.

  aau/submit.sh cross3dmm/ws2_views.sbatch
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
REPO_ROOT = AAU_DIR.parent
INTRINSIC_DIR = REPO_ROOT / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic"
sys.path.insert(0, str(INTRINSIC_DIR))
sys.path.insert(0, str(REPO_ROOT / "v2_work" / "train_v2"))
sys.path.insert(0, str(AAU_DIR / "ict"))

from robustness import train_runner as v1  # noqa: E402  (stesse funzioni del training)
from robustness.data_utils import GTReadyDataset, rebuild_subject_split  # noqa: E402
from train_v2 import domain_of  # noqa: E402
from make_ict_eval_views import link, read_heldout  # noqa: E402

DATASETS = REPO_ROOT / "datasets"
RUNS = AAU_DIR / "runs"
RUN_DIR_NAME = (
    "mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_ks0_poolmeanmax_noise60_"
    "sig5e-4-2e-2_latentnoise_seed1234__9a81466d"
)
BFM_DIR = DATASETS / "REMESH" / "npz_data_topo_500_withops_areanorm"
BFM_GT = (REPO_ROOT / "face_embedding" / "gt_encdec" / "autoencoder" / "latent_analysis"
          / "gt_distance_matrix" / "normalized_matrix_distances.npz")
ICT_DIR = DATASETS / "ICT" / "train_ready" / "npz_withops"
ICT_GT = DATASETS / "ICT" / "train_ready" / "gt_matrix.npz"
ICT_REXPR_DIR = DATASETS / "ICT" / "expressions_random_withops"

# Data dir e GT del training, dalle righe `[x3dmm] data_dir=` / `dist_npz=` dei log (il
# config.json ha /tmp/<job>/data, cioe' la copia di staging di queste stesse cartelle).
# `n_files` e' il conteggio del log di staging ("30000 file", "33000 file"): se la cartella
# e' cambiata dopo il training lo split ricostruito non vale piu', e lo script si ferma.
# `subject_re`: train_v2 passa SUBJECT_RE_ANY alla GT, il v1 il default a 4 cifre; su BFM
# (id a 4 cifre) danno la stessa mappa, ma si usa quella che ha usato il training.
# `eval_domain`: il filtro di train_v2._single_domain_eval prima dell'eval online
# ("" = dominio piu' numeroso, None = training v1 senza filtro).
MODELS = {
    "bfm_only": {
        "run_dir": RUNS / "remesh_v1recipe_areanorm_s1234_1019310" / RUN_DIR_NAME,
        "data_dir": BFM_DIR, "gt": BFM_GT, "n_files": 3000,
        "subject_re": "4digit", "eval_domain": None,
        "expected_heldout": {"bfm": 100},
    },
    "ict_only": {
        "run_dir": RUNS / "x3dmm_ict_only_s1234_1019531" / RUN_DIR_NAME,
        "data_dir": ICT_DIR, "gt": ICT_GT, "n_files": 30000,
        "subject_re": "any", "eval_domain": "",
        "expected_heldout": {"ict": 1000},
    },
    "joint": {
        "run_dir": RUNS / "x3dmm_joint_bfm_ict_s1234_1019532" / RUN_DIR_NAME,
        "data_dir": DATASETS / "JOINT_BFM_ICT" / "npz_withops",
        "gt": DATASETS / "JOINT_BFM_ICT" / "gt_matrix.npz", "n_files": 33000,
        "subject_re": "any", "eval_domain": "bfm",
        "expected_heldout": {"bfm": 108, "ict": 992},
    },
}

# Eval esistenti del BFM-only (stessa chiave di eval_common.sh): i loro selected_subjects
# sono le liste delle due celle BFM-only, e devono coincidere con quelle ricostruite qui.
EXISTING = {
    "bfm_only__bfm": RUNS / f"eval_{RUN_DIR_NAME}_47e4a6be" / "ranking_rm" / "ranking_summary.json",
    "bfm_only__ict": (RUNS / f"eval_{RUN_DIR_NAME}_3c44d0ba" / "ict_zeroshot" / "ranking"
                      / "ranking_summary.json"),
}

TOPOLOGIES = ("crop", "down8k", "noisy", "original", "remesh", "up60k")
REXPR_RE = re.compile(r"^(id\d+)_rexpr_(\d+)\.npz$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--views-root", type=Path, default=DATASETS / "WS2_CROSS3DMM")
    p.add_argument("--out-json", type=Path, default=RUNS / "ws2_cross3dmm" / "splits.json")
    return p.parse_args()


def reproduce_split(name: str, spec: dict) -> dict:
    """Le righe di run_training che decidono lo split, con le funzioni del training."""
    n_files = len(GTReadyDataset(str(spec["data_dir"])).files)
    if n_files != spec["n_files"]:
        raise SystemExit(f"{name}: {spec['data_dir']} ha {n_files} npz, il training ne aveva "
                         f"{spec['n_files']}: lo split ricostruito non sarebbe quello del training")

    config = json.loads((spec["run_dir"] / "config.json").read_text())
    seed = int(config["args"]["seed"])
    subject_re = v1.SUBJECT_RE_ANY if spec["subject_re"] == "any" else None
    dataset = GTReadyDataset(str(spec["data_dir"]))
    subj_map = v1.build_subject_map(dataset.files, subject_re=v1.SUBJECT_RE_ANY)
    if subject_re is None:
        _, name_to_idx = v1.load_gt_distance_matrix(str(spec["gt"]), dtype=np.float64)
    else:
        _, name_to_idx = v1.load_gt_distance_matrix(str(spec["gt"]), subject_re=subject_re, dtype=np.float64)
    subjects = sorted([sid for sid in subj_map.keys() if sid in name_to_idx])
    train, heldout = rebuild_subject_split(subjects=subjects, eval_fraction=0.2, seed=seed, max_subjects=0)

    # Verifica 1: conteggi held-out per dominio, come stampati nel log del training.
    counts: dict[str, int] = {}
    for sid in heldout:
        counts[domain_of(sid)] = counts.get(domain_of(sid), 0) + 1
    if counts != spec["expected_heldout"]:
        raise SystemExit(f"{name}: held-out per dominio {counts}, il log dice {spec['expected_heldout']}")

    # Verifica 2: i 16 soggetti dell'eval online scritti dal training.
    kept = heldout
    if spec["eval_domain"] is not None:
        dom = spec["eval_domain"] or max(counts, key=lambda d: (counts[d], d))
        kept = [s for s in heldout if domain_of(s) == dom]
    online = v1._select_online_eval_subjects(
        eval_subjects=kept,
        max_subjects_eval_train=int(config["args"]["max_subjects_eval_train"]),
        seed=seed,
    )
    written = json.loads((spec["run_dir"] / "online_eval_summary.json").read_text())["selected_subjects"]
    if list(online) != list(written):
        raise SystemExit(f"{name}: eval online ricostruita {online[:4]}.. diversa da quella del training "
                         f"{written[:4]}..: lo split NON e' quello del training")

    print(f"[ws2-views] {name}: seed={seed} soggetti={len(subjects)} train={len(train)} "
          f"held-out={len(heldout)} {counts}; eval online 16/16 identica al training")
    return {"seed": seed, "train": train, "heldout": heldout, "online_eval": list(written),
            "data_dir": str(spec["data_dir"]), "gt": str(spec["gt"]), "run_dir": str(spec["run_dir"])}


def existing_subjects(path: Path) -> list[str]:
    return sorted(json.loads(path.read_text())["selected_subjects"])


def build_view(out: Path, subjects: list[str], files) -> int:
    """Vista piatta; `files(sid)` da' le coppie (nome nella vista, sorgente)."""
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.npz"):
        stale.unlink()
    n = 0
    for sid in subjects:
        for name, src in files(sid):
            link(out / name, src)
            n += 1
    return n


def topology_files(src_dir: Path):
    return lambda sid: [(f"{sid}_GTready_{t}.npz", src_dir / f"{sid}_GTready_{t}.npz") for t in TOPOLOGIES]


def rexpr_index() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for entry in sorted(p.name for p in ICT_REXPR_DIR.iterdir()):
        m = REXPR_RE.match(entry)
        if m:
            found.setdefault(m.group(1), []).append(m.group(2))
    return found


def main() -> None:
    args = parse_args()
    splits = {name: reproduce_split(name, spec) for name, spec in MODELS.items()}

    ict_pool = read_heldout(DATASETS / "ICT")  # id14500-id14999, il pool di zero-shot e espressioni
    bfm_only_bfm = existing_subjects(EXISTING["bfm_only__bfm"])
    bfm_only_ict = existing_subjects(EXISTING["bfm_only__ict"])
    if bfm_only_bfm != sorted(splits["bfm_only"]["heldout"]):
        raise SystemExit("i 100 soggetti dell'eval BFM-only esistente non sono i suoi held-out")

    def held(model: str, domain: str, pool=None) -> list[str]:
        out = [s for s in splits[model]["heldout"] if domain_of(s) == domain]
        return sorted(set(out) & set(pool)) if pool is not None else sorted(out)

    cells = {
        "bfm_only__bfm": ("bfm_only", "bfm", bfm_only_bfm),
        "bfm_only__ict": ("bfm_only", "ict", bfm_only_ict),
        "ict_only__bfm": ("ict_only", "bfm", bfm_only_bfm),
        "ict_only__ict": ("ict_only", "ict", held("ict_only", "ict", ict_pool)),
        "joint__bfm": ("joint", "bfm", held("joint", "bfm")),
        "joint__ict": ("joint", "ict", held("joint", "ict", ict_pool)),
    }

    # Il controllo che conta: nessun soggetto valutato e' nel training del modello valutato.
    # Per completezza anche quanti erano nei 16 dell'eval online, cioe' hanno pesato sulla
    # scelta del checkpoint (non sul training): stesso protocollo delle eval esistenti.
    report = {}
    for cell, (model, domain, subjects) in cells.items():
        train = set(splits[model]["train"])
        leak = sorted(set(subjects) & train)
        wrong_domain = [s for s in subjects if domain_of(s) != domain]
        report[cell] = {
            "model": model, "domain": domain, "n_eval": len(subjects),
            "n_train_model": len(train), "n_eval_in_train": len(leak),
            "n_eval_in_online_selection": len(set(subjects) & set(splits[model]["online_eval"])),
            "first": subjects[:5], "last": subjects[-3:],
        }
        print(f"[ws2-views] {cell:<14} valutati={len(subjects):>3} training={len(train):>4} "
              f"in comune={len(leak)} (nell'eval online: {report[cell]['n_eval_in_online_selection']}) "
              f"primi={' '.join(subjects[:5])}")
        if leak or wrong_domain or not subjects:
            raise SystemExit(f"{cell}: LEAK {leak[:5]} / dominio sbagliato {wrong_domain[:5]}")

    # Viste. Le celle BFM-only non servono per il ranking (riusato), ma bfm_only__bfm serve
    # al breakdown con pair_metrics, che l'eval esistente non ha scritto (niente CI senza).
    root = args.views_root
    root.mkdir(parents=True, exist_ok=True)
    views = {}
    for cell, (model, domain, subjects) in cells.items():
        if cell == "bfm_only__ict":
            continue  # pair_metrics gia' scritte dallo zero-shot (job 1019695)
        src = BFM_DIR if domain == "bfm" else ICT_DIR
        n = build_view(root / cell, subjects, topology_files(src))
        views[cell] = str(root / cell)
        print(f"[ws2-views] vista {cell}: {n} symlink")

    # Espressioni, regime (c) misto + baseline neutra, per i due modelli che hanno visto ICT.
    index = rexpr_index()
    for model in ("ict_only", "joint"):
        subjects = cells[f"{model}__ict"][2]
        missing = [s for s in subjects if len(index.get(s, [])) != 5]
        if missing:
            raise SystemExit(f"{model}: espressioni casuali incomplete per {missing[:5]}")
        base = root / f"{model}__rexpr"
        n_mixed = build_view(base / "mixed", subjects, lambda sid: [
            (f"{sid}_GTready_rexpr{k}.npz", ICT_REXPR_DIR / f"{sid}_rexpr_{k}.npz") for k in index[sid]])
        n_neutral = build_view(base / "neutral", subjects, lambda sid: [
            (f"{sid}_GTready_neutral.npz", ICT_DIR / f"{sid}_GTready_original.npz")])
        views[f"{model}__rexpr"] = str(base)
        print(f"[ws2-views] vista {model}__rexpr: mixed {n_mixed}, neutral {n_neutral} symlink")

    payload = {
        "models": splits,
        "cells": {c: {"model": m, "domain": d, "subjects": s} for c, (m, d, s) in cells.items()},
        "leak_report": report,
        "views": views,
        "ict_pool": "datasets/ICT/train_ready/split_heldout.txt (id14500-id14999)",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"[ws2-views] scritto {args.out_json}")


if __name__ == "__main__":
    main()
