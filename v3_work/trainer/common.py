"""Percorsi e piccole utility condivise dal trainer v3 (v3_work/trainer/).

Il trainer v3 e' un FORK appiattito della catena v2 (v2_work/fastio/train_steps.py -> train_fast.py ->
v2_work/train_v2/train_v2.py -> robustness/train_runner.py), che era fatta di monkeypatch a strati.
Cosa e' copiato e cosa e' importato:
  * copiato e modificato qui: orchestrazione (blocchi, staging, cache, split, passi, schedule, ciclo di
    training, checkpoint), campionamento delle epoche, forward a gruppi;
  * importato in SOLA LETTURA dal pacchetto congelato v1 (face_embedding/, mai modificato): le funzioni
    numeriche (stress, rank, perturbazioni, campionatore delle mesh per soggetto, eval online, modello
    DiffusionEncoderOnly). E' cio' che rende l'equivalenza con v2 verificabile: stessi mattoni, stesso
    ordine delle chiamate al generatore casuale.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
INTRINSIC_DIR = REPO_ROOT / "face_embedding/gt_encdec/remeshing/intrinsic"
AUTOENC_DIR = REPO_ROOT / "face_embedding/gt_encdec/autoencoder"
DIFFNET_DIR = REPO_ROOT / "diffusion-net/src"
PREPASS = REPO_ROOT / "aau/data_scale/prepass_ops.py"

for _p in (INTRINSIC_DIR, AUTOENC_DIR, DIFFNET_DIR, THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# GNM Head (id100000-199999) prima degli offset, come train_v2.GNM_RANGE; ICT da 10000 prima di FaceScape.
GNM_RANGE = (100000, 200000)
DOMAIN_OFFSETS = ((10000, "ict"), (3000, "facescape"), (2000, "facescape"), (1000, "flame"), (0, "bfm"))


def domain_of(subject_id: str) -> str:
    """Dominio dall'id, IDENTICO a train_v2.domain_of (bfm 0, flame 1000, facescape 2000/3000, ict 10000, gnm)."""
    num = int(str(subject_id).lower().lstrip("id"))
    if GNM_RANGE[0] <= num < GNM_RANGE[1]:
        return "gnm"
    for offset, name in DOMAIN_OFFSETS:
        if num >= offset:
            return name
    raise ValueError(f"no domain for subject {subject_id!r}")


def split_name(name: str) -> tuple[str, str]:
    """``idNNNN_GTready_<etichetta>.npz`` -> (soggetto, etichetta)."""
    sid, label = name[:-4].split("_GTready_", 1)
    return sid, label


def dist_info() -> tuple[int, int, int]:
    """(rank, world_size, local_rank) dall'ambiente di torchrun; (0, 1, 0) fuori da torchrun."""
    return (int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)),
            int(os.environ.get("LOCAL_RANK", 0)))


def log0(msg: str) -> None:
    """Stampa solo dal rank 0 (con un solo processo: sempre)."""
    if dist_info()[0] == 0:
        print(msg, flush=True)
