"""Interazione tardiva (stile ColBERT) sulle feature PER VERTICE del nostro encoder: parte comune.

Pilota di ricerca sui domini di sviluppo (HIFI3D, FaceScape dev), senza riaddestrare nulla e senza Ava-256: i due semi
factorized congelati (``SEEDS``: checkpoint e sha256 della tabella della sez. 7.1 del protocollo di Ava-256, passo
21.096, pesi EMA) danno per ogni mesh, oltre al vettore [s, u] del pooling, l'insieme dei token per vertice (f_i, a_i):

  * f_i = feature del vertice i PRIMA del pooling, cioe' l'uscita di ``vertex_bottleneck`` (256 dimensioni):
    ``v3_work/trainer/model_v3.py:92-93`` (``EncoderV3.forward``: ``Z = self.encoder(...)``, ``Z =
    self.vertex_bottleneck(Z)``); il pooling e' la riga 98 (``Z_global = self.pool(Z, mass, w)``), cioe'
    ``FactorizedEncoderV3.pool`` (``v3_work/trainer/factorized_v3.py:58-72``: media pesata con w e max, poi
    ``pool_proj`` per u e ``size_head`` per s);
  * a_i = il peso dello stesso pooling (``area_v3.pool_weights`` in modo ``smooth``, ``model_v3.py:97``: aree della
    geometria passa-basso), normalizzato a somma 1. I vertici con peso 0 (non referenziati) non sono token.

Le regole (varianti, normalizzazione, sottocampione, epsilon, calibrazione, scelta sul pilota) sono in
``aau/runs/evidence/late_interaction/REGOLE.md``, scritte prima dei numeri; qui solo le costanti che le traducono.

Insiemi di mesh (``sets.json`` di ``li_tokens.py stage``, ``heldout.json`` del passo GPU):
  * ``<vista>`` (``hifi3d``, ``facescape``): i 100 soggetti valutati della matrice x le 5 topologie senza crop, con gli
    operatori di sviluppo (la ``data_dir`` di ``eval_key.txt`` degli embedding della matrice, ``fact_paired``);
  * ``pilot_<vista>``: i primi ``PILOT_SUBJECTS`` soggetti NON valutati di ``blmm.template_subjects`` (gli stessi del
    pilota di GW e percettivi) x le 5 topologie senza crop, operatori areanorm k_eig 128 calcolati qui;
  * ``heldout``: ``HELDOUT_SUBJECTS`` soggetti per generatore (bfm, ict, gnm; ``rng(1234)`` sugli held-out di
    ``fact_calib.heldout_subjects``) x le 5 etichette, operatori della calibrazione di c (``heldout_calib``).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
AAU = THIS.parent
REPO = AAU.parent
TRAINER = REPO / "v3_work" / "trainer"
OUT = AAU / "runs" / "evidence" / "late_interaction"      # REGOLE.md, risultati, plug-in (npz fuori da git)
WORK = AAU / "runs" / "late_interaction"                   # distanze, meta per mesh, tempi (fuori da git)
OPS_ROOT = REPO / "datasets" / "V3_OPS_CACHE" / "late_interaction"
SCALE_TABLES = AAU / "runs" / "evidence" / "trainer_v3" / "factorized" / "scale_tables"
HELDOUT_OPS = REPO / "datasets" / "V3_OPS_CACHE" / "heldout_calib"

_RUNS = "aau/runs/evidence/trainer_v3/ablations/c3f_runs/"
_TAG = ("v3_mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_poolmeanmax_noise60_sampler-balanced_pooling-area_"
        "meanmax_inputnorm-global_areaweights-smooth_emadecay-0.999_head-factorized_seed")
SEEDS = {"s1234": {"checkpoint": f"{_RUNS}factorized/{_TAG}1234__1f5d4a90/checkpoints/epoch072_ema.pth",
                   "sha256": "317532da3866e3be377bc57e26155c728db5760c61d1bfc23029b80f1a8ac84c",
                   "store_variant": "factorized"},
         "s2345": {"checkpoint": f"{_RUNS}factorized_s2345/{_TAG}2345__4de9be29/checkpoints/epoch072_ema.pth",
                   "sha256": "5489a97e93c9de25b184e69a27f68980ba61429a3ae0ff66030688d6acb690b0",
                   "store_variant": "factorizeds2345"}}
VIEWS = ("hifi3d", "facescape")
FP_DOM = {"hifi3d": "hifi", "facescape": "devfs"}           # fact_paired.FORM_DIR
TABLE = {"hifi3d": "hifi3d_eval.npz", "facescape": "devfs_eval.npz"}
NOCROP = ("original", "remesh", "down8k", "up60k", "noisy")  # fm_eval.NOCROP
PILOT_SUBJECTS = 12                                         # gp_common.PILOT_SUBJECTS
HELDOUT_SUBJECTS, HELDOUT_SEED = 12, 1234                   # per generatore (bfm, ict, gnm)
HELDOUT_LABELS = ("down8k", "noisy", "original", "remesh", "up60k")   # fact_calib.LABELS

# varianti (REGOLE.md, sez. 2): nome -> (token, accoppiamento)
VARIANTS = {"chamfer": ("tutti i vertici", "Chamfer simmetrico pesato per area"),
            "chamfer_sub": ("sottocampione FPS", "Chamfer simmetrico pesato per area"),
            "sinkhorn": ("sottocampione FPS", "OT entropico (Sinkhorn)")}
N_SUB = 1000                                                # sottocampione FPS per mesh
EPS_REL = 0.05                                              # epsilon = EPS_REL x s0 (s0: REGOLE.md)
SK_TOL, SK_MAX_ITER, SK_CHECK = 1e-5, 3000, 10              # errore L1 sui marginali, iterazioni, ogni quante
TIE = 0.01                                                  # pilota: entro TIE dalla migliore vince la piu' economica
LABEL = {"chamfer": "interazione tardiva, Chamfer su tutti i vertici",
         "chamfer_sub": "interazione tardiva, Chamfer sul sottocampione",
         "sinkhorn": "interazione tardiva, OT entropico sul sottocampione"}
MODES = ("fr", "sr")
MODE_LABEL = {"fr": "ricomposta con S e c' (FR, post hoc)", "sr": "d_LI (SR)"}


# ------------------------------------------------------------------------------------------- percorsi

def checkpoint(seed: str) -> Path:
    return REPO / SEEDS[seed]["checkpoint"]


def pilot_ops(view: str) -> Path:
    return OPS_ROOT / f"pilot_{view}"


def dist_path(seed: str, name: str) -> Path:
    """Distanze di un insieme per un seme (``li_dist``): ``WORK/dist/<seme>/<insieme>.npz``."""
    return WORK / "dist" / seed / f"{name}.npz"


def meta_path(seed: str, name: str) -> Path:
    """Per mesh: [s, u] del pooling, numero di token, indici e masse del sottocampione (``li_tokens``)."""
    return WORK / "meta" / seed / f"{name}.npz"


def plugin_dir(seed: str, variant: str) -> Path:
    """Cartella del plug-in (``fm_eval.plugin_dirs`` con radice ``OUT``): ``OUT/factorized_<seme>/<variante>/``."""
    return OUT / f"factorized_{seed}" / variant


def load_sets() -> dict:
    return json.loads((WORK / "sets.json").read_text())


def pair_index(subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coppie non ordinate i < j di soggetti diversi (stessa topologia compresa): le righe senza crop della matrice."""
    i, j = np.triu_indices(len(subjects), 1)
    keep = subjects[i] != subjects[j]
    return i[keep], j[keep]


# ------------------------------------------------------------------------------------------- uscite

def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def stamp() -> dict:
    return {"host": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID", "none"),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")}


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
