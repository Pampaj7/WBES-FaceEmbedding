"""Protocollo adattabile (few-shot): parte comune. Regole in ``aau/runs/evidence/few_shot/REGOLE.md`` (scritte prima
dei numeri); qui solo le costanti che le traducono, le estrazioni e i percorsi.

Domanda: chi ha un dataset proprio dedica k identita' etichettate (GT FR / SR) all'adattamento; con quanti k la nostra
metrica adattata (testa di R6 su encoder congelato) ordina le identita' RESTANTI meglio dei concorrenti, se anche loro
usano le stesse k identita' per scegliere la configurazione?

  * Domini (``DOMAINS``): HIFI3D, FaceScape dev, FaceVerse con espressioni, FaceVerse neutra (i 100 soggetti valutati
    della matrice x 5 topologie senza crop); FaMoS TEST (15 soggetti, 150 scansioni grezze di ``esterna_famos_id``) solo
    per il riconoscimento dell'identita'.
  * Estrazioni (``drawn``): per dominio e ripetizione q una permutazione dei soggetti ordinati,
    ``default_rng(SeedSequence([SEED, j_dominio, q]))``; adattamento = i primi k (annidati in k), test = il resto.
  * Griglia dei concorrenti (``GRID``): le varianti della stessa famiglia gia' nella matrice, nessun parametro numerico
    interno (``aau/robustezza_iperparametri/`` non esiste al 13 ottobre 2026).

Nessun file di Ava-256 letto (i 56 di calibrazione restano alla R6).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
AAU = THIS.parent
REPO = AAU.parent
for _p in (AAU / "diagnostics", AAU / "full_matrix", AAU / "competitors" / "protocolli", AAU / "esterna_famos_id",
           AAU / "famos", AAU / "baselines_mm", AAU / "baselines_param"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

EVID = AAU / "runs" / "evidence" / "few_shot"            # REGOLE.md, risultati, controlli (csv / json / md in git)
WORK = AAU / "runs" / "few_shot"                         # teste, distanze NoW, repliche (npz fuori da git)
HEADS = WORK / "heads"
NOW_WORK = WORK / "now_det"

DOMAINS = ("hifi3d", "facescape", "faceverse", "faceverse_neutral")      # matrice, 100 soggetti
FAMOS = "famos"
DOMAIN_INDEX = {d: k for k, d in enumerate(DOMAINS + (FAMOS,))}         # j_dominio del seme delle estrazioni
NAMES = {"hifi3d": "HIFI3D", "facescape": "FaceScape dev", "faceverse": "FaceVerse con espressioni",
         "faceverse_neutral": "FaceVerse neutra", "famos": "FaMoS TEST (riconoscimento)"}
TARGET = {"hifi3d": "hifi3d", "facescape": "facescape", "faceverse": "faceverse", "faceverse_neutral": "faceverse",
          "famos": "famos"}                                              # generatore escluso dalle sorgenti (LOGO)
N_SUBJ = {d: 100 for d in DOMAINS} | {FAMOS: 15}
KS = (5, 10, 20, 50)
KS_FAMOS = (5, 10)
R_DRAWS = 5
SEED = 20261127
ARMS = ("factorized_s1234", "factorized_s2345")
ARM_NAMES = {"factorized_s1234": "s1234", "factorized_s2345": "s2345"}
R_HEAD, LAM_HEAD = 64, 1e-6                              # configurazione di R6 (d3_e2_heads.csv, joint)
NODE_HEAD = "i256-a10-10"                                # nodo di D3 e2, D5 e R6
NOCROP = ("original", "remesh", "down8k", "up60k", "noisy")
GTS = ("fr", "sr")
N_BOOT = 1000
BOOT_SEED = {"hifi3d": 990708, "facescape": 796786, "faceverse": 271049, "faceverse_neutral": 271049}  # semi T1
FAMOS_BOOT_SEED = 1234                                   # famos_eval.bootstrap_counts(15, 1000, 1234), come ei.py
NOW_TOL_MM = 1e-4                                        # ei.TOL["now_mm"]

# griglia dei concorrenti: famiglia -> [(configurazione, colonna, GT dichiarate)] nell'ordine dei pari
GRID = {
    "icp": [("mm", "mm_rigid_icp_chamfer", GTS), ("cs", "cs_rigid_icp_chamfer", GTS)],
    "now_det": [("metrical", "now_det:now_metrical", GTS), ("originale", "now_det:now_original", GTS)],
    "fit_vb": [(f"{m}|{r}", f"{m}_vb_{r}", g) for m in ("gnm", "flame2023")
               for r, g in (("coef", GTS), ("fr", ("fr",)), ("sr", ("sr",)), ("comp", ("fr",)))],
}
# solo descrittiva (fuori dal k di sorpasso e dal migliore): NoW coi landmark ESATTI della matrice, che un utente non ha
GRID_DESC = {"now_esatti": [("metrical", "plugin:now_metrical|mm", GTS), ("originale", "plugin:now_original|mm", GTS)]}
GRID_ALL = {**GRID, **GRID_DESC}
FAMILY_NAMES = {"icp": "ICP + Chamfer", "now_det": "NoW col rilevatore", "fit_vb": "fit 3DMM vB",
                "now_esatti": "NoW coi landmark esatti (descrittivo)"}
GRID_FAMOS = {
    "icp": [("mm", "icp|mm"), ("cs", "icp|cs")],
    "now_det": [("metrical", "now_metrical|mm"), ("originale", "now_original|mm")],
}
FAMOS_METRICS = ("rank1", "mAP", "AUC", "EER")
FAMOS_MAIN = "mAP"


def configs(family: str, gt: str) -> list[tuple[str, str]]:
    """[(configurazione, colonna)] della famiglia con una lettura dichiarata per la GT."""
    return [(c, col) for c, col, g in GRID_ALL[family] if gt in g]


# ------------------------------------------------------------------------------------------ estrazioni

def ks_of(domain: str) -> tuple[int, ...]:
    return KS_FAMOS if domain == FAMOS else KS


def permutation(domain: str, q: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([SEED, DOMAIN_INDEX[domain], q]))
    return rng.permutation(N_SUBJ[domain])


def drawn(domain: str, k: int, q: int) -> tuple[np.ndarray, np.ndarray]:
    """(adattamento, test): indici dei soggetti ordinati; adattamento = i primi k della permutazione (dominio, q)."""
    p = permutation(domain, q)
    return np.sort(p[:k]), np.sort(p[k:])


def check_draws() -> dict:
    """Estrazioni disgiunte, complete, annidate in k, k esatti."""
    out = {}
    for d in DOMAINS + (FAMOS,):
        for q in range(R_DRAWS):
            prev = np.zeros(0, dtype=int)
            for k in ks_of(d):
                a, t = drawn(d, k, q)
                if len(a) != k or np.intersect1d(a, t).size or len(a) + len(t) != N_SUBJ[d] or not np.isin(prev, a).all():
                    raise SystemExit(f"estrazione non valida: {d} k={k} q={q}")
                prev = a
        out[d] = {"n": N_SUBJ[d], "ks": list(ks_of(d)), "draws": R_DRAWS,
                  "seed": f"SeedSequence([{SEED}, {DOMAIN_INDEX[d]}, q]).permutation({N_SUBJ[d]})"}
    return {"domains": out, "pass": True}


def draws_record(subjects: dict) -> dict:
    """{dominio: {q: soggetti nell'ordine della permutazione}}: i primi k sono l'adattamento di (dominio, k, q)."""
    return {d: {"seed": f"SeedSequence([{SEED}, {DOMAIN_INDEX[d]}, q]).permutation({N_SUBJ[d]})",
                "permutations": {str(q): [subjects[d][i] for i in permutation(d, q)] for q in range(R_DRAWS)}}
            for d in subjects}


# ------------------------------------------------------------------------------------------ teste

def head_path(arm: str, domain: str, k: int, q: int) -> Path:
    """Testa adattata; k = 0: le sole sorgenti (una per braccio e generatore escluso, q = 0)."""
    if k == 0:
        return HEADS / f"{arm}__k0__{TARGET[domain]}.npz"
    return HEADS / f"{arm}__{domain}__k{k:02d}__q{q}.npz"


def head_tasks() -> list[tuple]:
    """(braccio, dominio, k, q) di tutte le teste; k = 0 una per generatore escluso."""
    out = []
    for arm in ARMS:
        for t in ("hifi3d", "facescape", "faceverse", "famos"):
            out.append((arm, t, 0, 0))
        for d in DOMAINS + (FAMOS,):
            out += [(arm, d, k, q) for k in ks_of(d) for q in range(R_DRAWS)]
    return out


# ------------------------------------------------------------------------------------------ uscite

def stamp() -> dict:
    return {"host": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID", "none"),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")}


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default) + "\n")
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
