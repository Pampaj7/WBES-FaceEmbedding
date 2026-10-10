"""Diagnostica D (``aau/runs/evidence/diagnostics/PROTOCOL_D.md``, scritto prima dei numeri): libreria comune.

    insiemi di D1 (``SETS``), nomi delle mesh, righe (coppie di mesh), distanze dei bracci (``fact_paired.distances``,
    copiata), conteggi del bootstrap per soggetto e Spearman pesato (ranghi medi di un campione con righe ripetute
    c_a c_b volte, come ``fact_paired._rep``, senza ripeterle: stesso numero bit per bit a meno dell'arrotondamento).

Uscite in ``aau/runs/evidence/diagnostics`` (npz fuori da git), mesh grezze in ``datasets/DIAG_D1`` e operatori in
``datasets/V3_OPS_CACHE/diag_d1`` (fuori da git, cancellati a fine lavoro).
"""
from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
REPO = THIS.parents[1]
EV = REPO / "aau/runs/evidence/diagnostics"
D1 = EV / "d1"
RAW = REPO / "datasets/DIAG_D1"                              # mesh grezze {V, F} di tutti gli insiemi nuovi
RAW_IN = RAW / "in"
OPS = REPO / "datasets/V3_OPS_CACHE/diag_d1/ops"
TV3 = REPO / "aau/runs/evidence/trainer_v3"
CALIB_EMB = TV3 / "factorized/calib_heldout"                 # embedding held-out della calibrazione c
CALIB_TABLE = REPO / "datasets/V3_OPS_CACHE/heldout_calib/scale_table.json"   # soggetti held-out (fact_calib.stage)
GT_SR = TV3 / "ablations/c3f/gt_sr.npz"                      # GT di training di C3F (fact_calib.GT_SR)
GT_FR = TV3 / "ablations/c3f/gt_frcal.npz"
CALIB_CSV = TV3 / "factorized_calibration.csv"
CALIB_E2 = REPO / "aau/runs/evidence/baselines_param/calib_e2"   # fit B sugli held-out (emendamento 2)
FRAMES = REPO / "aau/runs/evidence/e12/frames.json"

LABELS = ("original", "remesh", "down8k", "noisy", "up60k")   # senza crop
ARMS = ("factorized_s1234", "factorized_s2345", "factorizedc3m_e123")   # chiavi di fact_calib.CKPTS
RULE_ARMS = ARMS[:2]                                          # le regole: entrambi i semi C3F; C3M descrittivo
BP_MODELS = ("gnm", "flame2023")
SEED_ID, SEED_NOISE, SEED_BOOT = 20261111, 20261112, 20261113
SEED_SPLIT, SEED_BBOOT, SEED_PERM = 20261114, 20261115, 20261116
GNM_NOISE_SEED = 20261011                                     # aau/distill/gen_gnm_shard.py SEED_NOISE
N_FLAME, N_BOOT = 200, 1000

# insieme -> gruppo d'identita' (conteggi del bootstrap condivisi), sorgente dello stream, chiave del frame di E12,
# base degli id (None = id degli held-out), suddivisioni 1-a-4 della patch
SETS = {
    "bfm": {"group": 0, "static": True, "frame": "bfm"},
    "ict": {"group": 1, "static": True, "frame": "ict"},
    "gnm": {"group": 2, "static": True, "frame": "gnm"},
    "regen_ict": {"group": 1, "static": False, "source": "ict", "frame": "ict", "id_base": None, "subdiv": 0},
    "regen_gnm": {"group": 2, "static": False, "source": "gnm", "frame": "gnm", "id_base": None, "subdiv": 0},
    "flame2023_s1": {"group": 3, "static": False, "source": "flame2023", "frame": "flame", "id_base": 710000,
                     "subdiv": 1},
    "flame2023": {"group": 3, "static": False, "source": "flame2023", "frame": "flame", "id_base": 700000,
                  "subdiv": 0},
}
SEEN = ("bfm", "ict", "gnm")
NEW_SETS = tuple(s for s, v in SETS.items() if not v["static"])
FLAME_SETS = ("flame2023_s1", "flame2023")


def split_name(name: str) -> tuple[str, str]:
    """``id123_GTready_original.npz`` -> (``id123``, ``original``)."""
    sid, lab = Path(name).name[:-4].split("_GTready_", 1)
    return sid, lab


def heldout_subjects() -> dict:
    """{dominio: soggetti} degli held-out della calibrazione c (scritti da ``fact_calib.stage``)."""
    return json.loads(CALIB_TABLE.read_text())["subjects"]


def set_names(name: str) -> list[str]:
    """Nomi ordinati delle mesh dell'insieme (file ``<sid>_GTready_<etichetta>.npz``)."""
    if SETS[name]["static"] or SETS[name]["id_base"] is None:
        sids = heldout_subjects()[SETS[name]["frame"]]
    else:
        sids = [f"id{SETS[name]['id_base'] + k}" for k in range(N_FLAME)]
    return sorted(f"{s}_GTready_{lab}.npz" for s in sids for lab in LABELS)


def rows(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Coppie (i, j), i < j sui nomi ordinati, soggetti diversi, etichette diverse (``fact_calib.calib_one``)."""
    sid = np.asarray([split_name(n)[0] for n in names])
    lab = np.asarray([split_name(n)[1] for n in names])
    i, j = np.triu_indices(len(names), 1)
    keep = (sid[i] != sid[j]) & (lab[i] != lab[j])
    return i[keep], j[keep]


def subject_index(names: list[str]) -> tuple[np.ndarray, list[str]]:
    """(indice del soggetto per mesh, soggetti nell'ordine dei conteggi). L'ordine e' quello dei soggetti nei nomi
    ordinati: per FLAME l'indice k (``id700000+k`` e ``id710000+k`` hanno lo stesso ordine), per gli held-out e la
    loro rigenerazione gli stessi id."""
    sid = [split_name(n)[0] for n in names]
    subj = sorted(set(sid), key=lambda s: int(s[2:]))
    pos = {s: k for k, s in enumerate(subj)}
    return np.asarray([pos[s] for s in sid]), subj


def boot_counts(n: int, group: int, n_boot: int = N_BOOT, seed: int = SEED_BOOT) -> np.ndarray:
    """(1 + n_boot, n) conteggi per soggetto: riga 0 = tutti 1 (stima puntuale), poi multinomiali
    ``bincount(rng.integers(0, n, n))`` con ``default_rng(SeedSequence([seed, group]))`` (come fact_paired.main)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, group]))
    out = [np.ones(n, dtype=np.int64)]
    out += [np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(n_boot)]
    return np.stack(out)


# ------------------------------------------------------------------------------------------ Spearman pesato

def wranks(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Ranghi medi (base 1) di ``x`` nel campione in cui la riga k compare ``w[k]`` volte (w intero > 0)."""
    o = np.argsort(x, kind="mergesort")
    xs, ws = x[o], w[o].astype(np.float64)
    start = np.r_[True, xs[1:] != xs[:-1]]
    g = np.cumsum(start) - 1                                  # gruppo di pari merito
    gw = np.bincount(g, weights=ws)
    before = np.r_[0.0, np.cumsum(gw)[:-1]]
    r = np.empty(len(x))
    r[o] = (before + (gw + 1.0) / 2.0)[g]
    return r


def wpearson(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    w = w.astype(np.float64)
    ma, mb = (w @ a) / w.sum(), (w @ b) / w.sum()
    da, db = a - ma, b - mb
    den = np.sqrt((w @ (da * da)) * (w @ (db * db)))
    return float((w @ (da * db)) / den) if den > 0 else float("nan")


def wspearman(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Spearman del campione con righe ripetute ``w`` volte (le righe a peso 0 escono)."""
    k = w > 0
    if k.sum() < 3:
        return float("nan")
    x, y, w = x[k], y[k], w[k]
    return wpearson(wranks(x, w), wranks(y, w), w)


def ci(v: np.ndarray) -> tuple[float, float, float, float]:
    """(stima puntuale = replica 0, IC 95% percentile delle repliche 1.., P(<= 0) sulle repliche)."""
    b = v[1:][np.isfinite(v[1:])]
    lo, hi = np.percentile(b, [2.5, 97.5]) if len(b) else (np.nan, np.nan)
    return float(v[0]), float(lo), float(hi), float((b <= 0).mean()) if len(b) else float("nan")


# --------------------------------------------------------------------------------------- distanze dei bracci

def calibration() -> dict:
    """{chiave: c} (``c_median`` di factorized_calibration.csv, emendamento 4)."""
    return {r["key"]: float(r["c_median"]) for r in csv.DictReader(open(CALIB_CSV))}


@lru_cache(maxsize=None)
def dp_per_unit(ckpt: Path) -> float:
    """d_P per unita' di ||u||: json di ``--dist_npz`` / ``gt_scale`` (fact_paired.distances, copiata)."""
    import torch
    args = torch.load(ckpt, map_location="cpu", weights_only=False)["args"]
    if args.get("head", "embed") not in ("factorized", "factorized2"):
        raise SystemExit(f"{ckpt}: testa {args.get('head')}, attesa factorized")
    side = json.loads(Path(args["dist_npz"]).with_suffix(".json").read_text())
    return float(side.get("dp_per_unit", side.get("dP_per_unit"))) / float(args.get("gt_scale", 1.0))


def arm_distances(Z: np.ndarray, i: np.ndarray, j: np.ndarray, dpu: float, c: float) -> dict:
    """d_P (``shape``) e d_F calibrata (``form_cal``) delle righe (i, j): z = [s, u], S = exp(s)."""
    S = np.exp(np.asarray(Z[:, 0], np.float64))
    U = np.asarray(Z[:, 1:], np.float64)
    dP = np.linalg.norm(U[i] - U[j], axis=1) * dpu
    return {"shape": dP, "form_cal": np.sqrt((S[i] - S[j]) ** 2 + S[i] * S[j] * (c * dP) ** 2)}


def load_embeddings(path: Path, names: list[str]) -> tuple[np.ndarray, Path]:
    """(Z nell'ordine di ``names``, checkpoint) da un embeddings.npz di zs_embed (chiave = nome del file)."""
    with np.load(path, allow_pickle=True) as z:
        files = [Path(str(f)).name for f in z["files"]]
        Z = np.asarray(z["Z"], np.float64)
        ckpt = Path(str(z["checkpoint"]))
    pos = {f: k for k, f in enumerate(files)}
    miss = [n for n in names if n not in pos]
    if miss:
        raise SystemExit(f"{path}: mancano {len(miss)} mesh (es. {miss[:2]})")
    return Z[[pos[n] for n in names]], ckpt


def atomic_savez(path: Path, **arrays) -> None:
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def atomic_json(path: Path, obj) -> None:
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)
