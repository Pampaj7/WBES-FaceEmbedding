"""Studio umano v2: percorsi, identita', mesh nel frame di GT-F e frame del dominio (comuni agli script).

Dominio: **GNM Head v3.0** (``~/data/gnm_head/gnm_head.npz``, Apache-2.0 su codice e pesi: render pubblicabili con
attribuzione), 100 identita' campionate dal modello come i set zero-shot (z ~ N(0, 1) su tutti i 170 modi ``head_*``,
senza code ne' troncamento, ``v3_work.mm`` ``sample_identity(tails=False, trunc=0)``, seed 1234), testa intera nel
frame metrico del modello (metri), nessuna normalizzazione. CV della centroid size della regione 5.3% (E12,
``size_cv.csv``: 10.100 identita' GNM; qui sulle 100), contro 1.9% delle REMESH BFM, gia' allineate per
similarita' una per una.

Frame: GT-F di E12 (``aau/runs/evidence/e12/frames.json``: f = u_d R_d V + t_d, una trasformazione per dominio),
poi la rigida ROBUSTA per identita' verso mu (``cgt.Canon.rigid_robust``): e' la GT form di riferimento di E12 (la F
pura misura la posizione del volto nel frame del modello, non osservabile). I render usano la stessa rigida, cosi'
lo stimolo e la GT vedono la stessa forma; nessuna scala per identita'.

Dati derivati (matrici, render PNG) in ``datasets/HUMAN_STUDY_V2/`` (fuori da git); nella pagina vanno solo i JPEG.
"""

from __future__ import annotations

import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "v3_work" / "unified_gt", REPO_ROOT / "v3_work" / "canonical_gt",
           REPO_ROOT / "aau" / "baselines"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DATA_DIR = REPO_ROOT / "datasets" / "HUMAN_STUDY_V2"      # matrici e render (fuori da git)
GT_DIR = DATA_DIR / "gt"
RENDER_DIR = DATA_DIR / "renders"
DOCS_DIR = REPO_ROOT / "docs" / "human_study_v2"
DOMAIN = "gnm"
N_SUBJECTS = 100
SEED = 1234
RENDER_GROUPS = ("hockey_mask", "skin_exterior")         # triangoli renderizzati: la maschera del volto, pelle,
EYE_GROUP = "eye_exteriors"                               # piu' la superficie esterna degli occhi (niente fori neri)
# Codice da cui dipendono le GT: l'impronta finisce nei manifest, cosi' una GT calcolata prima di una
# correzione di E12 si riconosce.
GT_CODE = (REPO_ROOT / "v3_work" / "canonical_gt" / "cgt.py", REPO_ROOT / "v3_work" / "canonical_gt" / "gt.py")


def subjects() -> list[str]:
    return [f"gnm{k:04d}" for k in range(N_SUBJECTS)]


@lru_cache(maxsize=1)
def identity_weights() -> np.ndarray:
    """(N_SUBJECTS, 170) coefficienti standardizzati, campionati come i set zero-shot."""
    from v3_work.mm import load_model
    m = load_model(DOMAIN)
    rng = np.random.default_rng(SEED)
    return np.stack([m.sample_identity(rng, tails=False, trunc=0.0, purpose="eval") for _ in range(N_SUBJECTS)])


@lru_cache(maxsize=1)
def _model() -> dict:
    from cgt import domains
    tpl = domains.gnm()
    z = domains._gnm_raw()
    names = [str(n) for n in z["vertex_group_names"]]
    keep = np.ones(len(tpl["V"]), dtype=bool)
    for g in RENDER_GROUPS:
        keep &= z["vertex_groups"][names.index(g)] > 1e-4
    eye = z["vertex_groups"][names.index(EYE_GROUP)] > 1e-4
    F = tpl["F_render"]
    return {"V": tpl["V"], "basis": tpl["basis"], "F_render": F[keep[F].all(1) | eye[F].all(1)]}


def load_mesh(subject: str) -> tuple[np.ndarray, np.ndarray]:
    """(V della testa intera, metri, frame del modello; triangoli della maschera del volto)."""
    k = subjects().index(subject)
    m = _model()
    return m["V"] + m["basis"] @ identity_weights()[k], m["F_render"]


def face_patch(subject: str) -> np.ndarray:
    """Patch ``hockey_mask`` della pipeline zero-shot (``v3_work.mm``, quad a ventaglio), per la GT maxabs."""
    from v3_work.mm import load_model
    return load_model(DOMAIN).mesh(identity_weights()[subjects().index(subject)])


def sha256_files(paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def canon():
    """(Canon con il frame di GT-F del dominio, info sulla sorgente del frame)."""
    import cgt
    if cgt.FRAMES_JSON.exists():
        cn = cgt.Canon()
        src = {"source": "e12", "path": str(cgt.FRAMES_JSON.relative_to(REPO_ROOT)),
               "sha256": sha256_files([cgt.FRAMES_JSON])}
    else:
        import gt
        cn = cgt.Canon(frames={})
        fr, _, _, _ = gt.frames(cn)
        cn.frames = fr
        src = {"source": "provisional", "path": None,
               "note": "frames.json di E12 assente: frame calcolato con gt.frames (stesso codice), non salvato in e12/"}
    if DOMAIN not in cn.frames:
        raise SystemExit(f"frame di GT-F senza il dominio {DOMAIN}")
    src["frame"] = cn.frames[DOMAIN]
    src["code_sha256"] = sha256_files(GT_CODE)
    # E12 scrive gt.json per ultimo: finche' manca, la selezione e' provvisoria anche col frame di E12
    src["e12_complete"] = (cgt.EVID_DIR / "gt.json").exists()
    return cn, src


def to_mm(cn, V: np.ndarray) -> np.ndarray:
    """Vertici nativi -> frame di GT-F in mm (+y alto, +z naso, frame della media FLAME), senza la rigida."""
    return cn.to_F(DOMAIN, V)


def robust_rigid(cn, subj: list[str]) -> dict:
    """Rigida robusta per identita' (R (n,3,3), t (n,3), diagnostici) dalla regione in mm verso mu."""
    P = np.stack([cn.sp.map(DOMAIN, load_mesh(s)[0]) for s in subj])
    X0 = to_mm(cn, P)
    return {"X0": X0, **cn.rigid_robust(X0)}


def apply_rigid(X: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return X @ R.T + t


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_default) + "\n", encoding="utf-8")


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
