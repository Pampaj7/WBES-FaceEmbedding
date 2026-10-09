"""Studio umano v2: percorsi, soggetti, mesh nel frame di GT-F e frame del dominio (comuni agli script).

Dominio: BFM REMESH, topologia ``original``, i 100 soggetti held-out della v1 (``common.subject_set``), nel
frame e nelle unita' di GT-F di E12 (``v3_work/canonical_gt``): f = u_d R_d V + t_d, UNA trasformazione per
dominio, nessuna per mesh. Il frame viene da ``aau/runs/evidence/e12/frames.json`` se E12 l'ha scritto
(sorgente ``e12``), altrimenti si calcola qui con lo stesso codice (``gt.frames``, sorgente ``provisional``).

Dati derivati dai modelli con licenza (matrici, render PNG) in ``datasets/HUMAN_STUDY_V2/`` (fuori da git);
nella pagina vanno solo i JPEG dei volti sintetici, come nella v1.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
for _p in (REPO_ROOT / "v3_work" / "unified_gt", REPO_ROOT / "v3_work" / "canonical_gt",
           REPO_ROOT / "aau" / "baselines"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DATA_DIR = REPO_ROOT / "datasets" / "HUMAN_STUDY_V2"      # matrici e render (fuori da git)
GT_DIR = DATA_DIR / "gt"
RENDER_DIR = DATA_DIR / "renders"
DOCS_DIR = REPO_ROOT / "docs" / "human_study_v2"
DOMAIN = "bfm"
TOPOLOGY = "original"
SUBJECT_SET = "heldout"
# Codice da cui dipendono le GT: l'impronta finisce nei manifest, cosi' una GT calcolata prima di una
# correzione di E12 si riconosce.
GT_CODE = (REPO_ROOT / "v3_work" / "canonical_gt" / "cgt.py", REPO_ROOT / "v3_work" / "canonical_gt" / "gt.py")


def subjects() -> list[str]:
    import common
    return common.subject_set(SUBJECT_SET)


def load_mesh(subject: str) -> tuple[np.ndarray, np.ndarray]:
    """(V nativi della REMESH, unita' BFM; F) della topologia ``original``."""
    import common
    return common.load_verts_faces(subject, TOPOLOGY)


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
    """Vertici nativi -> frame di GT-F in mm (+y alto, +z naso, frame della media FLAME)."""
    return cn.to_F(DOMAIN, V)


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
