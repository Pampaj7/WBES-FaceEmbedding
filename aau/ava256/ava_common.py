"""Percorsi, catture e lettura dei file Ava-256 (Martinez et al., Codec Avatar Studio, NeurIPS D&B 2024).

CONFERMATIVO: non valutare prima del protocollo confermativo (``aau/ava256/README.md``,
``aau/ava256/PROTOCOL_CONFERMATIVO_bozza.md``). Nessun modello, baseline o metrica gira su questi dati.

LICENZA: CC BY-NC 4.0 (Meta). Nessun dato nel repo: grezzi, neutre, GT e viste stanno in ``datasets/AVA256``
(in .gitignore); nel repo solo codice, manifest (nomi e hash) e numeri aggregati senza geometria.

Formati (verificati sui file scaricati):
  - registrazioni ``raw/<cattura>/registration_vertices/<frame:06d>.ply``: PLY binario little endian, 7306 vertici
    float32 xyz, nessuna faccia (87.790 byte: header di 118 + 7306 x 12), un file per frame;
  - topologia ``meta/face_topology.obj`` (``assets/`` del repo di Meta): 7306 v, 5779 vt, 11.432 triangoli
    ``f v/vt``, la stessa per tutte le catture; i vertici non usati da nessun triangolo non fanno parte della superficie.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
REPO_ROOT = AAU_DIR.parent

DATA_ROOT = REPO_ROOT / "datasets" / "AVA256"
META_DIR = DATA_ROOT / "meta"
RAW_DIR = DATA_ROOT / "raw"
IDS_CSV = META_DIR / "256_ids.csv"
TOPOLOGY_OBJ = META_DIR / "face_topology.obj"
NEUTRAL_DIR = DATA_ROOT / "neutral"          # neutre per cattura (ava_neutral.py)
GT_DIR = DATA_ROOT / "gt"                    # GT FR / SR (ava_gt.py), FUORI da datasets/CANONICAL_GT/eval
EVID_DIR = DATA_ROOT / "evidence"            # diagnostici con geometria o render (mai in git)
SUMMARY_DIR = THIS_DIR                       # numeri aggregati in git (json piccoli, senza geometria)

NEUTRAL_SEG = "EXP_neutral_peak"
REPEAT_SEG = "EXP_eye_neutral"
N_VERTS = 7306
N_FACES = 11432
# id<offset + NNNN> come le viste zero-shot: fuori da 900000 HIFI3D, 910000 FaceVerse, 920000 held-out GNM/ICT,
# 930000 FaceScape dev, 940000 FaMoS
ID_OFFSET = 950000


def captures() -> list[dict]:
    """Le 256 catture nell'ordine di ``256_ids.csv``: ``ava_id`` (avaNNNN, NNNN = riga), ``view_id``, ``capture``."""
    rows = list(csv.DictReader(IDS_CSV.read_text().splitlines()))
    out = []
    for k, r in enumerate(rows):
        out.append({"ava_id": f"ava{k:04d}", "view_id": f"id{ID_OFFSET + k:06d}", "sid": r["sid"],
                    "capture": f"{r['mcd']}--{r['mct']}--{r['sid']}"})
    if len({c["sid"] for c in out}) != len(out):
        raise ValueError(f"{IDS_CSV}: sid ripetuti")
    return out


def index(capture: str, raw_dir: Path = RAW_DIR) -> dict:
    return json.loads((raw_dir / capture / "index.json").read_text())


def frames(capture: str, segment: str, raw_dir: Path = RAW_DIR) -> list[tuple[int, Path]]:
    """(frame, percorso) dei PLY scaricati di un segmento, in ordine di frame."""
    idx = index(capture, raw_dir)
    out = [(int(Path(f["file"]).stem), raw_dir / capture / f["file"]) for f in idx["files"]
           if f.get("segment") == segment]
    return sorted(out)


def read_ply(path: Path) -> np.ndarray:
    """Vertici (7306, 3) float64 di un PLY di registrazione (header controllato)."""
    b = Path(path).read_bytes()
    end = b.index(b"end_header\n") + len(b"end_header\n")
    head = b[:end].decode("ascii").split("\n")
    if head[1] != "format binary_little_endian 1.0" or f"element vertex {N_VERTS}" not in head or \
            [h for h in head if h.startswith("property")] != ["property float x", "property float y", "property float z"]:
        raise ValueError(f"{path}: header PLY inatteso {head}")
    if len(b) - end != N_VERTS * 12:
        raise ValueError(f"{path}: {len(b) - end} byte di vertici invece di {N_VERTS * 12}")
    return np.frombuffer(b[end:], dtype="<f4").reshape(N_VERTS, 3).astype(np.float64)


def topology() -> np.ndarray:
    """Triangoli (11432, 3) int64, base 0, dagli indici di vertice delle righe ``f v/vt v/vt v/vt``."""
    F, nv = [], 0
    for line in TOPOLOGY_OBJ.read_text().splitlines():
        if line.startswith("v "):
            nv += 1
        elif line.startswith("f "):
            c = line.split()[1:]
            if len(c) != 3:
                raise ValueError(f"{TOPOLOGY_OBJ}: faccia non triangolare")
            F.append([int(x.split("/")[0]) - 1 for x in c])
    F = np.asarray(F, dtype=np.int64)
    if nv != N_VERTS or len(F) != N_FACES or F.min() < 0 or F.max() >= N_VERTS:
        raise ValueError(f"{TOPOLOGY_OBJ}: {nv} vertici, {len(F)} triangoli")
    return F


def topology_template() -> np.ndarray:
    """I vertici (7306, 3) dell'obj di Meta (una forma di riferimento della topologia, NON un soggetto del set)."""
    V = [[float(x) for x in line.split()[1:4]] for line in TOPOLOGY_OBJ.read_text().splitlines()
         if line.startswith("v ")]
    return np.asarray(V, dtype=np.float64)
