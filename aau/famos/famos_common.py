"""Percorsi, split e lettura dei file FaMoS (Bolkart et al., TEMPEH, CVPR 2023).

LICENZA: FaMoS e' MPI, solo ricerca non commerciale. Nessun dato FaMoS nel repo: i grezzi stanno in
``external_data/famos`` e tutto quello che ne deriva (fotogrammi sottocampionati, patch, operatori,
embedding) in ``datasets/FAMOS``; entrambe le cartelle sono in .gitignore. Nel repo (``aau/famos``,
``aau/runs/evidence/famos``) solo codice, lo split (nomi dei soggetti) e numeri aggregati.

Formati dei grezzi (verificati su disco):
  - registrazioni ``registrations/FaMoS_subject_NNN/<sequenza>/<sequenza>.NNNNNN.ply``: ply binario
    little endian, 5023 vertici float32 xyz e 9976 triangoli (lista uchar + 3 int32), cioe' la
    topologia FLAME 2020, a 60 fps;
  - scansioni di test ``scans/FaMoS_subject_NNN/<sequenza>/<sequenza>.NNNNNN.obj``: obj testuali da
    ~110k vertici in mm, un sottoinsieme di ~20 fotogrammi per sequenza, ognuno con la registrazione
    dello stesso fotogramma (stesso nome).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
REPO_ROOT = AAU_DIR.parent

RAW_ROOT = REPO_ROOT / "external_data" / "famos" / "extracted"
REG_DIR = RAW_ROOT / "registrations" / "registrations"
SCAN_DIR = RAW_ROOT / "test_scans" / "scans"
SPLIT_JSON = THIS_DIR / "split.json"
OUT_ROOT = REPO_ROOT / "datasets" / "FAMOS"
EVID_DIR = AAU_DIR / "runs" / "evidence" / "famos"

# Fotogrammi sottocampionati e forma neutra: datasets/FAMOS/{train,test}/<soggetto>.npz
TRAIN_DIR = OUT_ROOT / "train"
TEST_DIR = OUT_ROOT / "test"
FLAME_FACES = OUT_ROOT / "flame_faces.npy"
# Set di test reale (formato delle viste di aau/zs3dmm): datasets/FAMOS/test_view
VIEW_DIR = OUT_ROOT / "test_view"
# id<offset + NNN> come le viste zero-shot: fuori da 900000 HIFI3D, 910000 FaceVerse, 920000 GNM/ICT
# held-out, 930000 FaceScape dev
ID_OFFSET = 940000

# Embedding dei 68 landmark iBUG su FLAME 2020 (formato DECA/MICA: full_lmk_faces_idx / bary_coords)
FLAME_LMK_EMBEDDING = REPO_ROOT / "external" / "MICA" / "data" / "FLAME2020" / "landmark_embedding.npy"

SUBJ_RE = re.compile(r"^FaMoS_subject_(\d{3})$")
FRAME_RE = re.compile(r"^(?P<seq>.+)\.(?P<frame>\d{6})\.(?:ply|obj)$")
N_VERTS = 5023
N_FACES = 9976


def load_split() -> dict:
    s = json.loads(SPLIT_JSON.read_text())
    assert not set(s["train"]) & set(s["test"]), "split: TRAIN e TEST non disgiunti"
    return s


def subject_num(subject: str) -> int:
    m = SUBJ_RE.match(subject)
    if not m:
        raise ValueError(f"nome di soggetto inatteso: {subject}")
    return int(m.group(1))


def view_id(subject: str) -> str:
    """``FaMoS_subject_079`` -> ``id940079`` (il soggetto delle viste di eval)."""
    return f"id{ID_OFFSET + subject_num(subject):06d}"


def frames_of(seq_dir: Path, ext: str) -> list[tuple[int, Path]]:
    """(numero del fotogramma, file) della sequenza, in ordine di fotogramma."""
    out = []
    for p in seq_dir.iterdir():
        m = FRAME_RE.match(p.name)
        if m and p.suffix == f".{ext}" and m["seq"] == seq_dir.name:
            out.append((int(m["frame"]), p))
    return sorted(out)


def reg_path(subject: str, seq: str, frame: int) -> Path:
    return REG_DIR / subject / seq / f"{seq}.{frame:06d}.ply"


def scan_path(subject: str, seq: str, frame: int) -> Path:
    return SCAN_DIR / subject / seq / f"{seq}.{frame:06d}.obj"


# ------------------------------------------------------------------------------ ply FLAME

_HEADER = None


def _ply_header(raw: bytes) -> int:
    """Offset dei dati; controlla che l'header sia quello atteso (5023 vertici float, 9976 facce)."""
    end = raw.find(b"end_header\n")
    if end < 0:
        raise ValueError("ply senza end_header")
    head = raw[:end].decode("ascii")
    if ("binary_little_endian" not in head or f"element vertex {N_VERTS}" not in head
            or f"element face {N_FACES}" not in head or "property float x" not in head):
        raise ValueError(f"header ply inatteso:\n{head}")
    return end + len("end_header\n")


def read_reg(path: Path, with_faces: bool = False):
    """Vertici (5023, 3) float32 di una registrazione; con ``with_faces`` anche F (9976, 3) int32."""
    raw = Path(path).read_bytes()
    off = _ply_header(raw)
    V = np.frombuffer(raw, dtype="<f4", count=3 * N_VERTS, offset=off).reshape(N_VERTS, 3).copy()
    if not with_faces:
        return V
    rec = np.frombuffer(raw, dtype=np.dtype([("n", "u1"), ("i", "<i4", (3,))]), count=N_FACES,
                        offset=off + 12 * N_VERTS)
    if not (rec["n"] == 3).all():
        raise ValueError(f"{path}: facce non triangolari")
    return V, rec["i"].astype(np.int32)


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(V float64, F int64) di una scansione: solo righe ``v`` e ``f`` (indici di vertice prima di '/')."""
    V, F = [], []
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("v "):
                V.append(line[2:].split()[:3])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) for t in line[2:].split()]
                for k in range(1, len(idx) - 1):          # ventaglio, se ci fossero poligoni
                    F.append((idx[0], idx[k], idx[k + 1]))
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    F = np.where(F > 0, F - 1, F + len(V))                # obj: 1-based, negativi relativi
    return V, F


# ------------------------------------------------------------------------------ landmark

def flame_lmk68() -> tuple[np.ndarray, np.ndarray]:
    """(facce (68,), baricentriche (68, 3)) dei 68 landmark iBUG sulla topologia FLAME 2020."""
    d = np.load(FLAME_LMK_EMBEDDING, allow_pickle=True, encoding="latin1")[()]
    fi = np.asarray(d["full_lmk_faces_idx"]).reshape(-1).astype(np.int64)
    bc = np.asarray(d["full_lmk_bary_coords"]).reshape(-1, 3).astype(np.float64)
    if len(fi) != 68:
        raise ValueError(f"{FLAME_LMK_EMBEDDING}: attesi 68 landmark, trovati {len(fi)}")
    return fi, bc


def landmarks(V: np.ndarray, F: np.ndarray, fi: np.ndarray, bc: np.ndarray, which) -> np.ndarray:
    """Landmark ``which`` (indici iBUG) sulla mesh FLAME (V, F)."""
    tri = np.asarray(V, dtype=np.float64)[np.asarray(F)[fi[list(which)]]]     # (k, 3, 3)
    return np.einsum("kvd,kv->kd", tri, bc[list(which)])
