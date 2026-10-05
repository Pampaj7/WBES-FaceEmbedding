"""Percorsi, I/O e controlli condivisi dai tre runner di ricostruzione 3D da immagine (WS3b).

Ogni metodo produce, per ogni immagine, due file nella stessa directory di output:

    <stem>.npz     V (n, 3) float32 e F (m, 3) int32, topologia FISSA del metodo
    <stem>.json    posa stimata, box del volto, 68 landmark, tempi, unita' e assi

Il sistema di coordinate e' quello nativo del metodo e per tutti e tre e' lo **spazio
immagine in pixel**: x a destra, y in giu' (riga dell'immagine), z profondita' nella
stessa scala di x/y.  Nessuno dei tre restituisce millimetri: da una singola foto la
scala metrica non e' osservabile, e infatti i protocolli tipo NoW allineano con una
similarita' prima di misurare.  Il json porta anche i 62 parametri 3DMM (3DDFA_V2,
SynergyNet), da cui si ricava la forma canonica nelle unita' del BFM.

Le mesh non vengono ne' centrate ne' riscalate: qualsiasi normalizzazione e' una scelta
del protocollo di valutazione, ed e' esattamente la variabile che WS3b vuole studiare.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
REPO_ROOT = AAU_DIR.parent
EXTERNAL = REPO_ROOT / "external"

# I tre cloni e il venv in cui gira ciascuno (vedi aau/recon/README.md).
METHOD_REPO = {
    "3ddfa_v2": EXTERNAL / "3DDFA_V2",
    "synergynet": EXTERNAL / "SynergyNet",
    "prnet": EXTERNAL / "PRNet",
}
METHOD_VENV = {
    "3ddfa_v2": "ddfa",
    "synergynet": "ddfa",
    "prnet": "prnet",
}

OUT_ROOT = Path(os.environ.get("WBES_RECON_OUT", AAU_DIR / "runs" / "recon"))

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

# Descrizione degli assi, copiata identica nei json: e' l'unica cosa che rende
# confrontabili tra loro output di metodi diversi.
COORD_SYSTEM = "image pixels: x right, y down (image row), z depth in the same scale"


def patch_numpy_aliases() -> None:
    """Rimette np.int / np.long / np.float, che numpy >= 1.24 ha tolto.

    Tutti e tre i repo sono del 2018-2021 e li usano ancora: ``bfm/bfm.py`` di 3DDFA_V2
    (np.long), la NMS cython di FaceBoxes (np.int, dentro il .pyx, quindi non basta
    guardare i .py) e ``utils/cv_plot.py`` di PRNet (np.float).  Erano alias esatti dei
    tipi nativi di python, quindi rimetterli non cambia nessun risultato.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for name, alias in (("int", int), ("float", float), ("bool", bool), ("long", int)):
            if not hasattr(np, name):
                setattr(np, name, alias)


def list_images(d: Path) -> list[Path]:
    """Le immagini della directory, in ordine stabile."""
    if not d.is_dir():
        raise SystemExit(f"ERRORE: directory di input inesistente: {d}")
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise SystemExit(f"ERRORE: nessuna immagine {IMAGE_SUFFIXES} in {d}")
    return files


def add_repo_to_path(method: str, front: bool = True) -> Path:
    """Mette la radice del clone su sys.path e ne restituisce il percorso.

    I tre repo hanno moduli top-level omonimi (``utils``, ``models``), quindi un processo
    solo puo' avere un solo metodo davanti: ``front=False`` serve a PRNet, che importa
    FaceBoxes da 3DDFA_V2 tenendo pero' il proprio ``utils``.
    """
    repo = METHOD_REPO[method]
    if not repo.is_dir():
        raise SystemExit(f"ERRORE: clone mancante: {repo}\n  Vedi aau/recon/README.md")
    p = str(repo)
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p) if front else sys.path.append(p)
    return repo


def faceboxes_detector():
    """FaceBoxes di 3DDFA_V2 (pesi nel clone), gia' costruito.

    E' il detector che usano sia 3DDFA_V2 sia SynergyNet nei loro demo.  PRNet nel suo
    demo usa dlib, che non e' installabile qui senza cmake: gli si passa lo stesso
    detector, il crop invece resta quello di PRNet.
    """
    add_repo_to_path("3ddfa_v2", front=False)
    from FaceBoxes import FaceBoxes  # noqa: E402

    return FaceBoxes()


def pick_box(boxes) -> tuple[list[float], int]:
    """Il box piu' grande in area, piu' quanti ne sono stati trovati.

    Le immagini di WS3b sono ritratti frontali con un volto; se il detector ne trova piu'
    di uno (uno sfondo, una folla) si tiene il volto principale, e il json ne tiene conto
    con ``n_faces``.
    """
    boxes = [np.asarray(b, dtype=np.float64) for b in boxes]
    if not boxes:
        return [], 0
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    return list(boxes[int(np.argmax(areas))]), len(boxes)


def mesh_stats(V: np.ndarray, F: np.ndarray) -> dict:
    """Statistiche di validita' di una mesh: usate da check_meshes.py e scritte nel json."""
    V = np.asarray(V)
    F = np.asarray(F)
    finite = bool(np.isfinite(V).all())
    bbox_min = V.min(axis=0) if finite else np.full(3, np.nan)
    bbox_max = V.max(axis=0) if finite else np.full(3, np.nan)
    return {
        "n_vertices": int(V.shape[0]),
        "n_faces": int(F.shape[0]),
        "finite": finite,
        "bbox_min": [float(x) for x in bbox_min],
        "bbox_max": [float(x) for x in bbox_max],
        "bbox_extent": [float(x) for x in (bbox_max - bbox_min)],
        "faces_in_range": bool(F.min() >= 0 and F.max() < V.shape[0]),
    }


def save_mesh(outdir: Path, stem: str, V: np.ndarray, F: np.ndarray) -> Path:
    """Scrive <stem>.npz con le chiavi V (float32) e F (int32), come il resto del repo."""
    V = np.ascontiguousarray(V, dtype=np.float32)
    F = np.ascontiguousarray(F, dtype=np.int32)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"V deve essere (n, 3), non {V.shape}")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"F deve essere (m, 3), non {F.shape}")
    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / f"{stem}.npz"
    np.savez(fp, V=V, F=F)
    return fp


def save_meta(outdir: Path, stem: str, meta: dict) -> Path:
    """Scrive <stem>.json (posa, box, landmark, tempi)."""
    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / f"{stem}.json"
    with open(fp, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return fp


def to_list(a) -> list:
    """numpy -> liste json-serializzabili, senza perdere la forma."""
    return np.asarray(a, dtype=np.float64).tolist()
