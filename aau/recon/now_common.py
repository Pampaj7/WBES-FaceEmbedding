"""Percorsi, nomi e geometria condivisa per la valutazione su NoW validation.

Il gemello di ``ws3b_common.py`` per NoW (protocollo in ``aau/runs/now_eval/protocol.md``).
Un elemento e' un'immagine di ``imagepathsvalidation.txt``,

    <soggetto>/<sfida>/<IMG>.jpg   ->   nome  <soggetto>__<sfida>__<IMG>

e la sua verita' a terra e' la scansione del soggetto (una per soggetto, 20 in tutto).
I runner chiamano l'uscita come lo stem dell'immagine, e ``now_recon.sbatch`` espone le
immagini come symlink gia' battezzati: il nome della mesh e' gia' quello dell'elemento.

Licenza NoW: ricerca non commerciale, niente redistribuzione.  Per questo tutto quello che
viene dai dati -- mesh, render, operatori, l'export nel layout ufficiale -- sta fuori dal
repo (``NOW_DIR``, ``RECON_ROOT``, ``WORK_ROOT``); nel repo (``OUT_ROOT``) vanno solo i csv
di numeri e i riassunti.

Le convenzioni
--------------
**Mano.**  Come in WS3b: i metodi restituiscono lo spazio immagine (y in giu') e la y si
nega prima di tutto.  Le scansioni sono gia' destrorse: mm, x a destra, y in su, z in avanti.

**Landmark.**  I 7 punti NoW, nell'ordine di ``scans_lmks_onlypp``: ex_r, en_r, en_l, ex_l,
subnasale, ch_r, ch_l.  Sulla ricostruzione sono gli iBUG 36, 39, 42, 45, 33, 48, 54: il
quinto e' il 33 (subnasale) e NON il 30 (punta) di ``ws3b_common.LMK7_IBUG``, perche'
``compute_mask`` del codice ufficiale lo chiama "nose bottom" e nelle scansioni sta ~48 mm
sotto la linea degli occhi.  E' anche l'indice di DECA e MICA per NoW.

**Frame canonico e ritaglio.**  Ogni mesh -- scansione o ricostruzione -- va nel frame del
template T7 con la similarita' che porta i SUOI landmark su T7, poi si ritaglia con la
regola di ``compute_mask`` calcolata sui suoi landmark.  Cosi' la ricostruzione non vede
mai la scansione del proprio soggetto, e le due regioni sono definite dalla stessa regola.
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
REPO_ROOT = AAU_DIR.parent
HOME_DATA = Path(os.environ.get("HOME", "/home/create.aau.dk/ga41wf")) / "data"

NOW_DIR = Path(os.environ.get("WBES_NOW_DIR", HOME_DATA / "now"))
RECON_ROOT = Path(os.environ.get("WBES_NOW_RECON", HOME_DATA / "now_recon"))
WORK_ROOT = Path(os.environ.get("WBES_NOW_WORK", HOME_DATA / "now_eval_work"))
OUT_ROOT = Path(os.environ.get("WBES_NOW_OUT", AAU_DIR / "runs" / "now_eval"))

NOW_EVAL_REPO = REPO_ROOT / "external" / "now_evaluation"
IMAGE_LIST = NOW_DIR / "imagepathsvalidation.txt"
SCANS_DIR = NOW_DIR / "scans"
SCAN_LMKS_DIR = NOW_DIR / "scans_lmks_onlypp"

METHODS = ("3ddfa_v2", "synergynet", "prnet", "mica")
CHALLENGES = ("multiview_neutral", "multiview_expressions", "multiview_occlusions", "selfie")

LMK7_NAMES = ("ex_r", "en_r", "en_l", "ex_l", "subnasale", "ch_r", "ch_l")
LMK7_IBUG = (36, 39, 42, 45, 33, 48, 54)

# Risoluzione comune delle patch: i triangoli di ``gt_face`` in WS3b (face_region.json).
TARGET_FACES = 5215

NAME_SEP = "__"


@dataclass(frozen=True)
class Item:
    """Un'immagine di NoW validation, quindi una ricostruzione per metodo."""

    name: str
    subject: str
    challenge: str
    image: str      # relativo a iphone_pictures, come nella lista ufficiale


def item_name(subject: str, challenge: str, image: str) -> str:
    return NAME_SEP.join((subject, challenge, Path(image).stem))


def load_items(path: Path = IMAGE_LIST) -> list[Item]:
    """Le 352 immagini, nell'ordine della lista ufficiale."""
    if not path.is_file():
        raise SystemExit(f"lista immagini assente: {path}")
    items = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        subject, challenge, image = line.split("/")
        if challenge not in CHALLENGES:
            raise ValueError(f"sfida inattesa in {line}")
        items.append(Item(item_name(subject, challenge, image), subject, challenge, line))
    return items


def subjects_of(items: list[Item]) -> list[str]:
    return sorted({it.subject for it in items})


# ------------------------------------------------------------------------- percorsi

def recon_dir(method: str) -> Path:
    """Mesh come escono dal metodo (pixel, y in giu'), piu' il json coi 68 landmark."""
    return RECON_ROOT / method


def scan_paths(subject: str) -> tuple[Path, Path]:
    """(obj della scansione, .pp dei 7 landmark): uno per soggetto, nome variabile."""
    objs = sorted((SCANS_DIR / subject).glob("*.obj"))
    pps = sorted((SCAN_LMKS_DIR / subject).glob("*.pp"))
    if len(objs) != 1 or len(pps) != 1:
        raise SystemExit(f"{subject}: attesi un obj e un pp, trovati {len(objs)} e {len(pps)}")
    return objs[0], pps[0]


def pred_dir(method: str) -> Path:
    """Export nel layout ufficiale: <soggetto>/<sfida>/<IMG>.obj + .npy (7 landmark)."""
    return WORK_ROOT / "pred" / method


def template_path() -> Path:
    return WORK_ROOT / "template_lmk7.json"


def scan_face_dir() -> Path:
    return WORK_ROOT / "scan_face"


def recon_face_dir(method: str) -> Path:
    return WORK_ROOT / "recon_face" / method


OPS_SUFFIX = "_withops_areanorm"


def scan_face_ops_dir() -> Path:
    """Operatori ad area unitaria (k_eig 128) di scan_face, per il modello congiunto."""
    return WORK_ROOT / f"scan_face{OPS_SUFFIX}"


def recon_face_ops_dir(method: str) -> Path:
    return WORK_ROOT / f"recon_face{OPS_SUFFIX}" / method


def official_csv_path(method: str) -> Path:
    return OUT_ROOT / f"now_official_{method}.csv"


def gt_csv_path(metric: str, method: str) -> Path:
    """Una riga per ricostruzione, contro la scansione del soggetto."""
    return OUT_ROOT / f"gt_{metric}_{method}.csv"


def pair_matrix_path(metric: str, method: str) -> Path:
    """Matrice (n, n) delle distanze fra le ricostruzioni di un metodo: fuori dal repo."""
    return WORK_ROOT / "pairs" / f"{metric}_{method}.npz"


# ------------------------------------------------------------------------- I/O

def load_pp(path: Path):
    """I 7 landmark di un .pp di MeshLab, nell'ordine del file (attributo name 0..6)."""
    import numpy as np

    pts = {}
    for m in re.finditer(r'<point\s+([^>]*)/>', path.read_text()):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        pts[int(attrs["name"])] = [float(attrs["x"]), float(attrs["y"]), float(attrs["z"])]
    if sorted(pts) != list(range(7)):
        raise ValueError(f"{path}: attesi i punti 0..6, trovati {sorted(pts)}")
    return np.asarray([pts[k] for k in range(7)], dtype=np.float64)


def load_recon(method: str, name: str):
    """(V destrorsi, F, 7 landmark destrorsi) di una ricostruzione, nelle unita' del metodo.

    La y si nega solo per le mesh in spazio immagine (``coordinate_system`` dei runner di
    WS3b, "image pixels: ..."); MICA restituisce la forma FLAME canonica, gia' destrorsa.
    """
    import numpy as np

    d = recon_dir(method)
    with np.load(d / f"{name}.npz") as z:
        V = np.array(z["V"], dtype=np.float64)
        F = np.asarray(z["F"], dtype=np.int64)
    with open(d / f"{name}.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    lmk = np.asarray(meta["landmarks_68"], dtype=np.float64)[list(LMK7_IBUG)]
    if meta["coordinate_system"].startswith("image pixels"):
        V[:, 1] *= -1.0
        lmk[:, 1] *= -1.0
    return V, F, lmk


def load_scan(subject: str):
    """(V, F, 7 landmark) della scansione, in mm."""
    import igl
    import numpy as np

    obj, pp = scan_paths(subject)
    V, F = igl.read_triangle_mesh(str(obj))
    return np.asarray(V, dtype=np.float64), np.asarray(F, dtype=np.int64), load_pp(pp)


# ------------------------------------------------------------------------- geometria

def similarity_to(src, dst):
    """(s, R, t) con dst ~ s * src @ R + t, senza riflessione (fg_metrics.procrustes_transform)."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "faceBench" / "latentVSpipeline"))
    import fg_metrics as fg

    return fg.procrustes_transform(src, dst, scaling=True)


def build_template(landmarks: list, n_iter: int = 20):
    """Media di Procrustes generalizzata dei 7 landmark, centrata, in mm.

    Ogni iterazione allinea tutti i set alla media corrente (similarita') e ricalcola la
    media; alla fine la media si riporta alla dimensione di centroide media dei set di
    partenza, cosi' il frame canonico resta in millimetri.
    """
    import numpy as np

    sets = [np.asarray(L, dtype=np.float64) for L in landmarks]
    size = float(np.mean([np.linalg.norm(L - L.mean(0)) for L in sets]))
    mean = sets[0] - sets[0].mean(0)
    for _ in range(n_iter):
        aligned = []
        for L in sets:
            s, R, t = similarity_to(L, mean)
            aligned.append(s * L @ R + t)
        new = np.mean(aligned, axis=0)
        new -= new.mean(0)
        new *= size / np.linalg.norm(new)
        if np.abs(new - mean).max() < 1e-9:
            mean = new
            break
        mean = new
    return mean


def to_canonical(V, lmk, template):
    """Mesh e landmark nel frame del template, con la similarita' dai SUOI landmark."""
    s, R, t = similarity_to(lmk, template)
    return s * V @ R + t, s * lmk @ R + t, s


def now_mask(lmk):
    """``compute_mask`` di now_evaluation/scan2mesh_computations.py, riscritta identica."""
    import numpy as np

    nose_bottom = lmk[4]
    nose_bridge = (lmk[1] + lmk[2]) / 2.0
    centre = nose_bottom + 0.3 * (nose_bridge - nose_bottom)
    outer_eye_dist = float(np.linalg.norm(lmk[0] - lmk[3]))
    nose_dist = float(np.linalg.norm(nose_bridge - nose_bottom))
    return centre, 1.4 * (outer_eye_dist + nose_dist) / 2.0


# ------------------------------------------------------------------------- csv

def write_rows(path: Path, fields, rows: list[dict], **params) -> None:
    """csv piu' sidecar json coi parametri, come ws3b_common.write_rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{row[k]:.10g}" if isinstance(row.get(k), float) else row.get(k, ""))
                             for k in fields})
    path.with_suffix(".json").write_text(
        json.dumps({"n_rows": len(rows), "params": params}, indent=2, default=str), encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))
