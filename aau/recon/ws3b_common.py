"""Percorsi, nomi e I/O dei csv per WS3b: i tre metodi di ricostruzione contro Multiface.

Il gemello di ``aau/multiface/ws3a_common.py``, ma il confronto qui non e' fra due mesh
del dataset: e' fra una RICOSTRUZIONE da immagine e la mesh tracciata dello stesso frame.
Un elemento di WS3b e' quindi una quaterna

    <soggetto>__<segmento>__<frame>__<camera>

e la sua verita' a terra e' ``<soggetto>__<segmento>__<frame>`` in
``datasets/Multiface/prep/tracked``, cioe' tre elementi (le tre camere frontali) hanno la
stessa GT.  I nomi delle mesh ricostruite sono gli stessi degli elementi, perche' i tre
runner (``tddfa_v2_run.py`` e compagnia) chiamano l'uscita come lo stem dell'immagine di
ingresso: le immagini vengono percio' esposte come symlink gia' battezzati
(``ws3b_stage_images.py``), invece di rinominare a posteriori.

Solo stdlib: questo modulo gira anche sul frontend, che non ha numpy.

Le tre convenzioni che valgono per tutto il cantiere
---------------------------------------------------
**Mano del sistema di riferimento.**  I tre metodi restituiscono lo spazio immagine
(x a destra, y in GIU', z profondita', vedi ``common.COORD_SYSTEM``), che rispetto al
frame testa di Multiface (x a destra, y in su, z in avanti) e' riflesso.  Una similarita'
non contiene riflessioni, quindi prima di qualunque allineamento la y va negata.  Non e'
una scelta di protocollo ma una conversione di unita'.  Verificato su 30 ricostruzioni per
metodo (job 1019511), errore mediano stile NoW contro la GT ritagliata, mesh come esce dal
metodo -> mesh con y negata: 3DDFA_V2 3.57 -> 1.27 mm, SynergyNet 4.09 -> 1.57 mm, PRNet
3.56 -> 1.40 mm.  Serve a tutti e tre, quindi anche a PRNet, che non viene dalla stessa
famiglia degli altri due.  Il fattore e' 2.5-2.8x e non un ordine di grandezza perche' una
faccia e' quasi simmetrica rispetto al piano sagittale: la mesh riflessa e' ancora una
faccia, e l'ICP ci si avvicina.

**Regione volto.**  La patch tracciata di Multiface arriva alla nuca e alla base del collo
(bbox mediana 190x341x227 mm), che nessun metodo da singola immagine ricostruisce.  Come
in NoW si ritaglia una sfera di 95 mm attorno alla punta del naso; la punta del naso e' il
vertice di z massima sulla mesh MEDIA delle 2848 tracciate, quindi un indice fisso, uguale
per tutti i soggetti e tutti i frame (la topologia tracciata e' in corrispondenza densa).

Sul lato ricostruzione la scala metrica non e' osservabile da una foto sola, quindi il
raggio si esprime in **aperture oculari**: origine nel landmark 30 (punta del naso), unita'
la distanza fra i landmark 36 e 45 (gli exocanthion).  Il raggio giusto e' allora
``95 mm / ex-ex della GT``, con l'ex-ex della GT misurato sui landmark stimati da
``ws3b_landmarks.py`` -- cosi' il ritaglio della ricostruzione e quello della GT sono la
stessa regione anatomica.  La prima versione ci metteva 90 mm, la media adulta di Farkas,
invece del valore misurato: con la scala vera stimata dall'ICP il ritaglio veniva 90.7 mm
(3DDFA_V2), 91.6 (SynergyNet), 91.0 (PRNet) contro i 95 della GT, cioe' i due supporti non
coincidevano e la Chamfer grezza pagava una corona di superficie che c'era da una parte e
non dall'altra.

**Chi si confronta con chi.**  ``gt_face`` (la GT ritagliata) sta da una parte in tutti e
tre i criteri.  Dall'altra: la ricostruzione INTERA per il criterio stile NoW (la distanza
va dai punti della GT alla superficie ricostruita, e superficie in piu' non disturba,
esattamente come nel protocollo NoW), la ricostruzione RITAGLIATA per la Chamfer grezza e
per il latent, dove la normalizzazione maxabs e il supporto della mesh contano.
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

MF_FULL = Path(os.environ.get("WBES_MF_FULL_DIR", REPO_ROOT / "datasets" / "Multiface" / "full"))
MF_PREP = Path(os.environ.get("WBES_MF_PREP_DIR", REPO_ROOT / "datasets" / "Multiface" / "prep"))
OUT_ROOT = Path(os.environ.get("WBES_WS3B_OUT", AAU_DIR / "runs" / "multiface_ws3b"))

METHODS = ("3ddfa_v2", "synergynet", "prnet")

# I criteri del cantiere, nell'ordine in cui compaiono nel summary.
#
# ``sim_icp_p2s_median`` si chiamava ``now_median``, ed era un nome sbagliato: il
# protocollo NoW stima la similarita' da SETTE landmark e non fa nessun ICP, mentre qui la
# similarita' viene da 30 iterazioni di ICP sull'intera superficie.  Sono due criteri
# diversi -- l'ICP puo' compensare un errore di forma ruotando e riscalando, i landmark no
# -- e ora ci sono tutti e due, ma non con lo stesso peso.  ``now_median``, il NoW vero, e'
# DIAGNOSTICO e resta fuori dalla classifica: i 7 landmark sulla GT non esistono in
# Multiface e quelli stimati hanno 5-7 mm di dispersione contro un segnale di 1.2-1.6 mm
# (vedi la nota in ws3b_analysis.py).  Il criterio allineato di riferimento e' quindi
# ``sim_icp_p2s_median``, col nome che dice cosa fa.
CRITERIA = ("sim_icp_p2s_median", "chamfer_raw", "latent_v1")
DIAGNOSTIC_CRITERIA = ("now_median",)

# Raggio del ritaglio NoW, in millimetri: NoW ritaglia la scansione a 95 mm dal centro del
# volto prima di misurare la distanza scansione->predizione.
FACE_RADIUS_MM = 95.0

# I 68 landmark iBUG, indici 0-based: 30 = punta del naso, 36/45 = angoli esterni degli occhi.
LMK_NOSE_TIP = 30
LMK_EYE_OUTER = (36, 45)

# I 7 punti del protocollo NoW, come indici nei 68 iBUG che i tre metodi restituiscono.
# ex = exocanthion (angolo esterno), en = endocanthion (angolo interno), prn = pronasale,
# ch = cheilion (angolo della bocca).  "_r"/"_l" sono destra/sinistra NELL'IMMAGINE.
LMK7_NAMES = ("ex_r", "en_r", "en_l", "ex_l", "prn", "ch_r", "ch_l")
LMK7_IBUG = (36, 39, 42, 45, 30, 48, 54)

# m--20180105--0000--002539136--GHS  ->  002539136   (come prepare_multiface.py)
CAPTURE_RE = re.compile(r"^m--\d+--\d+--(?P<subject>[^-]+)--GHS$")

MANIFEST_FIELDS = ("name", "subject", "segment", "frame", "camera", "version",
                   "capture", "image", "gt_name")

# Colonne dei csv per-ricostruzione (un elemento per riga) e per-coppia (l'AUC di identita').
ITEM_FIELDS = ("name", "subject", "segment", "frame", "camera")
PAIR_FIELDS = ("pair_index", "pair_class", "label",
               "subject_a", "subject_b", "segment_a", "segment_b", "frame_a", "frame_b",
               "name_a", "name_b", "distance")


@dataclass(frozen=True)
class Item:
    """Una ricostruzione: un'immagine, quindi una quaterna soggetto/segmento/frame/camera."""

    name: str
    subject: str
    segment: str
    frame: str
    camera: str
    version: str
    capture: str
    image: str
    gt_name: str


def subject_of(capture: str) -> str:
    m = CAPTURE_RE.match(capture)
    if m is None:
        raise ValueError(f"nome di capture inatteso: {capture}")
    return m.group("subject")


def version_of(segments) -> str:
    """v1 = i 10 soggetti Mugsy con segmenti `E0xx_*`, v2 = i 3 con `EXP_*`."""
    segments = list(segments)
    if all(s.startswith("EXP_") for s in segments):
        return "v2"
    if all(re.match(r"^E\d{3}_", s) for s in segments):
        return "v1"
    raise ValueError(f"segmenti ne' v1 ne' v2: {sorted(segments)[:3]}")


def item_name(subject: str, segment: str, frame: str, camera: str) -> str:
    return f"{subject}__{segment}__{frame}__{camera}"


def gt_name_of(subject: str, segment: str, frame: str) -> str:
    return f"{subject}__{segment}__{frame}"


def manifest_path(out_root: Path = None) -> Path:
    return (OUT_ROOT if out_root is None else out_root) / "manifest.csv"


def auc_manifest_path(out_root: Path = None) -> Path:
    """Manifest ridotto a UNA camera per soggetto, in ingresso a make_pairs_protocol.py.

    L'AUC di identita' confronta ricostruzioni fra loro: se dentro il protocollo entrassero
    piu' camere dello stesso frame, la classe "stesso soggetto stessa espressione" si
    riempirebbe di coppie che sono lo stesso istante visto da due obiettivi, cioe' del caso
    piu' facile che esista.  Si tiene la prima camera in ordine, una per soggetto.
    """
    return (OUT_ROOT if out_root is None else out_root) / "auc_manifest.csv"


def protocol_path(out_root: Path = None) -> Path:
    return (OUT_ROOT if out_root is None else out_root) / "pairs_protocol.json"


def images_dir(out_root: Path = None) -> Path:
    return (OUT_ROOT if out_root is None else out_root) / "images"


def recon_dir(method: str, out_root: Path = None) -> Path:
    """Mesh come escono dal metodo: topologia fissa del metodo, pixel dell'immagine."""
    return (OUT_ROOT if out_root is None else out_root) / "recon" / method


def recon_face_dir(method: str, out_root: Path = None) -> Path:
    """Mesh ritagliate alla regione volto e decimate alla risoluzione di ``gt_face``."""
    return (OUT_ROOT if out_root is None else out_root) / "recon_face" / method


def recon_face_ops_dir(method: str, out_root: Path = None, ops_suffix: str = "_withops") -> Path:
    return (OUT_ROOT if out_root is None else out_root) / f"recon_face{ops_suffix}" / method


def recon_face_gtclip_dir(method: str, out_root: Path = None) -> Path:
    """``recon_face`` ristretta alla patch GT di ``chamfer_gtclip_mm`` (ws3b_gtclip_meshes.py)."""
    return (OUT_ROOT if out_root is None else out_root) / "recon_face_gtclip" / method


def recon_face_gtclip_ops_dir(method: str, out_root: Path = None,
                              ops_suffix: str = "_withops") -> Path:
    return (OUT_ROOT if out_root is None else out_root) / f"recon_face_gtclip{ops_suffix}" / method


def gt_face_dir(out_root: Path = None) -> Path:
    """GT tracciata ritagliata a 95 mm dalla punta del naso: topologia fissa e condivisa."""
    return (OUT_ROOT if out_root is None else out_root) / "gt_face"


def gt_face_ops_dir(out_root: Path = None, ops_suffix: str = "_withops") -> Path:
    return (OUT_ROOT if out_root is None else out_root) / f"gt_face{ops_suffix}"


def gt_mesh_path(gt_name: str) -> Path:
    return MF_PREP / "tracked" / f"{gt_name}.npz"


def face_region_path(out_root: Path = None) -> Path:
    """Indici del ritaglio volto sulla topologia tracciata, piu' le sue statistiche."""
    return (OUT_ROOT if out_root is None else out_root) / "face_region.json"


def landmarks_path(out_root: Path = None) -> Path:
    """I 7 landmark NoW sulla topologia tracciata: indici di vertice piu' controlli.

    Li stima ``ws3b_landmarks.py``; li leggono ``ws3b_prepare_meshes.py`` (per il raggio
    del ritaglio) e ``ws3b_geometric.py`` (per il criterio NoW vero).
    """
    return (OUT_ROOT if out_root is None else out_root) / "gt_landmarks.json"


def gt_csv_path(method: str, out_root: Path = None) -> Path:
    """Criteri contro la GT, una riga per ricostruzione."""
    return (OUT_ROOT if out_root is None else out_root) / f"gt_metrics_{method}.csv"


def gt_latent_csv_path(method: str, out_root: Path = None, metric: str = "latent_v1") -> Path:
    """Il v1 tiene il nome storico ``gt_latent_<metodo>.csv``; un altro modello va in
    ``gt_<metrica>_<metodo>.csv``, cosi' i csv del v1 non vengono mai riscritti."""
    root = OUT_ROOT if out_root is None else out_root
    return root / (f"gt_latent_{method}.csv" if metric == "latent_v1" else f"gt_{metric}_{method}.csv")


def gt_latent_gtclip_csv_path(method: str, out_root: Path = None, metric: str = "latent_v1") -> Path:
    """Il latente con la ricostruzione ritagliata alla patch GT: csv a parte, cosi'
    ``gt_latent_<metodo>.csv`` resta quello della corsa originale."""
    root = OUT_ROOT if out_root is None else out_root
    return root / (f"gt_latent_gtclip_{method}.csv" if metric == "latent_v1"
                   else f"gt_{metric}_gtclip_{method}.csv")


def gtclip_mesh_csv_path(method: str, out_root: Path = None) -> Path:
    """Controlli del ritaglio alla patch GT, una riga per ricostruzione."""
    return (OUT_ROOT if out_root is None else out_root) / f"gtclip_meshes_{method}.csv"


def pair_csv_path(metric: str, method: str, out_root: Path = None) -> Path:
    """Distanze fra ricostruzioni, per l'AUC stesso/diverso soggetto."""
    return (OUT_ROOT if out_root is None else out_root) / f"pairs_{metric}_{method}.csv"


# ------------------------------------------------------------------------- manifest

def write_manifest(path: Path, items: list[Item]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for it in items:
            writer.writerow({k: getattr(it, k) for k in MANIFEST_FIELDS})


def load_manifest(path: Path = None) -> list[Item]:
    path = manifest_path() if path is None else path
    if not path.is_file():
        raise SystemExit(f"manifest assente: {path}\n"
                         f"  Crealo con: python3 aau/recon/ws3b_stage_images.py")
    with open(path, newline="") as fh:
        return [Item(**{k: row[k] for k in MANIFEST_FIELDS}) for row in csv.DictReader(fh)]


def write_auc_manifest(path: Path, items: list[Item]) -> list[Item]:
    """Il manifest a una camera per soggetto, nel formato che legge make_pairs_protocol.py.

    Quel modulo vuole le colonne name/subject/segment/frame/version/topology e filtra su
    ``topology``; qui la colonna vale sempre "recon" perche' di topologie ce n'e' una sola.
    """
    cameras: dict[str, str] = {}
    for it in items:
        if it.camera < cameras.get(it.subject, "~"):
            cameras[it.subject] = it.camera
    kept = [it for it in items if it.camera == cameras[it.subject]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(("name", "subject", "segment", "frame", "version", "topology"))
        for it in kept:
            writer.writerow((it.name, it.subject, it.segment, it.frame, it.version, "recon"))
    return kept


# ------------------------------------------------------------------------- coppie

@dataclass(frozen=True)
class PairRecord:
    """Una coppia di RICOSTRUZIONI del protocollo di identita'."""

    pair_index: int
    pair_class: str
    label: str
    name_a: str
    name_b: str
    subject_a: str
    subject_b: str
    segment_a: str
    segment_b: str
    frame_a: str
    frame_b: str


# Stesso ordine e stessi nomi di aau/multiface/ws3a_common.py: le due tabelle di AUC
# (scansioni vere in WS3a, ricostruzioni qui) devono potersi leggere una accanto all'altra.
CLASSES = (
    "a_same_subject_same_expression",
    "b_same_subject_diff_expression",
    "c_diff_subject_same_expression",
    "d_diff_subject_diff_expression",
)


def load_pairs(protocol: dict = None, path: Path = None) -> list[PairRecord]:
    if protocol is None:
        with open(protocol_path() if path is None else path, encoding="utf-8") as fh:
            protocol = json.load(fh)
    by_name = {it.name: it for it in load_manifest()}
    records: list[PairRecord] = []
    for pair_class in CLASSES:
        block = protocol["classes"][pair_class]
        for name_a, name_b in block["pairs"]:
            a, b = by_name[name_a], by_name[name_b]
            records.append(PairRecord(
                pair_index=len(records), pair_class=pair_class, label=block["label"],
                name_a=name_a, name_b=name_b,
                subject_a=a.subject, subject_b=b.subject,
                segment_a=a.segment, segment_b=b.segment,
                frame_a=a.frame, frame_b=b.frame,
            ))
    return records


# ------------------------------------------------------------------------- csv

def write_rows(path: Path, fields, rows: list[dict], **params) -> None:
    """csv piu' sidecar json coi parametri della corsa, come in ws3a_common.write_distances."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k, "")) for k in fields})
    path.with_suffix(".json").write_text(
        json.dumps({"n_rows": len(rows), "params": params}, indent=2, default=str),
        encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _fmt(value):
    return f"{value:.10g}" if isinstance(value, float) else value
