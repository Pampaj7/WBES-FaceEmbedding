"""Percorsi, coppie del protocollo e I/O dei csv per WS3a su Multiface.

Il gemello di ``aau/baselines/common.py``, ma per un protocollo a COPPIE invece che a
matrice: su REMESH le baseline riempiono una 100x100 e la confrontano con D_GT, qui non
c'e' nessuna D_GT e le uniche distanze che servono sono quelle delle 8000 coppie di
``pairs_protocol.json`` (2000 per classe).  Una matrice completa sarebbe 2848^2 = 8.1
milioni di celle per topologia, cioe' 250 volte il lavoro utile.

Ogni metrica scrive un csv per coppia di topologie, in
``aau/runs/multiface_ws3a/<metrica>_<topoA>_<topoB>.csv``, con una riga per coppia:
classe, etichetta same/different, soggetti, segmenti, frame, nomi delle mesh, distanza.
La riga contiene i soggetti perche' il bootstrap dell'analisi ricampiona i 13 SOGGETTI,
non le coppie (le coppie non sono indipendenti: ogni soggetto compare in migliaia).

Accanto al csv c'e' un sidecar ``.json`` con i parametri della corsa (variante della
metrica, numero di punti campionati, sigma, viste, checkpoint...): non stanno nel csv
perche' sarebbero 8000 volte la stessa stringa.

Quali coppie di topologie si calcolano lo decide ``WBES_WS3A_PAIRS`` (``clean``, il
default storico, oppure ``hard``): vedi ``TOPOLOGY_PAIRS`` piu' sotto.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
REPO_ROOT = AAU_DIR.parent

PREP_DIR = Path(os.environ.get("WBES_MF_PREP_DIR", REPO_ROOT / "datasets" / "Multiface" / "prep"))
PROTOCOL_PATH = Path(os.environ.get("WBES_MF_PROTOCOL", THIS_DIR / "pairs_protocol.json"))
MANIFEST_PATH = Path(os.environ.get("WBES_MF_MANIFEST", PREP_DIR / "manifest.csv"))
OUT_ROOT = Path(os.environ.get("WBES_WS3A_OUT", AAU_DIR / "runs" / "multiface_ws3a"))

# Convenzione degli operatori per il ramo latent.  Il checkpoint v1 e' stato addestrato
# sugli operatori STANDARD (vedi aau/multiface/multiface_ops.sbatch), quindi il default
# e' `_withops`; `_withops_areanorm` serve solo quando si valuta un modello areanorm.
OPS_SUFFIX = os.environ.get("WBES_MF_OPS_SUFFIX", "_withops")

# Le quattro configurazioni chieste da WS3a: due in corrispondenza densa (tracked e' la
# topologia tracciata comune a tutti i soggetti), una senza corrispondenza ma a topologia
# uguale (remesh->remesh), una cross-topologia (tracked->remesh), una decimata (down).
CLEAN_TOPOLOGY_PAIRS = (
    ("tracked", "tracked"),
    ("remesh", "remesh"),
    ("tracked", "remesh"),
    ("down", "down"),
)

# Le cinque configurazioni DURE, con le tre varianti speculari alle perturbazioni REMESH
# (`crop`, `noisy`, `up` di prepare_multiface.py).  Sul set pulito il protocollo e' saturo
# (AUC ~ 1 per tutte le metriche tranne il varifold), perche' le tre topologie sono
# gentili: scansioni complete, pulite e a risoluzione confrontabile.  Qui invece
# tracked->crop e remesh->crop sono a sovrapposizione parziale (~70% della faccia da un
# lato solo), tracked->noisy e' geometria sporca a topologia identica, down->up e' un
# salto di risoluzione 1:5.5, e crop->crop e' il taglio da entrambi i lati.
# `tracked->tracked` c'e' anche qui, e non e' una ripetizione del giro pulito: e' il
# riferimento del delta, e va ricalcolato DENTRO questo protocollo perche' il protocollo e'
# cambiato (render normalizzati per mesh, varifold/currents senza sottocampionamento).
# Confrontare un'AUC nuova con una vecchia misurerebbe la correzione, non la perturbazione.
HARD_TOPOLOGY_PAIRS = (
    ("tracked", "tracked"),
    ("tracked", "crop"),
    ("remesh", "crop"),
    ("tracked", "noisy"),
    ("down", "up"),
    ("crop", "crop"),
)

# Quale dei due insiemi usano tutti gli script del ramo WS3a.  E' una variabile d'ambiente
# e non un argomento perche' la scelta deve valere insieme per render, metriche e analisi:
# i csv si distinguono dal nome (`<metrica>_<topoA>_<topoB>.csv`).  I due insiemi NON
# condividono la out dir: il giro `hard` normalizza i render per mesh e non sottocampiona i
# triangoli, quindi i suoi render e i suoi embedding non sono quelli del giro `clean` e
# vanno tenuti separati (`WBES_WS3A_OUT=.../multiface_ws3a_hard`).
PAIR_SETS = {"clean": CLEAN_TOPOLOGY_PAIRS, "hard": HARD_TOPOLOGY_PAIRS}
PAIR_SET = os.environ.get("WBES_WS3A_PAIRS", "clean")
if PAIR_SET not in PAIR_SETS:
    raise SystemExit(f"WBES_WS3A_PAIRS={PAIR_SET!r} sconosciuto (attesi {sorted(PAIR_SETS)})")
TOPOLOGY_PAIRS = PAIR_SETS[PAIR_SET]

# `bbox_proxy` non e' una metrica di forma ma la riga di controllo: sta in fondo perche' si
# legge dopo le altre.  Vedi `ws3a_geometric.py`.  `latent_joint` e' il modello congiunto
# BFM+ICT di WS2 (`ws3a_latent.py --metric latent_joint` con
# `WBES_MF_OPS_SUFFIX=_withops_areanorm`, la convenzione del suo training).  Il seme del
# bootstrap dell'analisi dipende solo dal confronto, quindi una riga in piu' non cambia le altre.
METRICS = ("chamfer", "rigid_icp", "varifold", "currents",
           "lpips", "arcface", "clip", "dinov2", "latent_v1", "latent_joint", "bbox_proxy")

# Ordine stabile delle classi: e' l'ordine in cui le coppie finiscono nel csv.
CLASSES = (
    "a_same_subject_same_expression",
    "b_same_subject_diff_expression",
    "c_diff_subject_same_expression",
    "d_diff_subject_diff_expression",
)

CSV_FIELDS = (
    "pair_index", "pair_class", "label",
    "subject_a", "subject_b", "segment_a", "segment_b", "frame_a", "frame_b",
    "name_a", "name_b", "distance",
)


@dataclass(frozen=True)
class PairRecord:
    """Una coppia del protocollo, gia' risolta contro il manifest."""

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


def load_protocol(path: Path = None) -> dict:
    with open(PROTOCOL_PATH if path is None else path, encoding="utf-8") as fh:
        return json.load(fh)


def load_manifest(path: Path = None, topology: str = "tracked") -> dict[str, tuple[str, str, str]]:
    """(soggetto, segmento, frame) per nome mesh.  Le topologie hanno tutte gli stessi nomi."""
    out: dict[str, tuple[str, str, str]] = {}
    with open(MANIFEST_PATH if path is None else path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["topology"] != topology:
                continue
            out[row["name"]] = (row["subject"], row["segment"], row["frame"])
    if not out:
        raise RuntimeError(f"nessuna riga con topology={topology!r} in {MANIFEST_PATH}")
    return out


def load_pairs(max_pairs_per_class: int = 0, protocol: dict = None) -> list[PairRecord]:
    """Le coppie del protocollo in ordine a, b, c, d.

    ``max_pairs_per_class > 0`` assottiglia ogni classe a passo costante: serve solo per i
    test di velocita' (50 per classe = le 200 coppie con cui si stima il ``--time``), non
    per i numeri veri.  A passo costante e non con i primi N perche' il campionamento del
    protocollo e' round-robin sui gruppi: i primi 50 della classe (a) sono tutti dello
    stesso soggetto, e su quelli qualunque metrica separa perfettamente.
    """
    protocol = load_protocol() if protocol is None else protocol
    manifest = load_manifest()
    records: list[PairRecord] = []
    for pair_class in CLASSES:
        block = protocol["classes"][pair_class]
        pairs = block["pairs"]
        if 0 < max_pairs_per_class < len(pairs):
            pairs = pairs[:: len(pairs) // max_pairs_per_class][:max_pairs_per_class]
        for name_a, name_b in pairs:
            subject_a, segment_a, frame_a = manifest[name_a]
            subject_b, segment_b, frame_b = manifest[name_b]
            records.append(PairRecord(
                pair_index=len(records), pair_class=pair_class, label=block["label"],
                name_a=name_a, name_b=name_b,
                subject_a=subject_a, subject_b=subject_b,
                segment_a=segment_a, segment_b=segment_b,
                frame_a=frame_a, frame_b=frame_b,
            ))
    return records


def topology_names(records: list[PairRecord], topology_pairs=TOPOLOGY_PAIRS) -> dict[str, list[str]]:
    """Per ogni topologia usata, i nomi che vanno caricati da quella cartella.

    Con tracked->remesh solo il lato sinistro serve da tracked e solo il destro da
    remesh; ma tracked->tracked usa entrambi i lati, quindi in pratica tracked le vuole
    tutte.  Si tiene il conto esatto perche' con un protocollo troncato (i test) la
    differenza e' grossa.
    """
    out: dict[str, set[str]] = {}
    for topology_a, topology_b in topology_pairs:
        out.setdefault(topology_a, set()).update(rec.name_a for rec in records)
        out.setdefault(topology_b, set()).update(rec.name_b for rec in records)
    return {topology: sorted(names) for topology, names in sorted(out.items())}


def mesh_path(topology: str, name: str) -> Path:
    """npz mesh-only, chiavi V/F."""
    return PREP_DIR / topology / f"{name}.npz"


def ops_dir(topology: str) -> Path:
    """Cartella con gli operatori DiffusionNet, nel formato di GTReadyDatasetNPZ."""
    return PREP_DIR / f"{topology}{OPS_SUFFIX}"


def csv_path(metric: str, topology_a: str, topology_b: str, out_root: Path = None) -> Path:
    root = OUT_ROOT if out_root is None else out_root
    return root / f"{metric}_{topology_a}_{topology_b}.csv"


def write_distances(path: Path, records: list[PairRecord], values, metric: str,
                    topology_a: str, topology_b: str, seconds: float = None, **params) -> None:
    """csv delle distanze piu' il sidecar json con i parametri della corsa."""
    if len(values) != len(records):
        raise ValueError(f"{len(values)} distanze per {len(records)} coppie")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_FIELDS)
        for rec, value in zip(records, values):
            writer.writerow([
                rec.pair_index, rec.pair_class, rec.label,
                rec.subject_a, rec.subject_b, rec.segment_a, rec.segment_b,
                rec.frame_a, rec.frame_b, rec.name_a, rec.name_b,
                f"{float(value):.10g}",
            ])
    meta = {"metric": metric, "topology_a": topology_a, "topology_b": topology_b,
            "n_pairs": len(records), "protocol": str(PROTOCOL_PATH), "params": params}
    if seconds is not None:
        meta["seconds"] = float(seconds)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def read_distances(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))
