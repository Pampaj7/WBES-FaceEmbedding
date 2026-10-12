"""Percorsi, lettura dei CSV di evidenza con il numero di riga, formato dei numeri per il LaTeX.

Solo libreria standard: le tabelle si generano anche sul frontend (python3 senza numpy). Le figure
(make_figures.py, make_teaser.py) importano questo modulo dentro il container.

Ogni numero che finisce nel paper porta la sua fonte ``file:riga`` (riga 1 = intestazione del CSV), relativa alla
radice del repo: le tabelle generate la scrivono in un commento accanto a ogni riga, le figure in un file
``<figura>.sources.txt`` accanto al PDF.

Matrice unificata (``aau/runs/evidence/full_matrix/``, la produce un altro lavoro): finche' non esiste, ogni cella
che ne dipende esce come ``\\todo{matrice}`` (tabelle) o con il timbro FALLBACK / PLACEHOLDER (figure). Lo schema
atteso e' in ``FULL_MATRIX_SCHEMA``; se i file esistono ma lo schema e' diverso lo script si ferma con un errore,
invece di indovinare.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EV = REPO / "aau" / "runs" / "evidence"
FM = EV / "full_matrix"
PAPER = REPO / "paper"
TABLES = PAPER / "tables"
FIGS = PAPER / "figs"

# Domini della tabella principale: (chiave nei CSV, gruppo senza crop, etichetta nel paper).
DOMAINS = [
    ("hifi3d", "nocrop_cross", "HIFI3D"),
    ("facescape", "nocrop_cross", "FaceScape"),
    ("faceverse", "mesh_pair_nocrop", "FaceVerse"),
    ("faceverse_neutral", "mesh_pair_nocrop", "FaceVerse neutral"),
    ("famos", "scan gallery -> scan", "FaMoS"),
]
GROUP = {d: g for d, g, _ in DOMAINS}

# Schema atteso dei CSV della matrice unificata. Colonne minime; altre colonne sono ammesse.
FULL_MATRIX_SCHEMA = {
    # una riga per (dominio, gruppo, gt, metodo): Spearman con IC su righe e repliche comuni a tutti i metodi
    "spearman.csv": ["domain", "group", "gt", "method", "point", "ci_low", "ci_high", "n_rows", "seed"],
    # differenze appaiate metodo - riferimento sulle stesse righe e repliche
    "paired.csv": ["domain", "group", "gt", "method", "ref", "delta", "ci_low", "ci_high"],
    # Spearman per perturbazione (coppie che coinvolgono quella discretizzazione)
    "perturbation.csv": ["domain", "gt", "method", "perturbation", "point", "ci_low", "ci_high"],
}


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(REPO))


def read_csv(path: Path) -> list[tuple[int, dict]]:
    """Righe del CSV come (numero di riga nel file, dizionario); la riga 1 e' l'intestazione."""
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        return [(i, r) for i, r in enumerate(rd, 2)]


def find(rows: list[tuple[int, dict]], **kw) -> tuple[int, dict] | None:
    """L'unica riga che combacia con tutte le coppie chiave = valore; None se nessuna, errore se piu' d'una."""
    hits = [(i, r) for i, r in rows if all(r.get(k) == v for k, v in kw.items())]
    if len(hits) > 1:
        raise SystemExit(f"piu' righe per {kw}: {[i for i, _ in hits]}")
    return hits[0] if hits else None


def full_matrix(name: str) -> list[tuple[int, dict]] | None:
    """Un CSV della matrice unificata, se esiste; None altrimenti. Schema diverso da quello atteso: errore."""
    p = FM / name
    if not p.is_file():
        return None
    rows = read_csv(p)
    need = FULL_MATRIX_SCHEMA[name]
    have = set(rows[0][1]) if rows else set()
    missing = [c for c in need if c not in have]
    if missing:
        raise SystemExit(f"{rel(p)}: schema diverso dall'atteso, mancano {missing}. Adattare common.FULL_MATRIX_SCHEMA "
                         f"e i lettori in make_tables.py / make_figures.py, non indovinare.")
    return rows


def json_line(path: Path, key: str, after: tuple[str, ...] = ()) -> tuple[int, float]:
    """(riga, valore) della prima riga ``"key": valore`` del JSON, dopo le righe che contengono, in ordine, le
    stringhe di ``after`` (es. ``('"gt": {', '"hifi3d": {')``)."""
    lines = Path(path).read_text().splitlines()
    start = 0
    for a in after:
        start = next(i for i in range(start, len(lines)) if a in lines[i])
    for i in range(start, len(lines)):
        s = lines[i].strip()
        if s.startswith(f'"{key}"'):
            return i + 1, float(s.split(":", 1)[1].strip().rstrip(","))
    raise SystemExit(f"{rel(path)}: chiave {key} non trovata dopo {after}")


def num(x: float, signed: bool = False) -> str:
    """Tre decimali, segno meno tipografico; ``signed`` mette il + davanti ai positivi (differenze)."""
    s = f"{x:+.3f}" if signed else f"{x:.3f}"
    return s.replace("-", "$-$")


def ci_cell(point: float, lo: float, hi: float, signed: bool = False) -> str:
    """Cella ``\\ci{punto}{basso}{alto}`` (macro definita nel preambolo del paper)."""
    return f"\\ci{{{num(point, signed)}}}{{{num(lo, signed)}}}{{{num(hi, signed)}}}"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"scritto {rel(path)}")


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())
