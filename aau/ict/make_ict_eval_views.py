#!/usr/bin/env python3
"""Viste a symlink per valutare i modelli BFM sui dati ICT, senza toccare il codice di ricerca.

Due viste, entrambe in layout PIATTO `id<NNNNN>_GTready_<topologia>.npz` (e' l'unico che
`GTReadyDatasetNPZ` sa leggere: `os.listdir`, niente ricorsione) e con nomi che
`infer_topology_label_from_name` sa etichettare:

1. ``eval_view_heldout/`` -- le 500 identita' held-out x 6 topologie (3000 symlink).
   Serve a RESTRINGERE l'eval ai soli soggetti held-out: gli script di ranking rifanno
   il loro split (``_select_subject_subset`` -> ``rebuild_subject_split``) sui soggetti
   che TROVANO nella data dir, e con ``train_ready/npz_withops`` (5000 soggetti) l'80%
   di quelli scelti sarebbe roba di training ICT.  Con la vista il pool e' 500, e
   ``--max_subjects 500 --eval_fraction 0.2 --subject_split eval --seed 1234`` ne estrae
   100 in modo deterministico: stesso conteggio di coppie (148.500 mesh pair
   cross-topology) del numero storico BFM->FLAME 0.478 (v2_work/STATUS.md:398), quindi
   confrontabile.  Usare tutti i 500 soggetti (``--subject_split all --max_subjects 0``)
   vorrebbe dire 3,74 M di mesh pair per scenario, 25x il costo: non e' stato fatto.

2. ``expr_view/<espressione>_<intensita>/`` -- una vista per ognuna delle 15 combinazioni
   (5 espressioni x 3 intensita'), con DUE etichette di topologia per soggetto:
     - ``id<NNNNN>_GTready_original.npz`` -> la mesh ESPRESSIVA (expressions_withops)
     - ``id<NNNNN>_GTready_neutral.npz``  -> la mesh NEUTRA della stessa identita'
   Le due etichette escono da ``infer_topology_label_from_name`` (``original`` e' in
   KNOWN_TOPOLOGY_LABEL_TOKENS, ``neutral`` passa dal ramo "un solo token non numerico"),
   quindi lo stesso script di ranking da' i due regimi chiesti senza modifiche:
     - same-expression:      ``--pair_mode within_topology --topology_labels original``
     - expression-vs-neutral: ``--pair_mode cross_topology`` (le sole coppie miste
       possibili sono espressiva_i x neutra_j e neutra_i x espressiva_j)
   La GT resta quella delle identita' NEUTRE (``datasets/ICT/gt``): e' esattamente la
   domanda, cioe' se l'identita' sopravvive all'espressione.

Solo stdlib: gira sul frontend, dove non c'e' numpy.

  aau/ict/make_ict_eval_views.py                 # entrambe le viste
  aau/ict/make_ict_eval_views.py --only heldout  # solo la prima
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
ICT_DIR = REPO_ROOT / "datasets" / "ICT"

TOPOLOGIES = ("crop", "down8k", "noisy", "original", "remesh", "up60k")
EXPR_RE = re.compile(r"^(id\d+)_expr_([A-Za-z]+)_([0-9.]+)\.npz$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ict-dir", type=Path, default=ICT_DIR)
    p.add_argument("--only", choices=("heldout", "expr", "both"), default="both")
    return p.parse_args()


def read_heldout(ict_dir: Path) -> list[str]:
    path = ict_dir / "train_ready" / "split_heldout.txt"
    subjects = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not subjects:
        raise RuntimeError(f"split held-out vuoto: {path}")
    return subjects


def link(dst: Path, src: Path) -> None:
    """Symlink idempotente al percorso REALE del target.

    `realpath` perche' la sorgente e' spesso essa stessa un symlink
    (``train_ready/npz_withops`` punta a ``topo_withops``): una catena di symlink
    funzionerebbe, ma un `ls -l` della vista non direbbe piu' dove sta il file.
    """
    real = src.resolve()
    if not real.is_file():
        raise FileNotFoundError(f"target assente: {src}")
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and os.readlink(dst) == str(real):
            return
        dst.unlink()
    dst.symlink_to(real)


def build_heldout_view(ict_dir: Path, subjects: list[str]) -> Path:
    out = ict_dir / "eval_view_heldout"
    out.mkdir(parents=True, exist_ok=True)
    src_dir = ict_dir / "train_ready" / "npz_withops"
    n = 0
    for sid in subjects:
        for topo in TOPOLOGIES:
            name = f"{sid}_GTready_{topo}.npz"
            link(out / name, src_dir / name)
            n += 1
    print(f"[views] eval_view_heldout: {n} symlink ({len(subjects)} soggetti x {len(TOPOLOGIES)} topologie)")
    return out


def expression_conditions(ict_dir: Path, subjects: list[str]) -> dict[tuple[str, str], list[str]]:
    """Mappa (espressione, intensita') -> soggetti presenti, letta dai nomi dei file."""
    expr_dir = ict_dir / "expressions_withops"
    found: dict[tuple[str, str], list[str]] = {}
    wanted = set(subjects)
    for entry in sorted(os.listdir(expr_dir)):
        m = EXPR_RE.match(entry)
        if not m:
            continue
        sid, expr, intensity = m.group(1), m.group(2), m.group(3)
        if sid in wanted:
            found.setdefault((expr, intensity), []).append(sid)
    if not found:
        raise RuntimeError(f"nessun file id*_expr_*.npz in {expr_dir}")
    return found


def build_expr_views(ict_dir: Path, subjects: list[str]) -> Path:
    out_root = ict_dir / "expr_view"
    out_root.mkdir(parents=True, exist_ok=True)
    expr_dir = ict_dir / "expressions_withops"
    # Non `topo_withops`: quella dir usa i nomi nativi `ictNNNN`, senza l'offset +10000 che
    # la matrice GT e i file di espressione usano. La vista train-ready e' l'unico posto in
    # cui la mesh neutra si chiama con lo stesso `idNNNNN` dell'espressione.
    neutral_dir = ict_dir / "train_ready" / "npz_withops"

    conditions = expression_conditions(ict_dir, subjects)
    manifest = {
        "source_expressions": str(expr_dir),
        "source_neutral": str(neutral_dir),
        "subjects": len(subjects),
        "conditions": [],
    }
    for (expr, intensity), sids in sorted(conditions.items()):
        if len(sids) != len(subjects):
            raise RuntimeError(
                f"{expr}_{intensity}: {len(sids)} soggetti su {len(subjects)} held-out. "
                "Una vista incompleta cambierebbe il set di coppie da una condizione all'altra."
            )
        view = out_root / f"{expr}_{intensity}"
        view.mkdir(parents=True, exist_ok=True)
        for sid in sids:
            link(view / f"{sid}_GTready_original.npz", expr_dir / f"{sid}_expr_{expr}_{intensity}.npz")
            link(view / f"{sid}_GTready_neutral.npz", neutral_dir / f"{sid}_GTready_original.npz")
        print(f"[views] expr_view/{expr}_{intensity}: {2 * len(sids)} symlink")
        manifest["conditions"].append(
            {"expression": expr, "intensity": intensity, "n_subjects": len(sids), "dir": str(view)}
        )

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[views] expr_view: {len(conditions)} condizioni, manifest in {out_root / 'manifest.json'}")
    return out_root


def main() -> None:
    args = parse_args()
    subjects = read_heldout(args.ict_dir)
    print(f"[views] held-out: {len(subjects)} soggetti ({subjects[0]}..{subjects[-1]})")
    if args.only in ("heldout", "both"):
        build_heldout_view(args.ict_dir, subjects)
    if args.only in ("expr", "both"):
        build_expr_views(args.ict_dir, subjects)


if __name__ == "__main__":
    main()
