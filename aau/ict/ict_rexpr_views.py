#!/usr/bin/env python3
"""Viste a symlink per WS5 rifatto: espressioni casuali PER SOGGETTO, tre regimi.

Stessa idea di `aau/ict/make_ict_eval_views.py` (da cui questo file importa `link` e
`read_heldout`, invece di ricopiarli): layout PIATTO `id<NNNNN>_GTready_<etichetta>.npz`,
l'unico che `GTReadyDatasetNPZ` sa leggere, con nomi che `infer_topology_label_from_name`
sa etichettare. `rexpr1`..`rexpr5` passano dal ramo "un solo token non numerico" della
funzione, come gia' fa `neutral`.

Le tre famiglie di viste, sotto ``datasets/ICT/rexpr_view/``:

1. ``same_k<k>/``    una sola mesh per soggetto, la sua espressione k -> etichetta
   ``rexpr<k>``. Regime (a) same-k: ``--pair_mode within_topology``, cioe' entrambe le
   mesh della coppia portano la PROPRIA espressione k -- che non e' la stessa
   deformazione, perche' il vettore di espressione e' diverso da soggetto a soggetto.

2. ``neutral_k<k>/`` due mesh per soggetto, l'espressione k (``rexpr<k>``) e la neutra
   (``neutral``). Regime (b) espressione-contro-neutra: ``--pair_mode cross_topology``,
   le sole coppie a etichetta diversa sono espressiva_i x neutra_j e neutra_i x
   espressiva_j.

3. ``mixed/``        tutte e cinque le espressioni del soggetto, etichette ``rexpr1``..
   ``rexpr5``. Regime (c) misto: ``--pair_mode cross_topology`` accoppia la mesh A con
   espressione k e la mesh B con espressione k' != k -- 20 coppie di etichette per ogni
   coppia di soggetti. E' il caso reale: due scansioni qualunque, due espressioni
   qualunque, nessuna delle due nota.

La GT resta in tutti e tre i regimi quella delle identita' NEUTRE
(``train_ready/gt_matrix.npz``): la domanda e' se il ranking di identita' sopravvive
all'espressione, quindi il riferimento non deve muoversi con l'espressione.

Solo stdlib: gira sul frontend, dove non c'e' numpy.

  aau/ict/ict_rexpr_views.py
  aau/ict/ict_rexpr_views.py --only mixed
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
ICT_DIR = REPO_ROOT / "datasets" / "ICT"

sys.path.insert(0, str(THIS_DIR))
from make_ict_eval_views import link, read_heldout  # noqa: E402

REXPR_RE = re.compile(r"^(id\d+)_rexpr_(\d+)\.npz$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ict-dir", type=Path, default=ICT_DIR)
    p.add_argument("--only", choices=("same_k", "neutral_k", "mixed", "all"), default="all")
    return p.parse_args()


def expression_index(ict_dir: Path, subjects: list[str]) -> dict[str, list[str]]:
    """Mappa k -> soggetti presenti, letta dai nomi dei file withops."""
    expr_dir = ict_dir / "expressions_random_withops"
    found: dict[str, list[str]] = {}
    wanted = set(subjects)
    for entry in sorted(os.listdir(expr_dir)):
        match = REXPR_RE.match(entry)
        if not match:
            continue
        sid, k = match.group(1), match.group(2)
        if sid in wanted:
            found.setdefault(k, []).append(sid)
    if not found:
        raise RuntimeError(f"nessun file id*_rexpr_*.npz in {expr_dir}")
    for k, sids in found.items():
        if len(sids) != len(subjects):
            raise RuntimeError(
                f"k={k}: {len(sids)} soggetti su {len(subjects)} held-out. Una vista incompleta "
                "cambierebbe il set di coppie da un regime all'altro."
            )
    return found


def build_views(ict_dir: Path, subjects: list[str], only: str) -> Path:
    out_root = ict_dir / "rexpr_view"
    out_root.mkdir(parents=True, exist_ok=True)
    expr_dir = ict_dir / "expressions_random_withops"
    # Non `topo_withops`: quella dir usa i nomi nativi `ictNNNN`, senza l'offset +10000 che
    # la matrice GT e i file di espressione usano. La vista train-ready e' l'unico posto in
    # cui la mesh neutra si chiama con lo stesso `idNNNNN` dell'espressione.
    neutral_dir = ict_dir / "train_ready" / "npz_withops"

    index = expression_index(ict_dir, subjects)
    ks = sorted(index, key=int)
    manifest = {
        "source_expressions": str(expr_dir),
        "source_neutral": str(neutral_dir),
        "subjects": len(subjects),
        "k": ks,
        "views": [],
    }

    def record(view: Path, regime: str, n_links: int, labels: list[str]) -> None:
        print(f"[rexpr-views] {view.name}: {n_links} symlink, etichette {','.join(labels)}")
        manifest["views"].append({"dir": str(view), "regime": regime,
                                  "n_subjects": len(subjects), "n_links": n_links,
                                  "topology_labels": labels})

    if only in ("same_k", "all"):
        for k in ks:
            view = out_root / f"same_k{k}"
            view.mkdir(parents=True, exist_ok=True)
            for sid in index[k]:
                link(view / f"{sid}_GTready_rexpr{k}.npz", expr_dir / f"{sid}_rexpr_{k}.npz")
            record(view, "same_k", len(index[k]), [f"rexpr{k}"])

    if only in ("neutral_k", "all"):
        for k in ks:
            view = out_root / f"neutral_k{k}"
            view.mkdir(parents=True, exist_ok=True)
            for sid in index[k]:
                link(view / f"{sid}_GTready_rexpr{k}.npz", expr_dir / f"{sid}_rexpr_{k}.npz")
                link(view / f"{sid}_GTready_neutral.npz", neutral_dir / f"{sid}_GTready_original.npz")
            record(view, "expr_vs_neutral", 2 * len(index[k]), [f"rexpr{k}", "neutral"])

    if only in ("mixed", "all"):
        view = out_root / "mixed"
        view.mkdir(parents=True, exist_ok=True)
        n_links = 0
        for k in ks:
            for sid in index[k]:
                link(view / f"{sid}_GTready_rexpr{k}.npz", expr_dir / f"{sid}_rexpr_{k}.npz")
                n_links += 1
        record(view, "mixed", n_links, [f"rexpr{k}" for k in ks])

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[rexpr-views] {len(manifest['views'])} viste, manifest in {out_root / 'manifest.json'}")
    return out_root


def main() -> None:
    args = parse_args()
    subjects = read_heldout(args.ict_dir)
    print(f"[rexpr-views] held-out: {len(subjects)} soggetti ({subjects[0]}..{subjects[-1]})")
    build_views(args.ict_dir, subjects, args.only)


if __name__ == "__main__":
    main()
