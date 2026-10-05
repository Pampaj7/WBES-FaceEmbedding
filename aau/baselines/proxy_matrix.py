#!/usr/bin/env python3
"""Matrici 100x100 della riga di CONTROLLO ``bbox_proxy``.

Non e' una baseline: e' il metro con cui si leggono tutte le altre righe della Tabella 2
estesa.  Di ogni mesh si tengono **quattro numeri** -- il centro del bounding box e la
lunghezza della sua diagonale -- e la distanza fra due mesh e' la distanza euclidea fra i
due vettori di quattro numeri.  Nessuna forma entra nel conto.  Se una metrica di forma
non batte questa riga, quella cella non sta misurando la forma.

I quattro numeri si calcolano sulle mesh **gia' normalizzate maxabs**
(``common.maxabs_normalize``), cioe' esattamente le stesse mesh che vedono Chamfer, i
render percettivi e -- dopo la correzione di questa revisione -- varifold e currents.  Sul
grezzo il proxy misurerebbe posizione e taglia assoluta nel frame del dataset, che e'
informazione che nessuna delle metriche vere riceve: sarebbe un controllo su un altro
esperimento.  Dopo la maxabs il centro del bbox e la diagonale non sono costanti -- la
normalizzazione centra sulla MEDIA dei vertici e divide per il massimo scarto assoluto, non
sul bbox -- e quel che resta e' la forma dell'ingombro, che e' il residuo su cui la riga di
controllo ha senso.

Gira in un soffio (4 numeri per mesh, 600 mesh) e non ha bisogno ne' di GPU ne' di worker:
non esiste un ``proxy.sbatch``, la riga si calcola dentro ``rank.sbatch``.

  aau/run.sh aau/baselines/proxy_matrix.py --subject-set heldout
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

METRIC = "bbox_proxy"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", type=Path, default=common.OUT_ROOT)
    p.add_argument("--settings", type=str, default=",".join(common.SETTINGS))
    p.add_argument("--normalize", type=str, default="maxabs", choices=("maxabs", "raw"),
                   help="maxabs = le stesse mesh delle metriche vere; raw = unita' del file")
    p.add_argument("--subject-set", type=str, default="heldout", choices=common.SUBJECT_SETS,
                   help="heldout = split del repo; facebench_first100 = i soggetti della Tabella 2")
    p.add_argument("--max-subjects", type=int, default=0, help="0 = tutti; >0 per un test rapido")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def bbox_features(subject: str, topology: str, normalize: str) -> np.ndarray:
    """Centro del bounding box e sua diagonale: quattro numeri, nessuna forma."""
    V = common.load_verts_faces(subject, topology)[0]
    if normalize == "maxabs":
        V = common.maxabs_normalize(V)
    lo, hi = V.min(axis=0), V.max(axis=0)
    return np.append((lo + hi) / 2.0, np.linalg.norm(hi - lo))


def main() -> None:
    args = parse_args()
    subjects = common.subject_set(args.subject_set)
    if args.max_subjects > 0:
        subjects = subjects[: args.max_subjects]
    n = len(subjects)
    pair_i, pair_j = common.subject_pair_indices(n)

    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    topology_pairs = common.all_topology_pairs(settings)
    topologies = sorted({t for pair in topology_pairs for t in pair})
    print(f"[proxy] soggetti={len(subjects)} topologie={topologies} "
          f"coppie-topologia={len(topology_pairs)} normalize={args.normalize}", flush=True)

    t0 = time.time()
    feats = {topology: np.stack([bbox_features(s, topology, args.normalize) for s in subjects])
             for topology in topologies}
    print(f"[proxy] {n * len(topologies)} mesh lette in {time.time() - t0:.0f}s", flush=True)
    for topology in topologies:
        f = feats[topology]
        print(f"[proxy] {topology:9s} diagonale {f[:, 3].mean():.4f} +- {f[:, 3].std():.4f}, "
              f"|centro| mediano {np.median(np.linalg.norm(f[:, :3], axis=1)):.4f}", flush=True)

    for topology_a, topology_b in topology_pairs:
        out_path = common.matrix_path(METRIC, topology_a, topology_b, args.out_root)
        if out_path.exists() and not args.overwrite:
            print(f"[proxy] {topology_a}->{topology_b}: gia' presente, salto", flush=True)
            continue
        values = np.linalg.norm(feats[topology_a][pair_i] - feats[topology_b][pair_j], axis=1)
        D = common.empty_matrix(n)
        D[pair_i, pair_j] = values
        common.save_matrix(out_path, D, subjects, METRIC, topology_a, topology_b,
                           normalize=args.normalize, n_features=4)
        print(f"[proxy] {topology_a}->{topology_b}: {len(values)} coppie -> {out_path}", flush=True)

    print(f"[proxy] fine in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
