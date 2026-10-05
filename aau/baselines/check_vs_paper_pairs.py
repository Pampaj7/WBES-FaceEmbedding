#!/usr/bin/env python3
"""Confronta le matrici prodotte qui con i valori pair-level gia' nel repo.

Verifica indipendente dal ranking: le colonne ``raw_chamfer`` di
``paper_artifacts/bootstrap_ci/table1_pairlevel_exact/<tA>__to__<tB>/pair_metrics.csv``
sono state calcolate dall'eval del repo su queste stesse coppie, quindi
``matrices/chamfer_sq`` (variante eval_utils) deve coincidere valore per valore.  Se
coincide, caricamento delle mesh, normalizzazione e ordinamento delle coppie sono giusti,
e un eventuale scarto nello Spearman viene da altro.

  aau/run_baselines.sh aau/baselines/check_vs_paper_pairs.py --metric chamfer_sq
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", type=Path, default=common.OUT_ROOT)
    p.add_argument("--metric", type=str, default="chamfer_sq")
    p.add_argument("--column", type=str, default="raw_chamfer")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    matrix_dir = args.out_root / "matrices" / args.metric
    if not matrix_dir.is_dir():
        raise SystemExit(f"nessuna matrice in {matrix_dir}")

    n_checked = 0
    for path in sorted(matrix_dir.glob("*.npz")):
        D, subjects, _, topology_a, topology_b = common.load_matrix(path)
        csv_path = common.PAIR_TABLE_ROOT / f"{topology_a}__to__{topology_b}" / "pair_metrics.csv"
        if not csv_path.exists():
            print(f"[check] {topology_a}->{topology_b}: nessun pair_metrics.csv nel repo, salto")
            continue
        index = {s: i for i, s in enumerate(subjects)}
        mine, theirs = [], []
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                if row["subject_a"] not in index or row["subject_b"] not in index:
                    continue
                mine.append(D[index[row["subject_a"]], index[row["subject_b"]]])
                theirs.append(float(row[args.column]))
        mine = np.asarray(mine, dtype=np.float64)
        theirs = np.asarray(theirs, dtype=np.float64)
        mask = np.isfinite(mine) & np.isfinite(theirs)
        rel = np.abs(mine[mask] - theirs[mask]) / np.maximum(np.abs(theirs[mask]), 1e-12)
        print(f"[check] {topology_a}->{topology_b}: n={int(mask.sum())} "
              f"errore relativo max={rel.max():.2e} mediano={np.median(rel):.2e} "
              f"| range mio=[{mine[mask].min():.3e}, {mine[mask].max():.3e}] "
              f"loro=[{theirs[mask].min():.3e}, {theirs[mask].max():.3e}]", flush=True)
        n_checked += 1

    if n_checked == 0:
        raise SystemExit("[check] nessuna coppia confrontabile")


if __name__ == "__main__":
    main()
