#!/usr/bin/env python3
"""Riporta la GT estesa alla scala della GT in uso (decisione del PI, 5 ottobre sera).

    aau/run.sh aau/data_scale/rescale_gt.py --gt datasets/ICT_SCALE/gt_joint_bfm_ict.npz

``build_gt.py`` divide il blocco ICT per il proprio massimo (convenzione di ogni file GT del
repo); con 55.000 identita' il massimo sale da 0.1718 a 0.2038 e le coppie di ICT-5000 varrebbero
0.843 volte quelle del congiunto attuale, cambiando la scala della stress loss. Qui il blocco ICT
e' moltiplicato per ``ict_raw_max_new / ict_raw_max_old``, cioe' diviso per il massimo di
ICT-5000 come nella GT in uso: le coppie di ICT-5000 tornano IDENTICHE a quelle del congiunto
(controllato contro datasets/JOINT_BFM_ICT/gt_matrix.npz), quelle nuove possono superare 1
(massimo 1.186). Blocco BFM e NaN fuori blocco invariati.

ATTENZIONE: con massimo globale > 1, ``load_gt_distance_matrix`` dividerebbe TUTTO (BFM compreso)
per 1.186. Il trainer va lanciato con ``train_steps.py --gt-keep-scale``, che legge la matrice
senza quella divisione (sulla GT in uso, che ha massimo 1, le due letture coincidono).
Scrittura su file temporaneo e rename: il file e' o tutto vecchio o tutto nuovo.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
JOINT = REPO_ROOT / "datasets/JOINT_BFM_ICT/gt_matrix.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    a = ap.parse_args()
    man_p = a.gt.with_suffix(".json")
    man = json.loads(man_p.read_text())
    if man.get("scale") == "current":
        raise SystemExit(f"{a.gt} e' gia' alla scala attuale")
    f = float(man["ict_raw_max_new"]) / float(man["ict_raw_max_old"])
    with np.load(a.gt) as z:
        D = z["D_orig"]
        names = [str(n) for n in z["names"]]
    nb = int(man["n_bfm"])
    D[nb:, nb:] *= np.float32(f)

    with np.load(JOINT) as z:
        DJ = z["D_orig"]
        jn = [str(n) for n in z["names"]]
    pos = {n: i for i, n in enumerate(names)}
    idx_new = [pos[n] for n in jn]
    sub = D[np.ix_(idx_new, idx_new)].astype(np.float64)
    ref = DJ.astype(np.float64)
    fin = np.isfinite(ref)
    if not np.array_equal(fin, np.isfinite(sub)):
        raise SystemExit("i NaN fuori blocco non coincidono con il congiunto in uso")
    diff = float(np.abs(sub[fin] - ref[fin]).max())
    if diff > 1e-5:
        raise SystemExit(f"le coppie del congiunto in uso non tornano: max |diff| = {diff:.2e}")

    tmp = a.gt.with_name(f".{a.gt.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, D_orig=D, names=np.array(names))
    os.replace(tmp, a.gt)
    man.update({"scale": "current", "ict_block_factor_applied": f,
                "ict_block_max": float(np.nanmax(D[nb:, nb:])), "global_max": float(np.nanmax(D)),
                "joint_1019532_pairs_max_abs_diff": diff,
                "note": "blocco ICT diviso per il massimo di ICT-5000 (scala della GT in uso); "
                        "leggere con train_steps.py --gt-keep-scale"})
    man_p.write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({k: v for k, v in man.items() if k != "shards"}, indent=1))


if __name__ == "__main__":
    main()
