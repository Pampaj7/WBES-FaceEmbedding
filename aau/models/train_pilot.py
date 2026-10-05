#!/usr/bin/env python3
"""Entrypoint di training del pilota pot: hook di aau/models/pilot_hooks.py + train_fast.py.

Flag propri (tutto il resto va a v2_work/fastio/train_fast.py, e da li' al trainer v1):
    --pilot-arm masked     pozzo + pooling ristretto alla ROI (braccio pot_m55). --data_dir
                           deve essere la cartella degli operatori col pozzo (con roi_mask);
                           --masked-pooling/--roi-threshold vengono passati a train_fast.
    --pilot-arm dual       modello a due rami (aau/models/dn_dual_ops.py). --data_dir resta
                           quella standard, --dual-ops-dir e' quella del pozzo.

    aau/run.sh aau/models/train_pilot.py --pilot-arm dual --dual-ops-dir <pot> --frame current \
        --data_dir <withops> ... (ricetta v1, vedi aau/train_pilot_pot.sbatch)

Il trainer di ricerca sotto face_embedding/ e train_fast.py non sono modificati.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "v2_work/fastio"))
sys.path.insert(0, str(THIS_DIR))

import train_fast  # noqa: E402  (mette sul path il pacchetto robustness e fast_data)


def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--pilot-arm", required=True, choices=["masked", "dual"])
    p.add_argument("--dual-ops-dir", type=Path, default=None)
    p.add_argument("--roi-threshold", type=float, default=0.5)
    known, rest = p.parse_known_args()
    if "--masked-pooling" in rest:
        raise SystemExit("--masked-pooling lo aggiunge --pilot-arm masked, non va passato")

    import robustness.train_runner  # noqa: F401  (carica data_utils/eval_utils/model_helpers)
    from pilot_hooks import install

    if known.pilot_arm == "dual":
        if known.dual_ops_dir is None:
            raise SystemExit("--pilot-arm dual vuole --dual-ops-dir")
        install(pot_dir=known.dual_ops_dir, dual=True)
    else:
        if known.dual_ops_dir is not None:
            raise SystemExit("--dual-ops-dir non ha senso con --pilot-arm masked")
        install(require_roi=True)
        rest = ["--masked-pooling", "--roi-threshold", str(known.roi_threshold)] + rest

    sys.argv = [sys.argv[0]] + rest
    train_fast.main()


if __name__ == "__main__":
    main()
