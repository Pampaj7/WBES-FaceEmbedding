#!/usr/bin/env python3
"""Entrypoint di training delle ablazioni v3 (B, C, E): hook di aau/models/ablation_hooks.py +
v2_work/fastio/train_fast.py.

Flag propri (tutto il resto va a train_fast.py, e da li' al trainer v1):
    --size-token-json J   token di taglia (B, E): tabella di aau/models/size_token.py. Il seed
                          della standardizzazione deve essere quello del run (--seed).
Il frame (--frame rms per B ed E) e gli operatori (--data_dir robusti ad area 1 per C ed E)
passano a train_fast.py cosi' come sono: per C senza token questo entrypoint non installa niente.

    aau/run.sh aau/models/train_ablation.py --size-token-json <bfm.json> --frame rms \
        --data_dir <withops> ... (ricetta v1, vedi aau/models/train_ablation.sbatch)

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
    p.add_argument("--size-token-json", type=Path, default=None)
    known, rest = p.parse_known_args()

    if known.size_token_json is not None:
        import robustness.train_runner  # noqa: F401  (carica data_utils/eval_utils/model_helpers)
        from ablation_hooks import SizeTokenTable, install

        table = SizeTokenTable(known.size_token_json)
        seed = int(rest[rest.index("--seed") + 1]) if "--seed" in rest else None
        if table.collection != "bfm" or table.seed is None or seed != int(table.seed):
            raise SystemExit(f"token standardizzato su {table.collection} seed {table.seed}, run seed "
                             f"{seed}: le statistiche devono venire dal training di QUESTO run")
        install(size_token_json=known.size_token_json)

    sys.argv = [sys.argv[0]] + rest
    train_fast.main()


if __name__ == "__main__":
    main()
