#!/usr/bin/env python3
"""Lancia uno script di eval con gli hook delle ablazioni v3 installati prima (come eval_pilot.py).

    aau/run.sh aau/models/eval_ablation.py --frame rms --size-token-json <tabella> -- \
        aau/models/eval_cells.py --checkpoint ... --data-dir <operatori del braccio> ...
    aau/run.sh aau/models/eval_ablation.py --frame rms --size-token-json <ict.json> -- \
        face_embedding/.../perturbated/compare_model_vs_chamfer_rankings.py ... --scenarios clean

--frame e' il frame di training del braccio: il forward agganciato ri-inquadra V_in, perche' gli
script del repo danno al modello i vertici del loader (maxabs). Valido solo su input puliti:
con --frame diverso da current gli scenari perturbati vengono rifiutati.
--size-token-json e' la tabella della collezione VALUTATA (BFM per l'in-domain, ICT per lo
zero-shot): i nomi dei file sono le chiavi.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "v2_work/fastio"))
sys.path.insert(0, str(THIS_DIR))


def main() -> None:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("uso: eval_ablation.py --frame {current,rms} [--size-token-json J] -- script.py [args]")
    cut = argv.index("--")
    p = argparse.ArgumentParser()
    p.add_argument("--frame", required=True, choices=["current", "rms"])
    p.add_argument("--size-token-json", type=Path, default=None)
    known = p.parse_args(argv[:cut])
    script, script_args = Path(argv[cut + 1]).resolve(), argv[cut + 2:]
    if known.frame != "current" and script.name == "compare_model_vs_chamfer_rankings.py":
        sc = script_args[script_args.index("--scenarios") + 1] if "--scenarios" in script_args else None
        if sc != "clean":
            raise SystemExit(f"--frame {known.frame} ri-inquadra V_in nel forward: valido solo sullo "
                             f"scenario clean, passare --scenarios clean (ora: {sc})")

    import train_fast  # noqa: F401  (path del pacchetto robustness)
    import robustness.model_helpers  # noqa: F401
    import robustness.data_utils  # noqa: F401
    from ablation_hooks import install

    install(size_token_json=known.size_token_json, eval_frame=known.frame)

    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script)] + script_args
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
