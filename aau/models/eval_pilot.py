#!/usr/bin/env python3
"""Lancia uno script di eval del repo con gli hook del pilota pot installati prima.

    aau/run.sh aau/models/eval_pilot.py --pilot-arm dual --dual-ops-dir <pot> -- \
        v2_work/potential/eval_by_topology.py --checkpoint ... --data-dir <withops> ...
    aau/run.sh aau/models/eval_pilot.py --pilot-arm masked -- \
        face_embedding/.../perturbated/compare_model_vs_chamfer_topology_breakdown.py ...

Lo script gira come __main__ (runpy) con la sua directory in testa a sys.path, come da riga
di comando; i nomi che importa da robustness.* (GTReadyDataset, sample_to_device,
build_model, forward_model) sono gia' quelli agganciati. Per il braccio masked NON va passato
--masked-pooling allo script: il pooling mascherato lo installa questo launcher.
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
        raise SystemExit("uso: eval_pilot.py --pilot-arm {masked,dual} [--dual-ops-dir D] -- script.py [args]")
    cut = argv.index("--")
    p = argparse.ArgumentParser()
    p.add_argument("--pilot-arm", required=True, choices=["masked", "dual"])
    p.add_argument("--dual-ops-dir", type=Path, default=None)
    p.add_argument("--roi-threshold", type=float, default=0.5)
    known = p.parse_args(argv[:cut])
    script, script_args = Path(argv[cut + 1]).resolve(), argv[cut + 2:]
    if "--masked-pooling" in script_args:
        raise SystemExit("--masked-pooling allo script installerebbe il pooling due volte: lo fa il launcher")

    import train_fast  # noqa: F401  (path del pacchetto robustness)
    import robustness.model_helpers  # noqa: F401
    import robustness.data_utils  # noqa: F401
    from pilot_hooks import install

    if known.pilot_arm == "dual":
        if known.dual_ops_dir is None:
            raise SystemExit("--pilot-arm dual vuole --dual-ops-dir")
        install(pot_dir=known.dual_ops_dir, dual=True)
    else:
        install(require_roi=True)
        train_fast.install_masked_pooling(float(known.roi_threshold))

    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script)] + script_args
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
