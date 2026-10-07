#!/usr/bin/env python3
"""``zs_arcface_render.py`` su un insieme del riconoscimento in dominio (etichette di ``sets.json``).

    aau/baselines/run_bl.sh aau/indomain/ir_arcface_render.py rexpr --view-dir <vista> \\
        --subjects-json aau/runs/indomain_recog/subjects/rexpr.json --out-root <dir> --base-rotation x180 ...
    (ir_arcface.sbatch)

Il primo argomento e' l'insieme; il resto va com'e' a ``zs_arcface_render.main()``, importato e
non riscritto. L'unica differenza: ``TOPOLOGIES`` del modulo diventa la lista di etichette
dell'insieme (per ``rexpr``: neutral, rexpr1..5 invece delle 6 topologie), cosi' render, crop,
png di controllo ed ``arcface_views.npz`` coprono tutte e sole le mesh dell'insieme.
``--frame-check`` disegna la topologia ``original``: si usa su bfm e ict, non su rexpr (stesso
frame di ict).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
sys.path.insert(0, str(AAU_DIR / "zs3dmm"))

import zs_arcface_render as zar  # noqa: E402

SETS_JSON = AAU_DIR / "runs" / "indomain_recog" / "sets.json"


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("uso: ir_arcface_render.py <insieme> [argomenti di zs_arcface_render.py]")
    name = sys.argv.pop(1)
    labels = tuple(json.loads(SETS_JSON.read_text())["sets"][name]["labels"])
    zar.TOPOLOGIES = labels
    print(f"[ir-arcface] insieme {name}, etichette {zar.TOPOLOGIES}", flush=True)
    if "original" not in labels:
        # control_pngs disegna le viste della topologia "original", che rexpr non ha: la riga
        # "<soggetto>__original_views.png" mostra la prima etichetta (neutral).
        control_pngs, load_render = zar.control_pngs, zar.perc.load_render

        def control_first_label(args, subjects, yaws, calibration):
            zar.perc.load_render = lambda root, t, s, y: load_render(root, labels[0] if t == "original" else t, s, y)
            try:
                control_pngs(args, subjects, yaws, calibration)
            finally:
                zar.perc.load_render = load_render

        zar.control_pngs = control_first_label
    zar.main()


if __name__ == "__main__":
    main()
