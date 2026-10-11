#!/usr/bin/env python3
"""Tabella di scala per mesh delle viste di Ava-256, per l'ingresso a scala globale del modello (``--input-norm global``,
``v3_work/trainer/global_v3.py``), con la STESSA definizione di ``v3_work/trainer/tools/build_scale_table.py``:

    area_mm2 = u_d^2 * area(V grezza)          (fattore al servizio sqrt(area_mm2 / area(V_loader)), poi R_d)

    aau/run.sh aau/ava256/ava_scale_table.py          (dopo ava_views.py view; prima di ava_freeze.py)

``build_scale_table.py`` legge u_d e R_d solo dai frame di E12 (``aau/runs/evidence/e12/frames.json`` e
``global_v3.EXTRA_FRAMES``) e NON si modifica: qui le sue funzioni (``_work``: area totale e maxabs della geometria grezza,
stessa lettura delle viste) si importano, e il frame di Ava-256 viene da ``datasets/AVA256/gt/frame.json`` (u = 1 mm, R e
t calcolati come ``frames.json`` di E12, ava_gt.py). Stesse uscite e stesso json: ``names`` (nomi dei file della vista),
``domain``, ``area_mm2``, ``area_raw``, ``maxabs_raw``, ``check`` (NaN: viste di sola geometria, niente da verificare).

CONFERMATIVO: e' una proprieta' delle mesh (area), non una valutazione; la tabella si congela con le viste.
Uscite: ``datasets/AVA256/scale/{eval_view,calib_view}.npz`` + ``.json``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
for _p in (THIS_DIR, REPO_ROOT / "v3_work" / "trainer" / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ava_common as ac  # noqa: E402
import build_scale_table as bst  # noqa: E402  (sola lettura: _work, total_area)

OUT_DIR = ac.DATA_ROOT / "scale"
DOMAIN = "ava256"


def table(view_dir: Path, frame: dict, out: Path) -> dict:
    t0 = time.time()
    names = sorted(p.name for p in view_dir.glob("*.npz"))
    rows = bst._work(([(n, "view", str(view_dir / n)) for n in names], None))
    rows.sort(key=lambda r: r[0])
    if [r[0] for r in rows] != names:
        raise SystemExit(f"{view_dir}: mesh mancanti nelle righe")
    area_raw = np.asarray([r[1] for r in rows])
    area_mm2 = float(frame["u"]) ** 2 * area_raw
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, names=np.asarray(names), domain=np.asarray([DOMAIN] * len(names)), area_mm2=area_mm2,
             area_raw=area_raw, maxabs_raw=np.asarray([r[2] for r in rows]), check=np.asarray([r[3] for r in rows]))
    by = defaultdict(list)
    for n, v in zip(names, np.sqrt(area_mm2)):
        by[n[:-4].split("_GTready_", 1)[1]].append(v)
    info = {"view_dir": str(view_dir.relative_to(ac.REPO_ROOT)), "domain": DOMAIN, "n_meshes": len(names),
            "frames": "datasets/AVA256/gt/frame.json (ava_gt.py)",
            "domains": {DOMAIN: {"u": frame["u"], "R": frame["R"], "unit_source": frame["unit_source"]}},
            "definition": "area_mm2 = u_d^2 * area totale della geometria grezza (unita' native); fattore al servizio "
                          "sqrt(area_mm2 / area(V_loader)), poi rotazione R_d (global_v3.py); definizione di "
                          "v3_work/trainer/tools/build_scale_table.py, funzioni importate",
            "check_maxabs_vs_view": {"n_checked": 0, "max_abs": None},
            "sqrt_area_mm_median": {f"{DOMAIN}|{lab}": float(np.median(v)) for lab, v in sorted(by.items())},
            "seconds": time.time() - t0, "host": os.uname().nodename}
    out.with_suffix(".json").write_text(json.dumps(info, indent=1))
    return info


def main() -> None:
    if ac.FROZEN.exists():
        raise SystemExit(f"{ac.FROZEN}: viste congelate, non riscrivo la tabella di scala")
    frame = json.loads((ac.GT_DIR / "frame.json").read_text())
    for kind in ("eval_view", "calib_view"):
        info = table(ac.DATA_ROOT / kind / "npz", frame, OUT_DIR / f"{kind}.npz")
        print(f"[ava-scale] {kind}: {info['n_meshes']} mesh, sqrt(area) mediana {info['sqrt_area_mm_median']}", flush=True)


if __name__ == "__main__":
    main()
