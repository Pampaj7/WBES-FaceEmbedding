#!/usr/bin/env python3
"""Prova di ``d4_stats.py`` su dati finti, prima degli embedding canonici: "canonico" = gli store ufficiali (link),
uscite in una cartella temporanea, 50 repliche. Attese: delta canonico - originale = 0 ovunque (punto e IC), residui
canonico = originale, K1 passa (stesse righe e valori di D3), verdetto NO, results_d4.md scritto.

    UGT_THREADS=1 v3_work/unified_gt/run.sh aau/diagnostics/d4_test.py <cartella temporanea> --workers 16
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import d4_stats as S


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("tmp", type=Path)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    real_ev, real_d4 = S.EV, S.D4
    shutil.rmtree(a.tmp, ignore_errors=True)
    (a.tmp / "d4").mkdir(parents=True)
    for f in ("d3_spearman.csv", "PROTOCOL_D4.sha256"):
        shutil.copy(real_ev / f, a.tmp / f)
    rs = json.loads((real_d4 / "remesh_stats.json").read_text())
    rs.setdefault("K4_validity", {v: {"meshes": 600, "expected": 600, "valid": 600, "invalid": [], "fraction_invalid": 0.0,
                                      "pass": True} for v in S.VIEWS})
    (a.tmp / "d4" / "remesh_stats.json").write_text(json.dumps(rs))
    for view in S.VIEWS:
        for pre, v, e in S.R.ARMS:
            d = a.tmp / "d4" / "emb" / view / pre
            d.mkdir(parents=True)
            (d / "embeddings.npz").symlink_to(S.R.bp_e3.store(view, v, e))
    S.EV, S.D4, S.R.D4 = a.tmp, a.tmp / "d4", a.tmp / "d4"
    S.diag.N_BOOT = 50
    S.run(a.workers)
    S.summary()
    Dl = pd.read_csv(a.tmp / "d4_delta.csv")
    x = Dl[Dl["contrast"].isin(["canonico - originale", "residuo canonico - residuo originale"])]
    err = float(np.nanmax(np.abs(x[["delta", "ci_low", "ci_high"]].to_numpy())))
    C = json.loads((a.tmp / "d4" / "controls.json").read_text())
    Rd = json.loads((a.tmp / "d4" / "readings.json").read_text())
    ok = err == 0.0 and C["K1"]["pass"] and Rd["verdict"] == "NO" and (a.tmp / "results_d4.md").exists()
    print(f"[d4-test] max |delta finto| {err:.1e}, K1 {C['K1']['pass']} (scarto {C['K1']['max_abs_diff']:.1e}, righe "
          f"{C['K1']['rows_equal']}), K5 {C['K5']}, verdetto {Rd['verdict']}: {'OK' if ok else 'FALLITA'}", flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
