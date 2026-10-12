#!/usr/bin/env python3
"""K5 di G0 (``PROTOCOL_G0.md`` sez. 5), prima dei numeri veri: ``g0_stats.py`` con l'insegnante finto = d_P di
factorized s1234 (c_T = c del braccio), su tutti gli insiemi e le 1000 repliche. Attese: delta insegnante - d_P s1234 e
combinazione s1234 - d_F cal. s1234 nulli su ogni replica (K5), K1 passa (bracci, B e ICP mm = valori pubblicati), K6
passa, verdetto NO (delta 0 < +0.05). Uscite in ``g0/test`` (nessun numero dell'insegnante).

    UGT_THREADS=1 v3_work/unified_gt/run.sh aau/diagnostics/g0_test.py --workers 64       (g0.sbatch, passo test)
"""
from __future__ import annotations

import argparse
import json

import g0_stats as G


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args()
    G.FAKE, G.OUT_DIR = True, G.G0 / "test"
    G.run(a.workers)
    C = json.loads((G.OUT_DIR / "controls.json").read_text())
    Rd = json.loads((G.OUT_DIR / "readings.json").read_text())
    ok = C["K5"]["pass"] and C["K1"]["pass"] and C["c_T"]["K6"]["pass"] and Rd["verdict"] == "NO"
    print(f"[g0-test] K5 {C['K5']['pass']} (scarto {C['K5']['max_abs_diff']:.1e} su {len(C['K5']['cells'])} celle), "
          f"K1 {C['K1']['pass']} ({C['K1']['n_cells']} celle, scarto {C['K1']['max_abs_diff']:.1e}), K6 "
          f"{C['c_T']['K6']['pass']}, c_T finta {C['c_T']['c_T']:.4f}, verdetto {Rd['verdict']}: "
          f"{'OK' if ok else 'FALLITA'}", flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
