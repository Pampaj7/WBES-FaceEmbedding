#!/usr/bin/env python3
"""Prove dell'emendamento 2 di D3 (``d3_e2.py``) su dati finti, nessun dato valutato: perdita riscalata (alpha0 = 1 per
blocco, Spearman invariato), meta' di FLAME, griglia estesa.

    AAU_NV="" aau/run.sh aau/diagnostics/test_d3_e2.py          (d3_e2.sbatch, passo test)
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import d3_e2 as E  # noqa: E402
import d3_head as H  # noqa: E402
from test_d3 import planted  # noqa: E402


def test_rescale() -> None:
    rng = np.random.default_rng(1)
    a, b = planted(30, rng, name="a"), planted(30, rng, name="b")
    b = H.Block(3.0 * b.X, b.subj, b.lab, b.G, "b")              # scala di d_P diversa per sorgente
    rs = E.rescale([a, b])
    for x, y in zip([a, b], rs):
        assert abs(H.Problem([y]).alpha0 - 1.0) < 1e-12
        i, j = H.pair_index(x)
        assert spearmanr(x.G[x.subj[i], x.subj[j]], y.G[y.subj[i], y.subj[j]]).correlation > 1 - 1e-12
    h0, h1 = H.fit([a, b], 4, 1e-4), E.fit_e2([a, b], 4, 1e-4)
    assert np.isfinite(h0.W).all() and np.isfinite(h1.W).all()
    print(f"perdita riscalata: alpha0 = 1 per blocco, ranghi della GT invariati; alpha senza / con: {h0.alpha:.3f} / "
          f"{h1.alpha:.3f}", flush=True)


def test_half_grid() -> None:
    rng = np.random.default_rng(2)
    f = planted(200, rng, name="flame2023_s1")
    h = E.half(f)
    assert len(h.G) == 100 and set(np.unique(h.subj)) == set(range(100)) and len(h.X) == 500
    assert np.array_equal(h.G, f.G[:100, :100])
    assert len(E.CONFIGS_E2) == 43 and E.CONFIGS_E2[0] == (0, 0.0) and min(E.LAMBDAS_E2) == 1e-6
    print("meta' di FLAME (soggetti 0-99) e griglia estesa (43 configurazioni): ok", flush=True)


if __name__ == "__main__":
    test_rescale()
    test_half_grid()
    print("[test] tutte le prove passate", flush=True)
