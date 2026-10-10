#!/usr/bin/env python3
"""Prove della diagnostica D (nessun dato valutato): Spearman pesato contro le righe ripetute, righe e nomi degli
insiemi, sonda di D2 su dati finti (recupero esatto, permutazione).

    AAU_NV="" aau/run.sh aau/diagnostics/test_diag.py          (diag.sbatch, passo test)
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

import diag


def test_wspearman() -> None:
    rng = np.random.default_rng(0)
    for n in (5, 50, 2000):
        x = np.round(rng.normal(size=n), 1)          # pari merito voluti
        y = x + rng.normal(size=n)
        w = rng.integers(0, 4, n)
        k = w > 0
        xr, yr = np.repeat(x[k], w[k]), np.repeat(y[k], w[k])
        ref = np.corrcoef(rankdata(xr), rankdata(yr))[0, 1]
        got = diag.wspearman(x, y, w)
        assert abs(ref - got) < 1e-12, (n, ref, got)
        assert abs(diag.wspearman(x, y, np.ones(n, int)) - spearmanr(x, y).correlation) < 1e-12
    print("wspearman: uguale al campione con righe ripetute e a scipy", flush=True)


def test_sets() -> None:
    for s in diag.SETS:
        names = diag.set_names(s)
        n_subj = len({diag.split_name(x)[0] for x in names})
        i, j = diag.rows(names)
        expect = n_subj * (n_subj - 1) // 2 * len(diag.LABELS) * (len(diag.LABELS) - 1)
        assert len(names) == n_subj * len(diag.LABELS) and len(i) == expect, (s, len(names), len(i), expect)
        print(f"{s}: {n_subj} soggetti, {len(names)} mesh, {len(i)} righe", flush=True)
    a, b = diag.set_names("flame2023"), diag.set_names("flame2023_s1")
    ia, sa = diag.subject_index(a)
    ib, sb = diag.subject_index(b)
    assert np.array_equal(ia, ib) and [int(x[2:]) - 700000 for x in sa] == [int(x[2:]) - 710000 for x in sb]
    assert diag.set_names("ict") == diag.set_names("regen_ict") and diag.set_names("gnm") == diag.set_names("regen_gnm")
    c = diag.boot_counts(100, 1, n_boot=3)
    assert (c[0] == 1).all() and (c.sum(1) == 100).all() and np.array_equal(c, diag.boot_counts(100, 1, n_boot=3))
    print("insiemi, righe, ordine dei soggetti e conteggi: ok", flush=True)


def test_probe() -> None:
    sys.argv = [sys.argv[0]]
    import d2_probe as dp
    rng = np.random.default_rng(1)
    n, nl, d = 60, 5, 40
    T = rng.normal(size=(n, 8)) @ rng.normal(size=(8, d))           # bersagli per soggetto, rango 8
    ms = np.repeat(np.arange(n), nl)
    X = np.c_[T[ms] @ rng.normal(size=(d, 30)), rng.normal(size=(len(ms), 5))]   # lineare nei bersagli + rumore
    sa, sb = np.triu_indices(n, 1)
    ma, mb = sa * nl, sb * nl + 1
    gt = np.linalg.norm(T[sa] - T[sb], axis=1)
    D = {"sa": sa, "sb": sb, "ma": ma, "mb": mb, "n_subj": n}
    tr, te = np.arange(0, n, 2), np.arange(1, n, 2)
    c = np.ones(n, dtype=np.int64)
    P, lam = dp.fit_predict(X, ms, T, tr, c, np.random.default_rng(2))
    rp, rr = dp.evaluate(P, D, gt, gt, te, c)
    assert rp > 0.99 and abs(rr - 1.0) < 1e-12, (rp, rr, lam)
    # bersagli permutati: la sonda resta una mappa lineare dell'ingresso e ne conserva in parte le distanze, quindi
    # rho NON va a 0 (emendamento 1): deve solo stare sotto la sonda vera
    P0, _ = dp.fit_predict(X, ms, T, tr, c, np.random.default_rng(2), perm=np.random.default_rng(3).permutation(tr))
    r0, _ = dp.evaluate(P0, D, gt, gt, te, c)
    assert r0 < rp, (r0, rp)
    # nessuna fuga: con un ingresso di solo rumore la sonda non ordina i soggetti di test (200 soggetti, media su 10
    # estrazioni: con 30 soggetti di test una sola estrazione ha errore standard ~0.18 per la dipendenza fra coppie)
    n2 = 200
    T2 = np.random.default_rng(7).normal(size=(n2, 8)) @ np.random.default_rng(8).normal(size=(8, d))
    ms2 = np.repeat(np.arange(n2), nl)
    s2a, s2b = np.triu_indices(n2, 1)
    D2 = {"sa": s2a, "sb": s2b, "ma": s2a * nl, "mb": s2b * nl + 1, "n_subj": n2}
    g2 = np.linalg.norm(T2[s2a] - T2[s2b], axis=1)
    c2_ = np.ones(n2, dtype=np.int64)
    rns = []
    for q in range(10):
        Xn = np.random.default_rng(100 + q).normal(size=(len(ms2), 35))
        Pn, _ = dp.fit_predict(Xn, ms2, T2, np.arange(0, n2, 2), c2_, np.random.default_rng(q))
        rns.append(dp.evaluate(Pn, D2, g2, g2, np.arange(1, n2, 2), c2_)[0])
    rn = float(np.mean(rns))
    assert abs(rn) < 0.1, rns
    c2 = np.random.default_rng(4).integers(0, 3, n)
    tr2 = np.flatnonzero(c2 > 0)[::2]
    P2, _ = dp.fit_predict(X, ms, T, tr2, c2, np.random.default_rng(5))
    assert np.isfinite(P2).all()
    print(f"sonda: recupero rho {rp:.4f} (lambda {lam:g}), permutata {r0:+.3f}, ingresso di rumore {rn:+.3f} (media di 10), "
          f"pesi bootstrap ok", flush=True)


if __name__ == "__main__":
    test_wspearman()
    test_sets()
    test_probe()
    print("[test] tutte le prove passate", flush=True)
