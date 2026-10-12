#!/usr/bin/env python3
"""Prove di D5 (``d5_fewshot.py``) su dati finti, nessun dato valutato (PROTOCOL_D5.md sez. 5, Kw e K6): perdita pesata
con pesi uguali = ``d3_head.Problem``, gradiente con pesi diversi, estrazioni annidate e deterministiche, teste da fare,
blocchi di training nell'ordine delle sorgenti, un fit EQ e uno W da capo a fondo.

    AAU_NV="" aau/run.sh aau/diagnostics/test_d5.py          (d5.sbatch, passo test)
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import d3_e2 as E2  # noqa: E402
import d3_head as H  # noqa: E402
import d3_stats as S  # noqa: E402
import d5_fewshot as D  # noqa: E402
from test_d3 import planted  # noqa: E402


def test_weights() -> None:
    rng = np.random.default_rng(1)
    blocks = [planted(9, rng, p=12, k=3, name="a"), planted(7, rng, p=12, k=3, name="b"),
              planted(5, rng, p=12, k=3, name="c")]
    P, Q = H.Problem(blocks), D.WProblem(blocks, [1.0, 1.0, 1.0])
    th = np.concatenate([[np.log(P.alpha0) + 0.1], 0.3 * rng.normal(size=3 * P.p)])
    lp, gp = P.loss(th, 3, 0.05)
    lq, gq = Q.loss(th, 3, 0.05)
    assert abs(P.alpha0 - Q.alpha0) < 1e-12 and abs(P.Z - Q.Z) < 1e-12 * P.Z
    assert abs(lp - lq) < 1e-12 and np.abs(gp - gq).max() < 1e-10
    W = D.WProblem(blocks, [0.25, 0.25, 0.5])
    assert abs(sum(O.sum() for O in W.Om) - 2.0) < 1e-12                 # coppie ordinate: ognuna due volte
    assert abs(W.Om[2].sum() - 1.0) < 1e-12 and abs(W.Om[0].sum() - 0.5) < 1e-12
    _, g = W.loss(th, 3, 0.05)
    num = np.zeros_like(th)
    for i in range(len(th)):
        e = np.zeros_like(th)
        e[i] = 1e-6
        num[i] = (W.loss(th + e, 3, 0.05)[0] - W.loss(th - e, 3, 0.05)[0]) / 2e-6
    err = float(np.abs(num - g).max() / np.abs(num).max())
    assert err < 1e-5, err
    h0, h1 = H.fit(blocks, 3, 1e-3), D.fit_w(blocks, [1, 1, 1], 3, 1e-3)
    assert abs(h0.alpha - h1.alpha) < 1e-8 and np.abs(h0.W - h1.W).max() < 1e-8
    print(f"Kw: pesi uguali = Problem (perdita, gradiente, alpha0, fit); pesi 1/4, 1/4, 1/2: massa per blocco giusta, "
          f"gradiente errore relativo {err:.1e}", flush=True)


def test_draws() -> None:
    assert D.n_pool("flame2023_s1") == 100 and all(D.n_pool(t) == 200 for t in S.POOLS)
    assert D.ks_of("flame2023_s1") == [0, 5, 10, 20, 50, 100] and D.ks_of("hifi3d") == list(D.KS)
    k6 = D.check_draws()
    assert k6["pass"]
    for t in S.TARGETS:
        assert np.array_equal(D.drawn(t, 20, 3), D.drawn(t, 20, 3))
        assert not np.array_equal(D.drawn(t, 20, 3), D.drawn(t, 20, 4))
        assert len(D.drawn(t, 0, 0)) == 0 and len(D.drawn(t, D.n_pool(t), 0)) == D.n_pool(t)
    assert not np.array_equal(D.drawn("facescape", 20, 0), D.drawn("hifi3d", 20, 0))
    fl = D.pool_subjects("flame2023_s1")
    assert fl[0] == "id710000" and fl[-1] == "id710099"
    ps = D.pool_subjects("facescape")
    assert set(ps) == set(S.pool_info("facescape")["source"]) and ps == sorted(ps, key=lambda s: int(s[2:]))
    tk = D.tasks()
    per_arm = 3 * (1 + 5 * 10 * 2 + 2) + (1 + 4 * 10 * 2 + 2)
    assert len(tk) == len(set(tk)) == len(D.ARMS) * per_arm, len(tk)
    print(f"K6 sulle estrazioni: annidate, deterministiche, dentro il pool, fuori dai valutati; {len(tk)} teste "
          f"({per_arm} per braccio)", flush=True)


def test_blocks_and_fit() -> None:
    rng = np.random.default_rng(3)
    arm = D.ARMS[0]
    for s in S.SOURCES:
        S._B[(arm, s)] = planted(200 if s in S.TARGETS else 60, rng, name=s)
    for t in S.TARGETS:
        D._CFG[(arm, t)] = (4, 1e-4)
    bl, pos = D.train_blocks(arm, "hifi3d", 0, 0)
    assert pos == -1 and [b.name for b in bl] == list(S.settings()["hifi3d|all"])
    bl, pos = D.train_blocks(arm, "hifi3d", 10, 2)
    assert [b.name for b in bl] == list(S.SOURCES) and bl[pos].name == "hifi3d"
    assert len(np.unique(bl[pos].subj)) == 10 and len(bl[pos].X) == 50
    assert np.array_equal(np.unique(bl[pos].subj), D.drawn("hifi3d", 10, 2))
    bl, pos = D.train_blocks(arm, "flame2023_s1", 50, 1)
    assert bl[pos].subj.max() < 100
    with tempfile.TemporaryDirectory() as tmp:
        D.HEADS = Path(tmp)
        out = {}
        for var in ("eq", "w"):
            D._fit((arm, "facescape", 10, 0, var))
            out[var] = D.load_head(arm, "facescape", 10, 0, var)
        D._fit((arm, "facescape", 0, 0, "eq"))
        h0, _, m0 = D.load_head(arm, "facescape", 0, 0, "eq")
    ref = E2.fit_e2(D.train_blocks(arm, "facescape", 0, 0)[0], 4, 1e-4)
    assert abs(h0.alpha - ref.alpha) < 1e-12 and np.abs(h0.W - ref.W).max() < 1e-12 and m0["n_target"] == 0
    (he, ce, me), (hw, cw, mw) = out["eq"], out["w"]
    assert me["n_target"] == mw["n_target"] == 10 and np.isfinite([ce, cw]).all()
    assert np.abs(he.W - hw.W).max() > 1e-6                              # la variante W cambia la testa
    tb = S._B[(arm, "facescape")]
    rest = tb.take(np.setdiff1d(np.arange(200), D.drawn("facescape", 10, 0)))
    se, sw = H.score(he, [rest]), H.score(hw, [rest])
    print(f"blocchi in ordine di SOURCES, k soggetti x 5 etichette, FLAME dal pool 0-99; k = 0 = fit_e2 del LOGO; "
          f"Spearman sui soggetti finti non estratti EQ {se:.3f}, W {sw:.3f}", flush=True)


if __name__ == "__main__":
    test_weights()
    test_draws()
    test_blocks_and_fit()
    print("[test] tutte le prove passate", flush=True)
