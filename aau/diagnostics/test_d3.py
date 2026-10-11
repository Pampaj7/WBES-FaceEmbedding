#!/usr/bin/env python3
"""Prove di D3 (``d3_head.py``) su dati finti, nessun dato valutato (PROTOCOL_D3.md sez. 7, K5; emendamento 1):
gradiente, recupero di una metrica piantata, r = 0 come d_P, Mahalanobis intra-soggetto, Ledoit-Wolf contro sklearn,
CORAL, mediana pesata, fold per sorgente, scelta, c.

    AAU_NV="" aau/run.sh aau/diagnostics/test_d3.py          (d3.sbatch, passo test)
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import d3_head as H  # noqa: E402


def planted(n_subj: int, rng: np.random.Generator, p: int = 40, k: int = 4, nuis: float = 3.0,
            noise: float = 0.05, name: str = "x") -> H.Block:
    """Soggetti con forma latente t (k dimensioni, la GT) e un disturbo per soggetto ``nuis`` volte piu' grande nel
    complemento ortogonale (che d_P vede e la GT no), 5 etichette per soggetto con rumore di mesh."""
    Q = np.linalg.qr(np.random.default_rng(0).normal(size=(p, p)))[0]
    T = rng.normal(size=(n_subj, k))
    N = nuis * rng.normal(size=(n_subj, p - k))
    subj = np.repeat(np.arange(n_subj), 5)
    lab = np.tile(np.arange(5), n_subj)
    X = np.c_[T, N][subj] @ Q.T + noise * rng.normal(size=(len(subj), p))
    G = np.linalg.norm(T[:, None] - T[None], axis=-1)
    return H.Block(X, subj, lab, G, name)


def test_grad() -> None:
    rng = np.random.default_rng(1)
    blocks = [planted(9, rng, p=12, k=3, name="a"), planted(7, rng, p=12, k=3, name="b")]
    P = H.Problem(blocks)
    r, lam = 3, 0.05
    th = np.concatenate([[np.log(P.alpha0) + 0.1], 0.3 * rng.normal(size=r * P.p)])
    _, g = P.loss(th, r, lam)
    num = np.zeros_like(th)
    for i in range(len(th)):
        e = np.zeros_like(th)
        e[i] = 1e-6
        num[i] = (P.loss(th + e, r, lam)[0] - P.loss(th - e, r, lam)[0]) / 2e-6
    err = float(np.abs(num - g).max() / np.abs(num).max())
    assert err < 1e-5, err
    print(f"gradiente: errore relativo massimo {err:.1e} contro le differenze finite", flush=True)


def test_recovery() -> None:
    rng = np.random.default_rng(2)
    blocks = [planted(80, rng, name="a"), planted(80, rng, name="b")]
    fo = H.folds([80, 80], 0, [0, 1])
    fit_b, val_b = H.split(blocks, fo, 0)
    dP = H.score(H.Head(1.0, np.zeros((0, 40)), 0, 0.0, 1.0), val_b)
    t0 = time.time()
    head = H.fit(fit_b, 8, 1e-4)
    s = H.score(head, val_b)
    assert s > 0.85 and s > dP + 0.3, (s, dP, head.nit, head.success)
    strong = H.score(H.fit(fit_b, 8, 1.0), val_b)              # cresta forte: verso d_P
    assert abs(strong - dP) < abs(s - dP), (strong, dP, s)
    r0 = H.fit(fit_b, 0, 0.0)
    assert r0.r == 0 and abs(H.score(r0, val_b) - dP) < 1e-12
    print(f"metrica piantata: testa r=8 {s:.3f} ({head.nit} iterazioni, {time.time() - t0:.1f}s), d_P {dP:.3f}, "
          f"lambda=1 {strong:.3f}; r=0 = d_P", flush=True)


def test_lw_coral() -> None:
    from sklearn.covariance import ledoit_wolf
    rng = np.random.default_rng(3)
    X = rng.normal(size=(500, 64)) @ rng.normal(size=(64, 64)) + 2.0
    S, k = H.ledoit_wolf(X)
    S2, k2 = ledoit_wolf(X)
    assert abs(k - k2) < 1e-12 and np.abs(S - S2).max() < 1e-10 * np.abs(S2).max(), (k, k2)
    A = H.coral(S, S)
    assert np.abs(A - np.eye(64)).max() < 1e-8
    S_ref, _ = H.ledoit_wolf(rng.normal(size=(500, 64)) * np.linspace(0.5, 2.0, 64))
    A = H.coral(S_ref, S)
    assert np.abs(A @ S @ A.T - S_ref).max() < 1e-8 * np.abs(S_ref).max()
    Wh = H.whiten(S)
    assert np.abs(Wh @ S @ Wh.T - np.eye(64)).max() < 1e-8
    print(f"Ledoit-Wolf = sklearn (coefficiente {k:.4f}); CORAL porta la covarianza su quella di riferimento", flush=True)


def test_small() -> None:
    rng = np.random.default_rng(4)
    for n in (7, 8):
        x = rng.normal(size=n)
        assert H.wmedian(x, np.ones(n)) == np.median(x)
    assert H.wmedian(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 5.0])) == 3.0
    a = H.folds([100, 100, 100], 1, [0, 1, 2])
    b = H.folds([100, 100, 100, 200], 1, [0, 1, 2, 3])
    assert all(np.array_equal(x, y) for x, y in zip(a, b[:3])) and all(np.bincount(x).tolist() == [20] * 5 for x in a)
    assert np.array_equal(H.folds([100, 50], 1, [5, 6])[0], H.folds([100], 1, [5])[0])      # fold per sorgente
    blocks = [planted(20, rng, name="a")]
    fit_b, val_b = H.split(blocks, H.folds([20], 0, [0]), 2)
    assert not set(fit_b[0].subj) & set(val_b[0].subj) and len(set(val_b[0].subj)) == 4
    assert H.choose({(0, 0.0): 0.5, (4, 0.01): 0.6, (2, 0.1): 0.5995, (2, 1.0): 0.5992}) == (2, 1.0)
    c = H.calib_c(lambda b, i, j: 2.0 * b.G[b.subj[i], b.subj[j]], blocks)
    assert abs(c - 0.5) < 1e-12
    i, j = H.pair_index(blocks[0])
    assert np.all(blocks[0].subj[i] != blocks[0].subj[j]) and np.all(blocks[0].lab[i] != blocks[0].lab[j])
    assert len(i) == 20 * 19 // 2 * 20
    print("mediana pesata, fold per sorgente (indipendenti dagli altri blocchi), split disgiunto, scelta, c, coppie: ok",
          flush=True)


def test_within() -> None:
    """Disturbo per MESH (discretizzazione) grande in poche direzioni fuori dal sottospazio della forma: la Mahalanobis
    intra-soggetto lo toglie senza GT, d_P no."""
    rng = np.random.default_rng(6)
    p, k, n = 30, 4, 60
    Q = np.linalg.qr(np.random.default_rng(0).normal(size=(p, p)))[0]
    blocks = []
    for name in ("a", "b", "c"):
        T = rng.normal(size=(n, k))
        subj = np.repeat(np.arange(n), 5)
        noise = np.c_[0.1 * rng.normal(size=(5 * n, k)), 3.0 * rng.normal(size=(5 * n, 6)),
                      0.1 * rng.normal(size=(5 * n, p - k - 6))]
        X = np.c_[T[subj], np.zeros((5 * n, p - k))] @ Q.T + noise @ Q.T
        blocks.append(H.Block(X, subj, np.tile(np.arange(5), n), np.linalg.norm(T[:, None] - T[None], axis=-1), name))
    # la media delle covarianze di Ledoit-Wolf dei residui dalla media del soggetto, blocco per blocco
    ref = []
    for b in blocks[:2]:
        R = b.X - np.stack([b.X[b.subj == s].mean(0) for s in range(n)])[b.subj]
        ref.append(H.ledoit_wolf(R)[0])
    assert np.abs(H.within_cov(blocks[:2]) - np.mean(ref, axis=0)).max() < 1e-12
    m = H.Linear(H.sym_pow(H.within_cov(blocks[:2]), -0.5))
    s_w = H.score(m, blocks[2:])
    s_p = H.score(H.Head(1.0, np.zeros((0, p)), 0, 0.0, 1.0), blocks[2:])
    assert s_w > 0.8 and s_w > s_p + 0.3, (s_w, s_p)
    print(f"Mahalanobis intra-soggetto (stimata su a, b, valutata su c): {s_w:.3f} contro d_P {s_p:.3f}", flush=True)


def test_rank0_ranks() -> None:
    rng = np.random.default_rng(5)
    b = planted(30, rng)
    head = H.fit([b], 0, 0.0)
    i, j = H.pair_index(b)
    d = np.linalg.norm(b.X[i] - b.X[j], axis=1)
    g = b.G[b.subj[i], b.subj[j]]
    assert spearmanr(head.dist(b.X[i], b.X[j]), g).correlation == spearmanr(d, g).correlation
    print("r = 0: stessi ranghi di d_P", flush=True)


if __name__ == "__main__":
    test_grad()
    test_small()
    test_within()
    test_rank0_ranks()
    test_lw_coral()
    test_recovery()
    print("[test] tutte le prove passate", flush=True)
