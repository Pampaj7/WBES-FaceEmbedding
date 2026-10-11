"""D3 (``aau/runs/evidence/diagnostics/PROTOCOL_D3.md`` sez. 4-5, emendamento 1 sez. 4): testa metrica lineare (i), CV
per soggetto, Mahalanobis intra-soggetto (ii), Ledoit-Wolf, CORAL. Solo numpy e scipy, nessun dato: lo usano
d3_stats.py e test_d3.py.

Testa: d_h(i, j)^2 = alpha^2 ||x_i - x_j||^2 + ||W (x_i - x_j)||^2, alpha = e^a, W (r, p): d_P piu' una correzione di
rango r (Mahalanobis M = alpha^2 I + W^T W). Perdita sulle distanze (stress) con la cresta verso d_P:

    L = sum_p w_p (d_p - g_p)^2 / sum_p w_p g_p^2 + lambda ||W||_F^2 / alpha0^2

sulle coppie di un ``Block`` (un generatore): soggetti diversi, etichette diverse; ogni generatore pesa lo stesso.
Gradiente in forma chiusa: con C_ij = 2 w_ij e_ij / (Z d_ij) sulle coppie ordinate, dL/dW = 2 V^T (diag(C 1) - C) X
(V = X W^T) e dL/da = sum C alpha^2 ||dx||^2: O(n^2 r) per valutazione, nessuna coppia materializzata.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

R_GRID = (2, 4, 8, 16, 32, 64)
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
CONFIGS = [(0, 0.0)] + [(r, lam) for r in R_GRID for lam in LAMBDAS]   # r = 0: d_P da sola (alpha0)
N_FOLD, N_REPEAT = 5, 3
SEED_CV = 20261121
INIT_SCALE = 0.1                 # W iniziale = 0.1 alpha0 V_r (W = 0 e' stazionario in W)
MAXITER = 2000
TIE = 1e-3                       # scelta: entro 1e-3 dal massimo, r minore e poi lambda maggiore


@dataclass
class Block:
    """Le mesh di un generatore: x (n, p) in unita' di d_P, soggetto (indice in G) ed etichetta per mesh, G (s, s) GT
    d_P fra soggetti."""
    X: np.ndarray
    subj: np.ndarray
    lab: np.ndarray
    G: np.ndarray
    name: str = ""

    def take(self, subjects: np.ndarray) -> "Block":
        """Il blocco ristretto alle mesh dei soggetti ``subjects`` (indici di G, che resta intero)."""
        keep = np.isin(self.subj, subjects)
        return Block(self.X[keep], self.subj[keep], self.lab[keep], self.G, self.name)

    def pairs(self) -> np.ndarray:
        """Maschera (n, n) simmetrica delle coppie: soggetti diversi, etichette diverse."""
        return (self.subj[:, None] != self.subj[None, :]) & (self.lab[:, None] != self.lab[None, :])


def sqdist(X: np.ndarray) -> np.ndarray:
    n2 = (X * X).sum(1)
    return np.clip(n2[:, None] + n2[None, :] - 2.0 * X @ X.T, 0.0, None)


class Problem:
    """Le quantita' fisse di un fit: per blocco ||dx||^2, maschera pesata w, GT g; alpha0 e la base V_r."""

    def __init__(self, blocks: list[Block]) -> None:
        self.blocks = blocks
        self.p = blocks[0].X.shape[1]
        self.Dx2, self.Om, self.Gm = [], [], []
        for b in blocks:
            M = b.pairs()
            npairs = M.sum() / 2.0
            if npairs == 0:
                raise ValueError(f"{b.name}: nessuna coppia")
            self.Dx2.append(sqdist(b.X))
            self.Om.append(M / (len(blocks) * npairs))              # w_p, coppie ordinate (ognuna due volte)
            self.Gm.append(b.G[np.ix_(b.subj, b.subj)])
        self.Z = float(sum((O * G * G).sum() for O, G in zip(self.Om, self.Gm)))
        num = sum((O * G * np.sqrt(D)).sum() for O, G, D in zip(self.Om, self.Gm, self.Dx2))
        den = sum((O * D).sum() for O, D in zip(self.Om, self.Dx2))
        self.alpha0 = float(num / den)

    def basis(self, r: int) -> np.ndarray:
        """Le prime r direzioni principali delle x, centrate per blocco (r, p)."""
        Xc = np.concatenate([b.X - b.X.mean(0) for b in self.blocks])
        return np.linalg.svd(Xc, full_matrices=False)[2][:r]

    def loss(self, theta: np.ndarray, r: int, lam: float) -> tuple[float, np.ndarray]:
        a, W = theta[0], theta[1:].reshape(r, self.p)
        al2 = np.exp(2.0 * a)
        L, ga, gW = 0.0, 0.0, np.zeros_like(W)
        for b, Dx2, O, G in zip(self.blocks, self.Dx2, self.Om, self.Gm):
            V = b.X @ W.T
            d = np.sqrt(np.maximum(al2 * Dx2 + sqdist(V), 1e-24))
            e = d - G
            L += float((O * e * e).sum())
            C = 2.0 * O * e / (self.Z * d)
            ga += float((C * Dx2).sum()) * al2
            Lc = np.diag(C.sum(1)) - C
            gW += 2.0 * (V.T @ Lc) @ b.X
        L = L / self.Z + lam * float((W * W).sum()) / self.alpha0 ** 2
        gW += 2.0 * lam * W / self.alpha0 ** 2
        return L, np.concatenate([[ga], gW.ravel()])


@dataclass
class Head:
    alpha: float
    W: np.ndarray
    r: int
    lam: float
    alpha0: float
    nit: int = 0
    success: bool = True
    loss: float = float("nan")

    def dist(self, Xa: np.ndarray, Xb: np.ndarray) -> np.ndarray:
        """d_h fra le righe di Xa e Xb (stesse forme)."""
        dx = Xa - Xb
        d2 = self.alpha ** 2 * (dx * dx).sum(1)
        if self.r:
            v = dx @ self.W.T
            d2 = d2 + (v * v).sum(1)
        return np.sqrt(d2)


def fit(blocks: list[Block], r: int, lam: float, maxiter: int = MAXITER) -> Head:
    """La testa sui blocchi dati (solo gli array del dominio di training). r = 0: alpha0, nessuna ottimizzazione."""
    from scipy.optimize import minimize
    P = Problem(blocks)
    if r == 0:
        return Head(P.alpha0, np.zeros((0, P.p)), 0, 0.0, P.alpha0)
    W0 = INIT_SCALE * P.alpha0 * P.basis(r)
    th0 = np.concatenate([[np.log(P.alpha0)], W0.ravel()])
    res = minimize(P.loss, th0, args=(r, lam), jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    return Head(float(np.exp(res.x[0])), res.x[1:].reshape(r, P.p), r, lam, P.alpha0, int(res.nit),
                bool(res.success), float(res.fun))


def pair_index(b: Block) -> tuple[np.ndarray, np.ndarray]:
    """Coppie i < j del blocco (soggetti diversi, etichette diverse)."""
    i, j = np.triu_indices(len(b.subj), 1)
    keep = (b.subj[i] != b.subj[j]) & (b.lab[i] != b.lab[j])
    return i[keep], j[keep]


def score(head: Head, blocks: list[Block]) -> float:
    """Spearman di d_h con la GT d_P sulle coppie di ogni blocco, media sui blocchi."""
    from scipy.stats import spearmanr
    out = []
    for b in blocks:
        i, j = pair_index(b)
        out.append(spearmanr(head.dist(b.X[i], b.X[j]), b.G[b.subj[i], b.subj[j]]).correlation)
    return float(np.mean(out))


def folds(n_subjects: list[int], q: int, keys: list[int]) -> list[np.ndarray]:
    """Fold dei soggetti (ordinati) di ogni blocco per la ripetizione q (emendamento 1): il blocco con chiave j (indice
    della sorgente) usa ``SeedSequence([SEED_CV, q, j])``, cosi' ha gli stessi fold qualunque siano gli altri blocchi."""
    return [np.random.default_rng(np.random.SeedSequence([SEED_CV, q, j])).permutation(n) % N_FOLD
            for n, j in zip(n_subjects, keys)]


def split(blocks: list[Block], fold_of: list[np.ndarray], f: int) -> tuple[list[Block], list[Block]]:
    """(blocchi di fit, blocchi di validazione) del fold f; soggetti disgiunti per costruzione (verificato)."""
    fit_b, val_b = [], []
    for b, fo in zip(blocks, fold_of):
        tr, va = np.flatnonzero(fo != f), np.flatnonzero(fo == f)
        if np.intersect1d(tr, va).size or len(tr) + len(va) != len(b.G):
            raise RuntimeError(f"{b.name}: fold non disgiunti")
        fit_b.append(b.take(tr))
        val_b.append(b.take(va))
    for fb, vb in zip(fit_b, val_b):
        if np.intersect1d(np.unique(fb.subj), np.unique(vb.subj)).size:
            raise RuntimeError("soggetti di validazione nel fit")
    return fit_b, val_b


def choose(scores: dict) -> tuple[int, float]:
    """{(r, lambda): punteggio medio} -> la configurazione scelta (massimo; entro TIE, r minore poi lambda maggiore)."""
    best = max(scores.values())
    near = [k for k, v in scores.items() if v >= best - TIE]
    return min(near, key=lambda k: (k[0], -k[1]))


def wmedian(x: np.ndarray, w: np.ndarray) -> float:
    """Mediana pesata; con pesi tutti uguali e' ``np.median``."""
    x, w = np.asarray(x, np.float64), np.asarray(w, np.float64)
    if np.ptp(w) == 0:
        return float(np.median(x))
    o = np.argsort(x)
    x, cw = x[o], np.cumsum(w[o]) / w.sum()
    k = int(np.searchsorted(cw, 0.5))
    return float(0.5 * (x[k] + x[k + 1])) if np.isclose(cw[k], 0.5) and k + 1 < len(x) else float(x[k])


def calib_c(dist, blocks: list[Block]) -> float:
    """c = mediana pesata della GT / mediana pesata del modello sulle coppie (regola di fact_calib.calib_one);
    ``dist(b, i, j)`` -> distanze del modello sulle coppie (i, j) del blocco b."""
    g, m, w = [], [], []
    for b in blocks:
        i, j = pair_index(b)
        g.append(b.G[b.subj[i], b.subj[j]])
        m.append(dist(b, i, j))
        w.append(np.full(len(i), 1.0 / (len(blocks) * len(i))))
    g, m, w = np.concatenate(g), np.concatenate(m), np.concatenate(w)
    return wmedian(g, w) / wmedian(m, w)


# ------------------------------------------------------------------- (ii), CORAL: Ledoit-Wolf e metriche lineari fisse

@dataclass
class Linear:
    """d = ||A (x_i - x_j)||, A fissa (Mahalanobis intra-soggetto, CORAL)."""
    A: np.ndarray

    def dist(self, Xa: np.ndarray, Xb: np.ndarray) -> np.ndarray:
        return np.linalg.norm((Xa - Xb) @ self.A.T, axis=1)


def within_cov(blocks: list[Block]) -> np.ndarray:
    """Covarianza intra-soggetto (emendamento 1, testa ii), senza GT: per blocco i residui delle x dalla media del loro
    soggetto, Ledoit-Wolf; media sui blocchi (pesi uguali)."""
    covs = []
    for b in blocks:
        R = b.X.copy()
        for s in np.unique(b.subj):
            k = b.subj == s
            R[k] -= R[k].mean(0)
        covs.append(ledoit_wolf(R)[0])
    return np.mean(covs, axis=0)


def ledoit_wolf(X: np.ndarray) -> tuple[np.ndarray, float]:
    """(covarianza ristretta, coefficiente) come ``sklearn.covariance.ledoit_wolf`` (dati centrati sulla media,
    covarianza a massima verosimiglianza ristretta verso mu I), formula copiata."""
    X = np.asarray(X, np.float64)
    X = X - X.mean(0)
    n, p = X.shape
    S = X.T @ X / n
    X2 = X ** 2
    tr = X2.sum(0) / n
    mu = tr.sum() / p
    beta_ = float((X2.T @ X2).sum())
    delta_ = float(((X.T @ X) ** 2).sum()) / n ** 2
    beta = (beta_ / n - delta_) / (p * n)
    delta = (delta_ - 2.0 * mu * tr.sum() + p * mu ** 2) / p
    beta = min(beta, delta)
    k = 0.0 if beta == 0 else beta / delta
    return (1.0 - k) * S + k * mu * np.eye(p), float(k)


def sym_pow(S: np.ndarray, e: float) -> np.ndarray:
    """S^e di una simmetrica definita positiva (decomposizione spettrale)."""
    lam, V = np.linalg.eigh(0.5 * (S + S.T))
    if lam.min() <= 0:
        raise ValueError(f"covarianza non definita positiva (autovalore minimo {lam.min():.3e})")
    return (V * lam ** e) @ V.T


def coral(S_ref: np.ndarray, S_D: np.ndarray) -> np.ndarray:
    """A_D = S_ref^{1/2} S_D^{-1/2}: x -> A_D x porta la covarianza del dominio su quella di riferimento."""
    return sym_pow(S_ref, 0.5) @ sym_pow(S_D, -0.5)


def whiten(S_D: np.ndarray) -> np.ndarray:
    return sym_pow(S_D, -0.5)
