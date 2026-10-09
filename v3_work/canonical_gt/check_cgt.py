#!/usr/bin/env python3
"""Controlli delle funzioni nuove di cgt.py contro casi noti e forza bruta (nessun dato reale, solo mu).

    v3_work/unified_gt/run.sh v3_work/canonical_gt/check_cgt.py

1. ``rigid_fit`` recupera una rigida nota; 2. la rigida robusta non si fa spostare da un'area deformata (il
"naso" spinto avanti di 8 mm), la LS si'; 3. ``Ident`` (AUC, rank-1, rapporto, con i pesi delle repliche)
contro un calcolo di forza bruta; 4. EDM invariante alla rigida, EDM_s alla scala; 5. S: scala attorno al
centroide fisso di mu.
"""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
import cgt
from cgt import Canon

rng = np.random.default_rng(0)
cn = Canon(frames={})
mu, W = cn.mu, cn.W
# 1. rigid_fit recupera una rigida nota
def rand_rot(n):
    A = rng.normal(size=(n, 3, 3)); Q, _ = np.linalg.qr(A)
    Q = Q * np.sign(np.linalg.det(Q))[:, None, None]
    return Q
Rt = rand_rot(5); tt = rng.normal(size=(5, 3)) * 50
X = np.einsum("ab,nbc->nac", mu - 0, np.transpose(Rt, (0, 2, 1))) + tt[:, None]   # x = R mu + t
R, t = cgt.rigid_fit(X, mu, np.broadcast_to(W, X.shape[:2]))
back = np.einsum("nvb,nab->nva", X, R) + t[:, None]
print("1 rigid_fit, max |aligned - mu|:", np.abs(back - mu).max())
# 2. robusta con un'area deformata: LS sposta, robusta no
Y = mu.copy()
nose = np.argsort(-mu[:, 2])[:120]                     # punti piu' avanti: "naso" spostato di 8 mm in avanti
Y[nose, 2] += 8.0
Xo = np.einsum("vb,nab->nva", Y, Rt) + tt[:, None]
ls = cn.rigid_ls(Xo); rb = cn.rigid_robust(Xo)
other = np.setdiff1d(np.arange(len(mu)), nose)
print("2 residuo fuori dal naso, LS:", np.abs(ls["a"][:, other] - mu[other]).max(), " robusta:", np.abs(rb["a"][:, other] - mu[other]).max(),
      "giri", rb["iterations"], "conv", rb["converged"].all(), "area ridotta", rb["downweighted_area"].round(3))
# 3. Ident contro forza bruta
n_p, per = 7, 4
person = np.repeat(np.arange(n_p), per)
Z = rng.normal(size=(n_p, 3))[person] + 0.6 * rng.normal(size=(len(person), 3))
D = np.linalg.norm(Z[:, None] - Z[None], axis=-1)
idn = cgt.Ident(person, 50, 1)
r = idn.evaluate(D)
def brute(D, person, c):
    pidx = np.unique(person, return_inverse=True)[1]
    iu = np.triu_indices(len(D), 1)
    gen, imp, wg, wi = [], [], [], []
    for a, b in zip(*iu):
        if pidx[a] == pidx[b]:
            gen.append(D[a, b]); wg.append(c[pidx[a]])
        else:
            imp.append(D[a, b]); wi.append(c[pidx[a]] * c[pidx[b]])
    gen, imp, wg, wi = map(np.asarray, (gen, imp, wg, wi))
    num = sum(wg[k] * (wi[imp > gen[k]].sum() + 0.5 * wi[imp == gen[k]].sum()) for k in range(len(gen)))
    auc = num / (wg.sum() * wi.sum())
    Dn = D + np.diag(np.full(len(D), np.inf)); hit = pidx[Dn.argmin(1)] == pidx
    r1 = (c[pidx] * hit).sum() / c[pidx].sum()
    def wmed(x, w):
        o = np.argsort(x, kind="stable"); cw = np.cumsum(w[o]); return x[o][np.searchsorted(cw, 0.5 * cw[-1])]
    return auc, r1, wmed(gen, wg.astype(float)) / wmed(imp, wi.astype(float))
for k in (0, 1, 7):
    b = brute(D, person, idn.counts[k])
    print("3 Ident replica", k, "auc", r["auc"][k], b[0], "rank1", r["rank1"][k], b[1], "ratio", r["ratio"][k], b[2])
# 4. EDM invariante alla rigida, EDM_s alla scala
X2 = np.stack([mu, mu @ Rt[0].T + tt[0], 1.1 * mu, mu + rng.normal(size=mu.shape)])
E = cn.edm_features(X2); Es = cn.edm_features(X2, scale_free=True)
De, Ds = cgt.edm_distances(E), cgt.edm_distances(Es)
print("4 EDM(mu, rigida(mu))", De[0, 1], " EDM(mu, 1.1 mu)", De[0, 2], " EDM_s(mu, 1.1 mu)", Ds[0, 2], " EDM(mu, rumore)", De[0, 3])
iu = np.triu_indices(cgt.EDM_K, 1)
vidx, bary = cn.edm_points()
Q = np.einsum("kcd,kc->kd", mu[vidx], bary)
print("4b feature contro norma diretta, max |diff|:", np.abs(E[0] - np.linalg.norm(Q[:, None] - Q[None], axis=-1)[iu]).max())
# 5. S: invariante alla scala attorno a qualunque centro? no: solo scala per identita' attorno a m fisso
S = cn.shape_S(np.stack([mu, 1.1 * mu]))
print("5 S(mu) vs S(1.1 mu), RMS:", cn.distances(S[:1], S[1:])[0, 0], "(centro di scala diverso da m: atteso > 0)")
m = W @ mu
S2 = cn.shape_S(np.stack([mu, m + 1.1 * (mu - m)]))
print("5b scala attorno a m: RMS", cn.distances(S2[:1], S2[1:])[0, 0], "(atteso 0)")
