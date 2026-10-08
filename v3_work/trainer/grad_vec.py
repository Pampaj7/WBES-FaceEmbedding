"""``build_grad`` vettorizzato di diffusion-net: copia di aau/evidence/e9_bench/grad_vec.py (agente E9).

Stessa matrice di geometry.build_grad (diffusion-net/src/diffusion_net/geometry.py:209-273) senza il loop
Python per vertice: le somme per vertice sono ``np.bincount`` nell'ordine degli archi e A_i^-1 e' la formula
chiusa 2x2. Controllo di E9 (aau/runs/evidence/e9/grad_check.json, 20 mesh da 3k a 60k vertici): gradX/gradY
IDENTICI bit per bit dopo il cast a fp32 (scarto massimo 0), embedding del checkpoint e108 identici; circa
197 volte piu' veloce del build_grad originale. Sul pre-pass intero domina eigsh: E9 misura 9.3 contro 7.8 mesh/s
(k 128, mesh 3k-60k, 100 processi).

``install()`` sostituisce ``geometry.build_grad`` nel solo processo corrente.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse

EPS_REG = 1e-5   # == eps_reg di geometry.build_grad


def build_grad_vec(verts, edges, edge_tangent_vectors):
    """Drop-in di ``geometry.build_grad``: stessi argomenti, stessa csc complessa (V, V)."""
    from diffusion_net.utils import toNP

    e = np.asarray(toNP(edges), dtype=np.int64)
    t = np.asarray(toNP(edge_tangent_vectors), dtype=np.float64)
    n = int(verts.shape[0])
    keep = e[0] != e[1]
    tail, tip, t = e[0, keep], e[1, keep], t[keep]
    tx, ty = t[:, 0], t[:, 1]
    a = np.bincount(tail, weights=tx * tx, minlength=n) + EPS_REG
    b = np.bincount(tail, weights=tx * ty, minlength=n)
    d = np.bincount(tail, weights=ty * ty, minlength=n) + EPS_REG
    det = a * d - b * b
    cx = (d[tail] * tx - b[tail] * ty) / det[tail]
    cy = (a[tail] * ty - b[tail] * tx) / det[tail]
    diag = -(np.bincount(tail, weights=cx, minlength=n) + 1j * np.bincount(tail, weights=cy, minlength=n))
    rows = np.concatenate([np.arange(n), tail])
    cols = np.concatenate([np.arange(n), tip])
    vals = np.concatenate([diag, cx + 1j * cy])
    return scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()


def install() -> None:
    from diffusion_net import geometry
    geometry.build_grad = build_grad_vec
