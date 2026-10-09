"""``build_grad`` vettorizzato di E9, congelato per E1.

Copia di ``build_grad_vec`` da aau/evidence/e9_bench/grad_vec.py (sha256 del file alla copia:
ecdb67569b5e343a7fc57b536b72c16059ccb8935ac5021f2461bce58d3a065e), senza modifiche al corpo: una modifica
successiva del file di E9 non cambia i job di E1. Uguaglianza degli operatori del pre-pass con l'originale
verificata in aau/runs/evidence/e1/gradvec_check.json.
"""
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
    # A^-1 = [[d, -b], [-b, a]] / det, applicata a ogni e_j
    cx = (d[tail] * tx - b[tail] * ty) / det[tail]
    cy = (a[tail] * ty - b[tail] * tx) / det[tail]
    diag = -(np.bincount(tail, weights=cx, minlength=n) + 1j * np.bincount(tail, weights=cy, minlength=n))
    rows = np.concatenate([np.arange(n), tail])
    cols = np.concatenate([np.arange(n), tip])
    vals = np.concatenate([diag, cx + 1j * cy])
    return scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()
