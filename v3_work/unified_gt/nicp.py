"""NICP della regione FLAME su un template, guidato dai landmark.

Riusa i pezzi di ``faceBench/facebench/nonrigid_aligners/nonrigid_icp.py`` (il ``nonrigid_icp_align``
di ``aau/indomain/ir_template.py``): matrice d'incidenza spigoli-vertici
(``_triangles_to_edge_vertex_adjacent_matrix``), matrice D dei vertici omogenei
(``_sparse_matrix_from_vertices``), risolutore delle equazioni normali (``_spsolve_system``), stessa
rigidezza ``kron(M, diag(1, 1, 1, gamma))`` e stessa forma del sistema
``[alpha * A1; W D] X = [0; W U]``. Quattro differenze, tutte del metodo di Amberg et al. (CVPR
2007, "Optimal Step Nonrigid ICP") di cui la funzione di faceBench e' un port:
  1. il grafo della rigidezza sono gli spigoli dei triangoli FLAME, non la Delaunay 2D su (x, y)
     dei punti (che fra gli occhi aperti e le labbra collega bordi che non sono vicini sulla
     superficie, e ai lati del volto, dove la superficie e' quasi parallela a z, si ripiega);
  2. le righe dei landmark ``beta * DL X = beta * U_L`` (Amberg, eq. 9), con il landmark sorgente
     come combinazione baricentrica di tre vertici FLAME;
  3. corrispondenze sul punto piu' vicino della SUPERFICIE bersaglio (non sul vertice), scartate se
     cadono sul bordo del bersaglio, se le normali divergono oltre 72 gradi (cos 0.3) o se sono
     oltre 3 volte la mediana e oltre 0.1 in coordinate normalizzate (~9 mm; Amberg, sezione 3.2);
  4. D costruita una volta dai vertici iniziali: X e' la deformazione affine totale, come in
     Amberg (faceBench ricostruisce D dai vertici correnti a ogni iterazione, cosi' la rigidezza
     vincola solo l'incremento).
Coordinate normalizzate come in faceBench (centro della sorgente, |x| massimo = 1), cosi' la
rigidezza ha la sua scala; lo schedule parte dalle sue alpha (50, 30, 18) e scende fino a 1.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

import ugt as C
from facebench.nonrigid_aligners.nonrigid_icp import (  # noqa: E402
    _sparse_matrix_from_vertices, _spsolve_system, _triangles_to_edge_vertex_adjacent_matrix)

ALPHAS = (50.0, 30.3, 18.4, 11.2, 6.8, 4.1, 2.5, 1.5, 1.0)
BETA = 1.0
GAMMA = 1.0
MAX_ITER = 20
TOL = 1e-4
NORMAL_COS = 0.3
DIST_FACTOR = 3.0
DIST_FLOOR = 0.1


def landmark_matrix(Vs: np.ndarray, Fs: np.ndarray, tri: np.ndarray, bary: np.ndarray) -> sp.csr_matrix:
    """(L, 4n): riga l = sum_k bary_lk * [x, y, z, 1] del vertice k del triangolo tri_l."""
    L, n = len(tri), len(Vs)
    rows, cols, vals = [], [], []
    Vh = np.concatenate([Vs, np.ones((n, 1))], axis=1)
    for k in range(3):
        v = Fs[tri, k]
        for c in range(4):
            rows.append(np.arange(L))
            cols.append(4 * v + c)
            vals.append(bary[:, k] * Vh[v, c])
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(L, 4 * n))


def on_boundary(F: np.ndarray, tri: np.ndarray, bary: np.ndarray, bnd_edge: np.ndarray,
                bnd_vert: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Il punto (tri, bary) sta su uno spigolo o un vertice di bordo del bersaglio."""
    on_edge = (bary < eps) & bnd_edge[tri]
    on_vert = (bary > 1.0 - eps) & bnd_vert[F[tri]]
    return on_edge.any(1) | on_vert.any(1)


class Target:
    """Bersaglio in coordinate normalizzate: superficie, bordo, normali orientate."""

    def __init__(self, V: np.ndarray, F: np.ndarray):
        self.V, self.F = V, F
        self.surf = C.Surface(V, F)
        self.bnd_edge = C.boundary_edge_mask(F)
        self.bnd_vert = C.boundary_vertices(F, len(V))
        self.normals = C.face_normals(V, F)

    def orient_like(self, P: np.ndarray, N: np.ndarray) -> bool:
        """Gira le normali del bersaglio se in media sono opposte a quelle della sorgente; True se girate."""
        c = self.surf.closest(P)
        if np.sum(np.einsum("nd,nd->n", self.normals[c["tri"]], N)) < 0:
            self.normals = -self.normals
            return True
        return False

    def match(self, P: np.ndarray, N: np.ndarray) -> dict:
        c = self.surf.closest(P)
        c["boundary"] = on_boundary(self.F, c["tri"], c["bary"], self.bnd_edge, self.bnd_vert)
        c["cos"] = np.einsum("nd,nd->n", self.normals[c["tri"]], N)
        return c


def nicp(Vs: np.ndarray, Fs: np.ndarray, target: Target, lm_tri: np.ndarray, lm_bary: np.ndarray,
         lm_target: np.ndarray, alphas=ALPHAS, beta: float = BETA, gamma: float = GAMMA) -> dict:
    """Deforma la sorgente (Vs, Fs) sul bersaglio; tutto gia' in coordinate normalizzate.

    Ritorna ``Y`` (n, 3) deformata e il log per passo di alpha.
    """
    n = len(Vs)
    M = _triangles_to_edge_vertex_adjacent_matrix(Fs.T)
    if M.shape[1] != n:
        raise ValueError("vertici della sorgente non referenziati dai triangoli")
    A1 = sp.kron(M, sp.diags([1.0, 1.0, 1.0, gamma])).tocsr()
    D = _sparse_matrix_from_vertices(Vs.T).tocsr()
    DL = landmark_matrix(Vs, Fs, lm_tri, lm_bary)
    X = np.tile(np.vstack([np.eye(3), np.zeros((1, 3))]), (n, 1))
    log = []
    for alpha in alphas:
        it = 0
        for it in range(1, MAX_ITER + 1):
            Y = D @ X
            c = target.match(Y, C.vertex_normals(Y, Fs))
            ok = ~c["boundary"] & (c["cos"] > NORMAL_COS)
            thr = max(DIST_FACTOR * np.median(c["dist"][ok]), DIST_FLOOR) if ok.any() else np.inf
            w = (ok & (c["dist"] < thr)).astype(np.float64)
            A = sp.vstack([alpha * A1, sp.diags(w) @ D, beta * DL]).tocsc()
            B = sp.csc_matrix(np.vstack([np.zeros((A1.shape[0], 3)), c["point"] * w[:, None],
                                         beta * lm_target]))
            Xn = _spsolve_system(A, B)
            step = float(np.linalg.norm(Xn - X) / np.sqrt(n))
            X = Xn
            if step < TOL:
                break
        Y = D @ X
        log.append({"alpha": alpha, "iterations": it, "valid_fraction": float(w.mean()),
                    "median_dist": float(np.median(c["dist"])),
                    "landmark_rms": float(np.sqrt(((DL @ X - lm_target) ** 2).sum(1).mean()))})
    return {"Y": D @ X, "log": log, "DL": DL, "X": X}
