"""Trasporto di un campo di spostamenti dalla regione unificata a un template, con raccordo biarmonico.

Un template A (la topologia di lavoro dello stream: ``sources.MMSource``, patch o media decimata) ha la sua
mappa della regione unificata (1478 punti della regione FLAME, ``unified_space.npz``; BFM 2019 da
``datasets/STREAM/maps``): i punti ``P_A`` stanno sulla superficie media di A. Un campo g sui punti (uno per
punto, (n_u, 3)) si porta sui vertici di A in tre parti:
  1. DENTRO la regione: punto piu' vicino del vertice sulla triangolazione unificata ``(P_A, F_u)`` (frame FLAME,
     mm veri), dentro se dista meno di ``IN_MM`` e la normale concorda (coseno > ``COS_IN``); il valore e'
     l'interpolazione baricentrica di g su quel triangolo, esatta sui punti;
  2. FUORI, entro ``BAND_MM`` di distanza geodetica dalla regione (Dijkstra sugli spigoli di A): raccordo
     BIARMONICO (L^T M^-1 L, cotangenti e massa di Voronoi di igl sulla media di A in mm), vincolato ai valori
     della regione e a zero oltre la banda. Riempie anche i buchi della regione (occhi, bocca, narici) e
     raccorda il bordo esterno con continuita' del valore e, circa, della derivata;
  3. oltre la banda: zero (orecchie, collo, nuca restano quelli di A).
Tutto e' lineare in g: ``apply`` costa un prodotto sparso e una soluzione col fattore LU precalcolato.

Unita' e frame: g entra nel frame FLAME in mm veri (``to_flame_mm`` del modello sorgente: u_B R_B, con
u = ``frame.mm_per_unit`` dichiarata, come la GT FR di E12, e R la rotazione canonica dello stream); esce nelle
unita' NATIVE di A (``from_flame_mm``). La taglia relativa dei domini NON si normalizza: uno spostamento di 5 mm
resta di 5 mm.

Controlli di validita' (``mesh_checks``): triangoli capovolti rispetto al template, triangoli degeneri (area
sotto ``AREA_RATIO_MIN`` volte quella del template), auto-intersezioni (test degli assi separatori fra triangoli
vicini che non condividono vertici, ``self_intersections``) e continuita' del raccordo (``edge_strain``).
"""
from __future__ import annotations

import numpy as np

IN_MM = 1.0             # distanza massima dalla triangolazione unificata per un vertice "dentro"
COS_IN = 0.5            # concordanza delle normali per un vertice "dentro"
BAND_MM = 40.0          # larghezza geodetica del raccordo fuori dalla regione
AREA_RATIO_MIN = 0.02   # triangolo degenere: area < 2% di quella del template
SI_TOL_MM = 1e-4        # sovrapposizione minima sugli assi separatori per contare un'intersezione


# --------------------------------------------------------------------------------------- geometria

def face_normals(V: np.ndarray, F: np.ndarray, unit: bool = True) -> np.ndarray:
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if unit:
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    return n


def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    n = face_normals(V, F, unit=False)
    out = np.zeros_like(V)
    for k in range(3):
        np.add.at(out, F[:, k], n)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-30)


def unique_edges(F: np.ndarray) -> np.ndarray:
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    return np.unique(np.sort(E, axis=1), axis=0)


def closest_on_mesh(Q: np.ndarray, V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(triangolo, baricentriche (q, 3), distanza) del punto piu' vicino di ogni Q sulla superficie (V, F)."""
    import igl
    d2, I, Cp = igl.point_mesh_squared_distance(np.ascontiguousarray(Q, dtype=np.float64),
                                                np.ascontiguousarray(V, dtype=np.float64),
                                                np.ascontiguousarray(F, dtype=np.int64))
    I = np.asarray(I, dtype=np.int64).ravel()
    a, b, c = (V[F[I, k]] for k in range(3))
    v0, v1, v2 = b - a, c - a, Cp - a
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = np.where(np.abs(d00 * d11 - d01 * d01) > 1e-30, d00 * d11 - d01 * d01, 1e-30)
    w1 = (d11 * d20 - d01 * d21) / den
    w2 = (d00 * d21 - d01 * d20) / den
    B = np.clip(np.stack([1.0 - w1 - w2, w1, w2], axis=1), 0.0, 1.0)
    return I, B / B.sum(1, keepdims=True), np.sqrt(np.maximum(np.asarray(d2).ravel(), 0.0))


# --------------------------------------------------------------------------------------- trasporto

class RegionTransfer:
    """Operatore lineare g (n_u, 3) sui punti della regione -> spostamenti sui vertici di A.

    ``X``: vertici della media di A nel frame FLAME in mm veri (n, 3); ``F``: triangoli di A (verso uscente);
    ``P``: punti della regione su A, stesso frame (n_u, 3); ``F_u``: triangoli della regione unificata."""

    def __init__(self, X: np.ndarray, F: np.ndarray, P: np.ndarray, F_u: np.ndarray,
                 band_mm: float = BAND_MM, in_mm: float = IN_MM):
        import scipy.sparse as sp
        from scipy.sparse.csgraph import dijkstra
        from scipy.sparse.linalg import splu

        n, n_u = len(X), len(P)
        self.n, self.n_u, self.band_mm = n, n_u, float(band_mm)
        tri, bary, dist = closest_on_mesh(X, P, F_u)
        nA = vertex_normals(X, F)
        nU = face_normals(P, F_u)
        if np.einsum("nd,nd->n", nA, vertex_normals(P, F_u)[F_u[tri, 0]]).mean() < 0:
            nU = -nU                                       # stesso verso delle normali di A
        cos = np.einsum("nd,nd->n", nA, nU[tri])
        inside = (dist < in_mm) & (cos > COS_IN)
        # dentro: interpolazione baricentrica sul triangolo della regione
        ii = np.flatnonzero(inside)
        rows = np.repeat(ii, 3)
        self.B_in = sp.csr_matrix((bary[ii].ravel(), (rows, F_u[tri[ii]].ravel())), shape=(n, n_u))
        # distanza geodetica (Dijkstra sugli spigoli, mm) dalla regione
        E = unique_edges(F)
        le = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
        G = sp.csr_matrix((np.concatenate([le, le]), (np.concatenate([E[:, 0], E[:, 1]]),
                                                       np.concatenate([E[:, 1], E[:, 0]]))), shape=(n, n))
        geo = dijkstra(G, directed=False, indices=ii, min_only=True) if len(ii) else np.full(n, np.inf)
        free = ~inside & (geo <= band_mm)
        self.inside, self.free, self.geo = inside, free, geo
        self.edges, self.edge_len = E, le
        self.tri, self.bary, self.dist = tri, bary, dist
        # raccordo biarmonico sui vertici liberi: Q_ff x_f = -Q_fi x_i (gli zeri oltre la banda non entrano)
        import igl
        L = igl.cotmatrix(np.ascontiguousarray(X), np.ascontiguousarray(F, dtype=np.int64))
        M = igl.massmatrix(np.ascontiguousarray(X), np.ascontiguousarray(F, dtype=np.int64),
                           igl.MASSMATRIX_TYPE_VORONOI)
        m = np.asarray(M.diagonal()).ravel()
        Minv = sp.diags(1.0 / np.maximum(m, 1e-12 * max(m.max(), 1e-30)))
        Q = (L.T @ Minv @ L).tocsr()
        fi, ci = np.flatnonzero(free), ii
        self.free_idx = fi
        if len(fi):
            Qff = Q[fi][:, fi].tocsc()
            eps = 1e-10 * float(abs(Qff.diagonal()).max())
            self._lu = splu((Qff + eps * sp.identity(len(fi), format="csc")).tocsc())
            self.Q_fi = Q[fi][:, ci].tocsr()
            self.B_in_rows = self.B_in[ci]                # valori dei vincoli: righe "dentro" di B_in
        else:
            self._lu = None

    def apply(self, g: np.ndarray) -> np.ndarray:
        """Spostamenti (n, 3) sui vertici di A dal campo g (n_u, 3) (stesse unita' e frame di g)."""
        g = np.asarray(g, dtype=np.float64)
        out = np.asarray(self.B_in @ g)
        if self._lu is not None:
            rhs = -(self.Q_fi @ np.asarray(self.B_in_rows @ g))
            out[self.free_idx] = self._lu.solve(np.ascontiguousarray(rhs))
        return out

    def hard_cut(self, g: np.ndarray) -> np.ndarray:
        """Il trasporto SENZA raccordo (zero fuori dalla regione): solo per mostrare la discontinuita' evitata."""
        return np.asarray(self.B_in @ np.asarray(g, dtype=np.float64))

    def describe(self) -> dict:
        return {"n_vertices": int(self.n), "inside": int(self.inside.sum()), "free_band": int(self.free.sum()),
                "zero_beyond_band": int((~self.inside & ~self.free).sum()), "band_mm": self.band_mm,
                "inside_dist_mm_p95": float(np.percentile(self.dist[self.inside], 95)) if self.inside.any() else None}


def edge_strain(d: np.ndarray, rt: RegionTransfer) -> dict:
    """Continuita' del raccordo: |d_i - d_j| / |x_i - x_j| sugli spigoli (d in mm), per zona dello spigolo.
    ``region``: entrambi dentro; ``seam``: uno dentro e uno nella banda (il bordo del raccordo); ``band``: entrambi
    nella banda; ``edge_out``: uno nella banda e uno oltre (zero)."""
    E = rt.edges
    s = np.linalg.norm(d[E[:, 0]] - d[E[:, 1]], axis=1) / np.maximum(rt.edge_len, 1e-9)
    zone = np.where(rt.inside, 0, np.where(rt.free, 1, 2))
    za, zb = np.sort(np.stack([zone[E[:, 0]], zone[E[:, 1]]], axis=1), axis=1).T
    out = {}
    for name, m in (("region", (za == 0) & (zb == 0)), ("seam", (za == 0) & (zb == 1)),
                    ("band", (za == 1) & (zb == 1)), ("edge_out", (za == 1) & (zb == 2))):
        if m.any():
            out[name] = {"p50": float(np.median(s[m])), "p99": float(np.percentile(s[m], 99)),
                         "max": float(s[m].max()), "n": int(m.sum())}
    return out


# --------------------------------------------------------------------------------------- controlli

def self_intersections(V: np.ndarray, F: np.ndarray, tol: float = SI_TOL_MM, max_pairs: int = 20_000_000) -> np.ndarray:
    """(k, 2) coppie di triangoli che si intersecano, fra quelli che NON condividono vertici (test degli assi
    separatori: 2 normali, 9 prodotti di spigoli, 6 normali degli spigoli nel piano per i casi quasi complanari).
    Candidati: sfere dal baricentro che si toccano e scatole che si sovrappongono."""
    from scipy.spatial import cKDTree
    V = np.asarray(V, dtype=np.float64)
    T = V[F]
    c = T.mean(1)
    r = np.linalg.norm(T - c[:, None], axis=2).max(1)
    pairs = cKDTree(c).query_pairs(2.0 * float(r.max()), output_type="ndarray")
    if len(pairs) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if len(pairs) > max_pairs:
        raise ValueError(f"{len(pairs)} coppie candidate: mesh troppo degenere per il test")
    i, j = pairs[:, 0], pairs[:, 1]
    keep = np.linalg.norm(c[i] - c[j], axis=1) <= r[i] + r[j]
    lo, hi = T.min(1), T.max(1)
    keep &= (lo[i] <= hi[j] + tol).all(1) & (lo[j] <= hi[i] + tol).all(1)
    i, j = i[keep], j[keep]
    share = (F[i][:, :, None] == F[j][:, None, :]).any((1, 2))
    i, j = i[~share], j[~share]
    if len(i) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    A, B = T[i], T[j]
    eA = np.stack([A[:, 1] - A[:, 0], A[:, 2] - A[:, 1], A[:, 0] - A[:, 2]], axis=1)
    eB = np.stack([B[:, 1] - B[:, 0], B[:, 2] - B[:, 1], B[:, 0] - B[:, 2]], axis=1)
    nA, nB = np.cross(eA[:, 0], eA[:, 1]), np.cross(eB[:, 0], eB[:, 1])
    axes = [nA, nB] + [np.cross(eA[:, a], eB[:, b]) for a in range(3) for b in range(3)]
    # normali degli spigoli nel piano: senza, due triangoli vicini quasi complanari risultano "intersecati"
    axes += [np.cross(nA, eA[:, a]) for a in range(3)] + [np.cross(nB, eB[:, b]) for b in range(3)]
    separated = np.zeros(len(i), dtype=bool)
    for ax in axes:
        nrm = np.linalg.norm(ax, axis=1)
        ok = nrm > 1e-12
        u = ax / np.where(ok, nrm, 1.0)[:, None]
        pa, pb = np.einsum("nkd,nd->nk", A, u), np.einsum("nkd,nd->nk", B, u)
        gap = np.maximum(pb.min(1) - pa.max(1), pa.min(1) - pb.max(1))
        separated |= ok & (gap > -tol)
    return np.stack([i[~separated], j[~separated]], axis=1)


def mesh_checks(V: np.ndarray, F: np.ndarray, V_ref: np.ndarray, full: bool = True) -> dict:
    """Controlli di validita' di una mesh contro il template della stessa topologia (stesse unita', mm).
    ``full``: anche le auto-intersezioni (il test piu' caro)."""
    n0 = face_normals(V_ref, F, unit=False)
    n1 = face_normals(V, F, unit=False)
    a0, a1 = 0.5 * np.linalg.norm(n0, axis=1), 0.5 * np.linalg.norm(n1, axis=1)
    cos = np.einsum("nd,nd->n", n0, n1) / np.maximum(4.0 * a0 * a1, 1e-30)
    ratio = a1 / np.maximum(a0, 1e-30)
    out = {"finite": bool(np.isfinite(V).all()), "flipped": int((cos < 0).sum()),
           "degenerate": int((ratio < AREA_RATIO_MIN).sum()), "area_ratio_min": float(ratio.min()),
           "area_ratio_max": float(ratio.max())}
    if full and out["finite"]:
        si = self_intersections(V, F)
        out["self_intersections"] = int(len(si))
    return out


def checks_ok(c: dict) -> bool:
    return bool(c["finite"] and c["flipped"] == 0 and c["degenerate"] == 0 and c.get("self_intersections", 0) == 0)
