"""Percorsi e geometria comuni della GT unificata (D2 / E8, paper/PLAN_MASSIVE.md sezione 3).

Solo numpy/scipy e open3d (ray casting e punto piu' vicino su superficie, ``Surface``). Ogni
funzione prende e restituisce array ``(V float64 (n, 3), F int (m, 3))``.

Convenzioni:
  - similarita' ``(s, R, t)``: ``y = s * R @ x + t``, cioe' ``Y = s * X @ R.T + t``;
  - mappa baricentrica: ``vidx`` (n, 3) indici di vertice del dominio, ``bary`` (n, 3), punto =
    ``sum_k bary[:, k] * V[vidx[:, k]]``. Gli indici sono di vertice, non di triangolo: la mappa
    vale per ogni mesh del dominio con gli stessi vertici, qualunque tabella di facce si usi.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
DATA_ROOT = REPO_ROOT / "datasets" / "UNIFIED_GT"
CORR_DIR = DATA_ROOT / "corr"
EVID_DIR = REPO_ROOT / "aau" / "runs" / "evidence" / "e8"

for _p in (REPO_ROOT / "faceBench", REPO_ROOT / "v2_work" / "genflame", REPO_ROOT / "v2_work" / "genict",
           REPO_ROOT / "aau" / "zs3dmm"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ------------------------------------------------------------------------------ geometria

def face_normals(V: np.ndarray, F: np.ndarray, unit: bool = True) -> np.ndarray:
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if unit:
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    return n


def face_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    return 0.5 * np.linalg.norm(face_normals(V, F, unit=False), axis=1)


def vertex_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Area baricentrica per vertice: un terzo dell'area di ogni triangolo incidente."""
    a = face_areas(V, F) / 3.0
    out = np.zeros(len(V))
    for k in range(3):
        np.add.at(out, F[:, k], a)
    return out


def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Normali per vertice pesate per area (somma delle normali non normalizzate)."""
    n = face_normals(V, F, unit=False)
    out = np.zeros_like(V)
    for k in range(3):
        np.add.at(out, F[:, k], n)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-30)


def edges_of(F: np.ndarray) -> np.ndarray:
    """Spigoli orientati (3m, 2), nell'ordine dei triangoli: lo spigolo k e' opposto al vertice k."""
    return np.concatenate([F[:, [1, 2]], F[:, [2, 0]], F[:, [0, 1]]])


def boundary_edge_mask(F: np.ndarray) -> np.ndarray:
    """(m, 3) bool: lo spigolo opposto al vertice k del triangolo e' di bordo (un solo triangolo)."""
    E = np.sort(edges_of(F), axis=1)
    _, inv, cnt = np.unique(E, axis=0, return_inverse=True, return_counts=True)
    return (cnt[inv.ravel()] == 1).reshape(3, len(F)).T


def boundary_vertices(F: np.ndarray, n_verts: int) -> np.ndarray:
    m = boundary_edge_mask(F)
    out = np.zeros(n_verts, dtype=bool)
    for k in range(3):
        e = edges_of(F)[k * len(F):(k + 1) * len(F)][m[:, k]]
        out[e.ravel()] = True
    return out


def largest_component(F: np.ndarray) -> np.ndarray:
    """Triangoli della componente connessa (per spigolo) piu' grande."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    m = len(F)
    E = np.sort(edges_of(F), axis=1)
    _, inv = np.unique(E, axis=0, return_inverse=True)
    inv = inv.ravel()
    tri = np.tile(np.arange(m), 3)
    order = np.argsort(inv, kind="stable")
    inv_s, tri_s = inv[order], tri[order]
    same = inv_s[1:] == inv_s[:-1]
    a, b = tri_s[1:][same], tri_s[:-1][same]
    G = sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(m, m))
    _, lab = connected_components(G, directed=False)
    return F[lab == np.bincount(lab).argmax()]


def compact(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vertici usati da F, ricompattati: (V', F', indici originali dei vertici di V')."""
    used = np.unique(F)
    remap = -np.ones(len(V), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return V[used], remap[F].astype(np.int64), used


def bary_interp(V: np.ndarray, vidx: np.ndarray, bary: np.ndarray) -> np.ndarray:
    """Punti della mappa baricentrica su una mesh coi vertici ``V`` (anche (..., nv, 3))."""
    return np.einsum("...nkd,nk->...nd", V[..., vidx, :], bary)


# ----------------------------------------------------------------------------- similarita'

def umeyama(X: np.ndarray, Y: np.ndarray, w: np.ndarray | None = None,
            scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """(s, R, t) che minimizzano sum_v w_v ||s R x_v + t - y_v||^2 (rotazione propria)."""
    w = np.ones(len(X)) if w is None else np.asarray(w, dtype=np.float64)
    W = w / w.sum()
    mx, my = W @ X, W @ Y
    Xc, Yc = X - mx, Y - my
    C = (Yc * W[:, None]).T @ Xc
    U, S, Vt = np.linalg.svd(C)
    D = np.ones(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2] = -1.0
    R = (U * D) @ Vt
    s = float((S * D).sum() / (W @ (Xc ** 2).sum(1))) if scale else 1.0
    return s, R, my - s * R @ mx


def apply_sim(X: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return s * X @ R.T + t


def compose_sim(a: tuple, b: tuple) -> tuple:
    """b dopo a: y = b(a(x))."""
    sa, Ra, ta = a
    sb, Rb, tb = b
    return sb * sa, Rb @ Ra, sb * Rb @ ta + tb


def rot_x(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


# -------------------------------------------------------------------- superficie (open3d)

class Surface:
    """Mesh per ray casting e punto piu' vicino (open3d RaycastingScene, Embree su CPU)."""

    def __init__(self, V: np.ndarray, F: np.ndarray):
        import open3d as o3d

        self.V = np.asarray(V, dtype=np.float64)
        self.F = np.asarray(F, dtype=np.int64)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(self.V.astype(np.float32)),
                                 o3d.core.Tensor(self.F.astype(np.uint32)))
        self._o3d = o3d

    @staticmethod
    def _bary(uv: np.ndarray) -> np.ndarray:
        # open3d: punto = (1 - u - v) a + u b + v c
        return np.stack([1.0 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1]], axis=1)

    def closest(self, P: np.ndarray) -> dict:
        """Punto piu' vicino sulla superficie: ``tri``, ``bary``, ``point`` (ricalcolato in float64), ``dist``."""
        q = self._o3d.core.Tensor(np.asarray(P, dtype=np.float32))
        r = self.scene.compute_closest_points(q)
        tri = r["primitive_ids"].numpy().astype(np.int64)
        bary = np.clip(self._bary(r["primitive_uvs"].numpy().astype(np.float64)), 0.0, 1.0)
        bary /= bary.sum(1, keepdims=True)
        point = np.einsum("nkd,nk->nd", self.V[self.F[tri]], bary)
        return {"tri": tri, "bary": bary, "point": point, "dist": np.linalg.norm(point - P, axis=1)}

    def cast(self, origins: np.ndarray, dirs: np.ndarray) -> dict:
        """Raggi: ``hit`` bool, ``t``, ``tri``, ``bary`` (validi solo dove ``hit``)."""
        rays = np.concatenate([origins, dirs], axis=1).astype(np.float32)
        r = self.scene.cast_rays(self._o3d.core.Tensor(rays))
        t = r["t_hit"].numpy().astype(np.float64)
        hit = np.isfinite(t)
        tri = r["primitive_ids"].numpy().astype(np.int64)
        tri[~hit] = -1
        bary = self._bary(r["primitive_uvs"].numpy().astype(np.float64))
        return {"hit": hit, "t": t, "tri": tri, "bary": bary}


# ---------------------------------------------------------------- visibilita' esterna

CONE_DEG = 75.0
N_DIRS = 64
OFFSET_MM = 0.05


def fibonacci_cone(n: int, max_deg: float) -> np.ndarray:
    """n direzioni quasi uniformi nella calotta attorno a +z di semiapertura max_deg."""
    zmin = np.cos(np.radians(max_deg))
    i = np.arange(n) + 0.5
    z = 1.0 - (1.0 - zmin) * i / n
    phi = i * np.pi * (3.0 - np.sqrt(5.0))
    r = np.sqrt(1.0 - z ** 2)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def exterior(V: np.ndarray, F: np.ndarray, cand: np.ndarray, normals: np.ndarray | None = None) -> np.ndarray:
    """Per i vertici ``cand``: visibili da almeno una direzione entro CONE_DEG da +z, con la normale
    rivolta verso quella direzione (raggio dalla superficie spostata di OFFSET_MM lungo la normale;
    occlusori = tutta la mesh). Frame: +z fuori dal volto, mm. ``normals`` uscenti (default: quelle
    di F, cioe' F col verso uscente)."""
    surf = Surface(V, F)
    N = vertex_normals(V, F) if normals is None else normals
    vis = np.zeros(len(cand), dtype=bool)
    for d in fibonacci_cone(N_DIRS, CONE_DEG):
        facing = N[cand] @ d > 0.05
        o = V[cand] + OFFSET_MM * N[cand] + 1e-3 * d
        h = surf.cast(o, np.tile(d, (len(cand), 1)))
        vis |= facing & ~h["hit"]
    return vis


# ------------------------------------------------------------------------------- I/O

def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n")
    tmp.replace(path)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(type(o))


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)
