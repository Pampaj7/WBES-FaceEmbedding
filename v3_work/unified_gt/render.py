"""Render frontale ortografico (ray casting open3d, CPU) e detector 2D denso (mediapipe FaceMesh).

La stessa procedura per ogni template, FLAME compreso:
  1. ``search_orientation``: le 24 rotazioni assiali del frame del dato, render a bassa
     risoluzione, FaceMesh; si tiene la vista con un volto diritto (dalla bocca agli occhi verso
     l'alto dell'immagine entro 45 gradi) e col
     naso piu' sporgente verso la camera rispetto agli angoli esterni degli occhi (esclude profili
     e il retro concavo di una maschera aperta, l'illusione della faccia cava);
  2. ``detect``: render a 1024 px, FaceMesh (468 punti, i 10 dell'iride esclusi), ogni punto 2D
     retroproiettato con UN raggio esattamente nella sua posizione sub-pixel: triangolo colpito e
     coordinate baricentriche, quindi un punto esatto della mesh.

Camera: guarda lungo -z, alto dell'immagine = +y, nel frame passato. Ombreggiatura lambertiana a
due facce (la normale si gira verso la camera), cosi' il verso dei triangoli non conta (BFM ha le
normali verso l'interno).
"""

from __future__ import annotations

import itertools

import numpy as np

from ugt import Surface, face_normals

# FaceMesh: indici fissi del modello a 468 punti (mediapipe/python/solutions/face_mesh_connections)
NOSE_TIP = 1
EYE_OUTER = (33, 263)
MOUTH_CORNERS = (61, 291)
N_MESH_LANDMARKS = 468
SKIN_RGB = np.array([0.86, 0.72, 0.62])
LIGHT = np.array([0.25, 0.35, 1.0]) / np.linalg.norm([0.25, 0.35, 1.0])
BACKGROUND = 0.08


def face_oval_indices() -> np.ndarray:
    """I 36 punti del contorno (FACEMESH_FACE_OVAL): stanno sulla silhouette, non su un punto anatomico."""
    from mediapipe.python.solutions.face_mesh_connections import FACEMESH_FACE_OVAL
    return np.unique(np.array(sorted(FACEMESH_FACE_OVAL)).ravel())


class View:
    """Proiezione ortografica: frame di vista = ``(V - center) @ R.T``; pixel quadrati."""

    def __init__(self, R: np.ndarray, center: np.ndarray, half: float, res: int):
        self.R, self.center, self.half, self.res = np.asarray(R, float), np.asarray(center, float), float(half), int(res)
        self.px = 2.0 * self.half / self.res

    def to_view(self, X: np.ndarray) -> np.ndarray:
        return (X - self.center) @ self.R.T

    def from_view(self, Y: np.ndarray) -> np.ndarray:
        return Y @ self.R + self.center

    def rays(self, u: np.ndarray, v: np.ndarray, z0: float) -> tuple[np.ndarray, np.ndarray]:
        """Raggi (nel frame di vista) per coordinate immagine continue u (colonne), v (righe)."""
        x = -self.half + u * self.px
        y = self.half - v * self.px
        o = np.stack([x, y, np.full_like(x, z0)], axis=1)
        d = np.tile([0.0, 0.0, -1.0], (len(x), 1))
        return o, d

    def project(self, X: np.ndarray) -> np.ndarray:
        """(u, v) continue dei punti X (frame del dato)."""
        Y = self.to_view(X)
        return np.stack([(Y[:, 0] + self.half) / self.px, (self.half - Y[:, 1]) / self.px], axis=1)


def framing(V: np.ndarray, R: np.ndarray, res: int, margin: float = 0.06) -> View:
    """Vista che inquadra tutta la mesh ruotata da R, centrata sul suo bbox."""
    Y = V @ R.T
    lo, hi = Y.min(0), Y.max(0)
    c_view = 0.5 * (lo + hi)
    half = 0.5 * max(hi[0] - lo[0], hi[1] - lo[1]) * (1.0 + 2 * margin)
    return View(R, c_view @ R, half, res)  # c_view @ R: il centro nel frame del dato


def render(surf: Surface, view: View) -> dict:
    """Immagine RGB uint8 (res, res, 3) + per pixel triangolo colpito (-1 = sfondo)."""
    res = view.res
    jj, ii = np.meshgrid(np.arange(res) + 0.5, np.arange(res) + 0.5)
    Vv = view.to_view(surf.V)
    z0 = Vv[:, 2].max() + 10.0 * view.px
    o, d = view.rays(jj.ravel(), ii.ravel(), z0)
    # la scena e' nel frame del dato: raggi riportati li'
    hit = surf.cast(view.from_view(o), d @ view.R)
    n = face_normals(surf.V, surf.F)[np.maximum(hit["tri"], 0)] @ view.R.T
    n[n[:, 2] < 0] *= -1.0
    shade = 0.22 + 0.78 * np.clip(n @ LIGHT, 0.0, 1.0)
    img = np.where(hit["hit"][:, None], SKIN_RGB[None] * shade[:, None], BACKGROUND)
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8).reshape(res, res, 3)
    return {"img": img, "tri": hit["tri"].reshape(res, res)}


_FACEMESH = None


def facemesh(img: np.ndarray) -> np.ndarray | None:
    """468 punti (u, v) in pixel continui, o None se nessun volto."""
    global _FACEMESH
    import mediapipe as mp

    if _FACEMESH is None:
        _FACEMESH = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                                    refine_landmarks=True, min_detection_confidence=0.3)
    r = _FACEMESH.process(img)
    if not r.multi_face_landmarks:
        return None
    lm = r.multi_face_landmarks[0].landmark
    h, w = img.shape[:2]
    return np.array([[p.x * w, p.y * h] for p in lm[:N_MESH_LANDMARKS]], dtype=np.float64)


def backproject(surf: Surface, view: View, uv: np.ndarray) -> dict:
    """Un raggio per punto 2D: ``hit``, ``tri``, ``bary``, ``point`` (frame del dato), ``cos`` (normale-vista)."""
    Vv = view.to_view(surf.V)
    z0 = Vv[:, 2].max() + 10.0 * view.px
    o, d = view.rays(uv[:, 0], uv[:, 1], z0)
    h = surf.cast(view.from_view(o), d @ view.R)
    tri = np.maximum(h["tri"], 0)
    point = np.einsum("nkd,nk->nd", surf.V[surf.F[tri]], h["bary"])
    n = face_normals(surf.V, surf.F)[tri] @ view.R.T
    return {"hit": h["hit"], "tri": h["tri"], "bary": h["bary"], "point": point,
            "cos": np.abs(n[:, 2])}


def proper_rotations() -> list[np.ndarray]:
    """Le 24 rotazioni che mandano assi in assi (permutazioni con segno, determinante +1)."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            M[np.arange(3), perm] = signs
            if np.isclose(np.linalg.det(M), 1.0):
                out.append(M)
    return out


def search_orientation(surf: Surface, res: int = 384) -> tuple[np.ndarray, list[dict]]:
    """Rotazione assiale R0 che mette il volto di fronte alla camera e diritto (vedi docstring)."""
    log = []
    best, best_score = None, -np.inf
    for R in proper_rotations():
        view = framing(surf.V, R, res)
        img = render(surf, view)["img"]
        uv = facemesh(img)
        rec = {"R": R.tolist(), "detected": uv is not None}
        if uv is not None:
            # dalla bocca agli occhi: verso l'alto dell'immagine entro 45 gradi
            up = uv[list(EYE_OUTER)].mean(0) - uv[list(MOUTH_CORNERS)].mean(0)
            bp = backproject(surf, view, uv[[NOSE_TIP, *EYE_OUTER]])
            if bp["hit"].all():
                Y = view.to_view(bp["point"])
                iod = np.linalg.norm(Y[1] - Y[2])
                protr = (Y[0, 2] - Y[1:, 2].mean()) / max(iod, 1e-12)
            else:
                protr = -np.inf
            upright = bool(-up[1] > abs(up[0]))
            rec.update({"upright": bool(upright), "protrusion_over_iod": float(protr)})
            if upright and protr > 0.15 and protr > best_score:
                best, best_score = R, protr
        log.append(rec)
    if best is None:
        raise RuntimeError("nessuna rotazione assiale con un volto frontale e diritto")
    return best, log


def detect(surf: Surface, view: View) -> dict:
    """Render, FaceMesh, retroproiezione dei 468 punti."""
    r = render(surf, view)
    uv = facemesh(r["img"])
    if uv is None:
        raise RuntimeError("FaceMesh: nessun volto nel render frontale")
    bp = backproject(surf, view, uv)
    return {"img": r["img"], "uv": uv, **bp}
