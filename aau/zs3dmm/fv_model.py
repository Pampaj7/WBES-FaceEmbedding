#!/usr/bin/env python3
"""Loader minimo del 3DMM FaceVerse v2 + forward solo forma, stessa interfaccia di ``hifi_model``.

    aau/run.sh aau/zs3dmm/fv_model.py            # self-check sul .npy di WBES_FV_NPY

Modello (``faceverse_simple_v2.npy``, un dict numpy; forward di FaceVerse)::

    V = meanshape + reshape(idBase @ id_coeff, (-1, 3))      # (28632, 3)

``idBase`` e' (85.896, 150), xyz interlacciati per riga: lo si verifica sugli spigoli come in
``hifi_model`` (layout interlacciato 0.0079 di spigolo medio dopo un campione, per coordinata
0.041 contro 0.0078 della media: misurato). Le norme dei 150 modi decrescono (1.64, 1.50, ...,
0.07) e la base non e' ortonormale: e' gia' scalata, e N(0, 1) e' il prior del modello.

Risoluzione e regione
---------------------
Due versioni nel file: la piena (28.632 vertici, 56.698 triangoli) e la semplificata
(``select_id``, 6.335 / 12.423). Si usa la PIENA: e' alla risoluzione di BFM (`original`
23.470 / 46.440), mentre la semplificata sta sotto la patch ICT (9.409) e la sua regione del
volto scenderebbe a ~1.800 vertici, meno della `down8k` degli altri domini.

Nessun crop: la mesh FaceVerse e' gia' una maschera del volto, non una testa chiusa -- da
orecchio a orecchio, dalla fronte al mento (i 66 landmark coprono y da -0.35, sopracciglia, a
+0.48, mento, su una mesh che va da -0.50 a +0.49), un solo bordo esterno (474 vertici, sul
retro delle guance) piu' la bocca aperta (92 vertici), come la patch HIFI3D che ha bocca e
occhi aperti. Proporzioni del bbox l:a:p = 1 : 1.06 : 0.61 contro 1 : 1.09 : 0.71 della patch
HIFI3D. ``skinmask`` NON e' una regione del volto: esclude labbra, occhi, sopracciglia e tutto
il contorno della mandibola (0/18 landmark delle labbra, 0/17 della mandibola), quindi non si
usa. ``face_vertices`` sono percio' tutti i vertici e ``f_face`` tutti i triangoli, passati
comunque per ``largest_component`` come negli altri domini.

Verso dei triangoli: 2 dei 56.698 triangoli del file sono capovolti rispetto ai vicini e
vengono riorientati al caricamento (``igl.bfs_orient``); senza, la decimazione di libigl non
termina. Vedi ``_load_cached``.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "v2_work" / "genict"))

import mesh_ops as mo  # noqa: E402
from hifi_model import _mean_edge, _to_vertices, face_patch, hifi_shape_mesh  # noqa: E402

shape_mesh = hifi_shape_mesh  # stesso forward lineare: template + shapedirs @ w


@lru_cache(maxsize=2)
def _load_cached(path: str, n_shape: int) -> dict:
    d = np.load(path, allow_pickle=True).item()
    mu = np.asarray(d["meanshape"], dtype=np.float64)
    nv = len(mu)
    B = np.asarray(d["idBase"], dtype=np.float64)      # (3n, k)
    if B.shape[0] != 3 * nv:
        raise ValueError(f"idBase {B.shape} incompatibile con {nv} vertici")
    k_all = B.shape[1]
    k = k_all if n_shape <= 0 else min(n_shape, k_all)
    B = B[:, :k].T                                        # (k, 3n), come hifi_model
    F = np.asarray(d["tri"]).astype(np.int64)
    if F.min() != 0 or F.max() != nv - 1:
        raise ValueError(f"tri: indici in [{F.min()}, {F.max()}], attesi [0, {nv - 1}]")
    # Due triangoli del file hanno il verso opposto ai vicini (4 spigoli interni percorsi due
    # volte nello stesso verso; HIFI3D e ICT: zero). Il collasso di spigoli di libigl lo
    # richiede coerente: con questi due triangoli igl.qslim (e igl.decimate) sotto ~27.000
    # triangoli esaurisce la memoria invece di terminare (job 1055751, 66 GB). bfs_orient
    # riallinea il verso, senza toccare vertici ne' connettivita'; il verso globale e' quello
    # della maggioranza, cioe' del file.
    import igl
    F_or = np.asarray(igl.bfs_orient(F)[0], dtype=np.int64)
    flipped = ~np.all(F_or == F, axis=1)
    if flipped.mean() > 0.5:
        F_or, flipped = F_or[:, ::-1], ~flipped
    F = np.ascontiguousarray(F_or, dtype=np.int32)

    # Layout: sul template il layout non conta (meanshape e' gia' (n, 3)); conta sulla base.
    # Si confronta lo spigolo medio dopo un campione N(0,1) con quello della media.
    z = np.random.default_rng(0).normal(size=k)
    edge0 = _mean_edge(mu, F)
    edges = {lay: _mean_edge(mu + _to_vertices(z @ B, lay), F) / edge0 for lay in ("interleaved", "planar")}
    layout = min(edges, key=edges.get)
    if max(edges.values()) < 2.0 * min(edges.values()):
        raise ValueError(f"layout della base ambiguo: spigolo relativo {edges}")
    shapedirs = np.ascontiguousarray(np.moveaxis(_to_vertices(B, layout), 0, -1))  # (nv, 3, k)
    norms = np.linalg.norm(B, axis=1)

    F_face = mo.largest_component(mo.remove_degenerate(mu, F))
    used = np.unique(F_face)
    return {
        "v_template": mu,
        "shapedirs": shapedirs,
        "f": F,
        "f_face": np.ascontiguousarray(F_face, dtype=np.int32),
        "face_vertices": used.astype(np.int32),
        "info": {
            "keys": {"mean": "meanshape", "basis": "idBase", "sigma": None, "tri": "tri", "mask": None},
            "all_keys": sorted(d),
            "n_verts_head": int(nv),
            "n_faces_head": int(len(F)),
            "n_basis_in_file": int(k_all),
            "n_shape_used": int(k),
            "layout": layout,
            "mean_edge_by_layout": edges,
            "scaling": "as_is (N(0,1) su idBase, gia' scalata: norme dei modi decrescenti)",
            "mode_norms_first5": [float(x) for x in norms[:5]],
            "mode_norms_last": float(norms[-1]),
            "sigma_first5": None,
            "region": "mesh piena senza crop (maschera del volto da orecchio a orecchio)",
            "faces_reoriented": int(flipped.sum()),
            "face_faces_largest_component": int(len(F_face)),
            "face_vertices": int(len(used)),
        },
    }


def load_fv(path: Path | str | None = None, n_shape: int = 0) -> dict:
    """Il 3DMM FaceVerse v2 (mesh piena); ``n_shape`` 0 = tutti i 150 modi."""
    path = Path(path or os.environ.get("WBES_FV_NPY", ""))
    if not path.is_file():
        raise FileNotFoundError(f"modello FaceVerse non trovato: '{path}' (WBES_FV_NPY)")
    return _load_cached(str(path.resolve()), int(n_shape))


@lru_cache(maxsize=2)
def _load_expressions_cached(path: str) -> dict:
    d = np.load(path, allow_pickle=True).item()
    nv = len(d["meanshape"])
    E = np.asarray(d["exBase"], dtype=np.float64)        # (3n, 52), come idBase
    names = [str(n) for n in d["exp_name_list"]]
    if E.shape != (3 * nv, len(names)):
        raise ValueError(f"exBase {E.shape} incompatibile con {nv} vertici e {len(names)} nomi")
    # Stesso layout di idBase (interlacciato, verificato sugli spigoli in _load_cached): lo si
    # ricontrolla qui sull'espressione piu' ampia, jawOpen a 1.0.
    layout = load_fv(path)["info"]["layout"]
    F = load_fv(path)["f"]
    mu = np.asarray(d["meanshape"], dtype=np.float64)
    j = names.index("jawOpen")
    edges = {lay: _mean_edge(mu + _to_vertices(E[:, j], lay), F) / _mean_edge(mu, F)
             for lay in ("interleaved", "planar")}
    if min(edges, key=edges.get) != layout:
        raise ValueError(f"layout di exBase {edges} diverso da quello di idBase ({layout})")
    return {"names": names,
            "exprdirs": np.ascontiguousarray(np.moveaxis(_to_vertices(E.T, layout), 0, -1)),  # (nv, 3, 52)
            "mean_edge_by_layout": edges}


def load_fv_expressions(path: Path | str | None = None) -> dict:
    """Base d'espressione di FaceVerse v2: ``exBase`` (52 blendshape ARKit, ``exp_name_list``).

    Forward di FaceVerse: ``V = meanshape + idBase @ id + exBase @ exp``; i coefficienti ARKit
    stanno in [0, 1]. ``exprdirs`` e' (nv, 3, 52) sulla testa intera, come ``shapedirs``.
    """
    path = Path(path or os.environ.get("WBES_FV_NPY", ""))
    if not path.is_file():
        raise FileNotFoundError(f"modello FaceVerse non trovato: '{path}' (WBES_FV_NPY)")
    return _load_expressions_cached(str(path.resolve()))


def _self_check() -> None:
    import json

    m = load_fv()
    print(json.dumps(m["info"], indent=2))
    k = m["shapedirs"].shape[2]
    V0 = shape_mesh(np.zeros(k), m)
    assert np.allclose(V0, m["v_template"]), "coefficienti nulli: deve tornare la forma media"
    V1 = shape_mesh(np.random.default_rng(0).normal(size=k), m)
    d = np.linalg.norm(V1 - V0, axis=1)
    bbox = V0.max(axis=0) - V0.min(axis=0)
    print(f"z~N(0,1): spostamento medio {d.mean():.4g}, max {d.max():.4g} "
          f"({d.mean() / np.linalg.norm(bbox):.3%} della diagonale)")
    Vf, Ff = face_patch(V1, m)
    Vp, Fp = mo.prepare_open_surface(Vf, Ff)
    assert len(Fp) == len(Ff) and len(Vp) == len(Vf), "la mesh non e' una superficie pulita"
    print(f"patch: V={len(Vf)} F={len(Ff)} bordo={len(mo.extract_boundary_vertices(Ff))} vertici")
    print("OK fv_model")


if __name__ == "__main__":
    _self_check()
