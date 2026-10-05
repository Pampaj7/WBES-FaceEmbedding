#!/usr/bin/env python3
"""Loader minimo del 3DMM HIFI3D (Tencent AI-NExT, versione est-asiatica) + forward solo forma.

    aau/run.sh aau/zs3dmm/hifi_model.py            # self-check sul .mat di WBES_HIFI_MAT

Modello (verbatim da ``3DMM/scripts/test_basis_io.py`` di tencent-ailab/hifi3dface,
``np_get_geometry``)::

    geo = mu_shape + para_shape @ basis_shape        # (1, 3n), poi reshape (n, 3)

e il campionatore ufficiale e' ``np.random.normal(size=[1, n_basis])``: N(0, 1) direttamente
su ``basis_shape``. Se il .mat porta anche una deviazione standard per modo
(``sigma_shape``), la base viene confrontata con lei: base ortonormale -> si moltiplica per
sigma; base gia' scalata -> si usa com'e'. La decisione e' presa sui numeri e scritta nel
manifest, non assunta (``describe()``).

Layout dei vertici: ``np_get_geometry`` fa ``reshape(-1, n, 3)`` (xyz interlacciati), mentre
il loader npy di ``utils/basis.py`` fa ``reshape(3, n).T`` (per coordinata). Qui si provano i
due e si tiene quello con gli spigoli dei triangoli piu' corti sulla forma media: col layout
sbagliato le coordinate di un vertice sono tre vertici diversi e gli spigoli esplodono.

Regione del volto
-----------------
I vertici HIFI3D sono la testa intera (20.481: volto, orecchie, collo, nuca). Il protocollo
REMESH-2 confronta patch del volto (BFM `original` e' un crop del volto; ICT tiene la
geometria #0 "Face" del suo README). Come per ICT si usa la regione documentata dal modello
stesso: ``mask_face``, la maschera booleana per vertice del .mat ("face vertex mask in bool",
docstring di ``load_3dmm_basis``). Si tengono i triangoli con i tre vertici nella maschera,
poi la sola componente connessa piu' grande (``mesh_ops.largest_component``, come ICT), e
gli indici si compattano. E' un insieme di indici FISSO, uguale per ogni identita': e' questo
che fa di `original` una topologia.

Misurato su AI-NEXT-Shape.mat (sha256 nel manifest delle identita'): ``tri`` contiene GIA'
solo il volto (18.684 triangoli, base 0), e coincide triangolo per triangolo con i triangoli
della testa intera di AI-NEXT-Shape-NoAug.mat (40.832) che cadono in ``mask_face``; la
maschera e' identica nei due file. Patch: 9.518 vertici, 18.684 triangoli, una componente,
nessuno spigolo non-manifold, un bordo da 358 vertici -- la stessa scala della patch ICT
(9.409 / 18.460). Il .mat non ha deviazioni standard e le norme dei modi decrescono (106, 36,
25, ...): la base e' gia' scalata, e z ~ N(0, 1) e' il prior del modello.
"""

from __future__ import annotations

import hashlib
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "v2_work" / "genict"))

import mesh_ops as mo  # noqa: E402  (genict: igl, niente open3d)

MEAN_KEYS = ("mu_shape",)
BASIS_KEYS = ("basis_shape", "bases_shape")
SIGMA_KEYS = ("sigma_shape", "sig_shape", "ev_shape")
TRI_KEYS = ("tri", "tri_v")
MASK_KEYS = ("mask_face",)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_mat(path: Path) -> dict:
    """Variabili del .mat come array numpy; v7.3 (HDF5) via h5py, trasposto all'ordine MATLAB."""
    try:
        import scipy.io
        raw = scipy.io.loadmat(str(path))
        return {k: v for k, v in raw.items() if not k.startswith("__")}
    except NotImplementedError:
        import h5py
        out = {}
        with h5py.File(str(path), "r") as fh:
            for k, v in fh.items():
                if isinstance(v, h5py.Dataset):
                    out[k] = np.asarray(v).T  # h5py legge in ordine C: MATLAB e' column-major
        return out


def _first(mat: dict, keys: tuple[str, ...], what: str, required: bool = True):
    for k in keys:
        if k in mat:
            return k, mat[k]
    if required:
        raise KeyError(f"{what}: nessuna delle chiavi {keys} nel .mat (chiavi: {sorted(mat)})")
    return None, None


def _mean_edge(V: np.ndarray, F: np.ndarray) -> float:
    e = np.concatenate([V[F[:, 0]] - V[F[:, 1]], V[F[:, 1]] - V[F[:, 2]], V[F[:, 2]] - V[F[:, 0]]])
    return float(np.linalg.norm(e, axis=1).mean())


def _to_vertices(flat: np.ndarray, layout: str) -> np.ndarray:
    """(..., 3n) -> (..., n, 3) col layout dato."""
    n = flat.shape[-1] // 3
    if layout == "interleaved":
        return flat.reshape(*flat.shape[:-1], n, 3)
    return np.swapaxes(flat.reshape(*flat.shape[:-1], 3, n), -1, -2)


@lru_cache(maxsize=2)
def _load_cached(path: str, n_shape: int) -> dict:
    mat = read_mat(Path(path))
    mean_key, mu = _first(mat, MEAN_KEYS, "forma media")
    basis_key, B = _first(mat, BASIS_KEYS, "base di forma")
    sigma_key, sigma = _first(mat, SIGMA_KEYS, "deviazione standard", required=False)
    tri_key, tri = _first(mat, TRI_KEYS, "triangoli")
    mask_key, mask = _first(mat, MASK_KEYS, "maschera del volto", required=False)

    mu = np.asarray(mu, dtype=np.float64).ravel()
    n3 = mu.size
    if n3 % 3:
        raise ValueError(f"{mean_key}: {n3} valori, non multiplo di 3")
    nv = n3 // 3
    B = np.asarray(B, dtype=np.float64)
    if B.ndim != 2 or n3 not in B.shape:
        raise ValueError(f"{basis_key}: forma {B.shape} incompatibile con {n3} coordinate")
    if B.shape[1] != n3:  # (3n, k) -> (k, 3n), la convenzione di test_basis_io
        B = B.T
    k_all = B.shape[0]
    k = k_all if n_shape <= 0 else min(n_shape, k_all)
    B = B[:k]

    F = np.asarray(tri).astype(np.int64)
    if F.shape[1] != 3:
        F = F.T
    # In AI-NEXT-Shape.mat `tri` e' gia' la sola regione del volto (18.684 triangoli, indici
    # 1..19.980 in base 0: il vertice 0 non e' usato), quindi min == 1 NON vuol dire indici
    # MATLAB. Base 1 solo se l'indice massimo e' proprio nv.
    if F.max() == nv:
        F = F - 1
    if F.min() < 0 or F.max() > nv - 1:
        raise ValueError(f"{tri_key}: indici in [{F.min()}, {F.max()}], attesi in [0, {nv - 1}]")
    F = F.astype(np.int32)

    edges = {lay: _mean_edge(_to_vertices(mu, lay), F) for lay in ("interleaved", "planar")}
    layout = min(edges, key=edges.get)
    other = max(edges, key=edges.get)
    if edges[other] < 3.0 * edges[layout]:
        raise ValueError(f"layout dei vertici ambiguo: spigolo medio {edges}")

    v_template = _to_vertices(mu, layout)
    norms = np.linalg.norm(B, axis=1)
    scaling = "as_is (N(0,1) su basis_shape, come test_basis_io)"
    sigma_used = None
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=np.float64).ravel()[:k]
        if np.allclose(norms, 1.0, rtol=1e-2):
            B = B * sigma[:, None]
            scaling = f"base ortonormale x {sigma_key}"
            sigma_used = sigma
        elif np.allclose(norms, sigma, rtol=1e-2):
            scaling = f"base gia' scalata ({sigma_key} == norme dei modi)"
        else:
            raise ValueError(f"{basis_key}: norme dei modi {norms[:4]}.. ne' 1 ne' {sigma_key} {sigma[:4]}..")
    shapedirs = np.ascontiguousarray(np.moveaxis(_to_vertices(B, layout), 0, -1))  # (nv, 3, k)

    if mask is None:
        raise KeyError(f"il .mat non ha {MASK_KEYS}: serve una regione del volto documentata "
                       f"(chiavi: {sorted(mat)})")
    mask = np.asarray(mask).ravel().astype(bool)
    if mask.size != nv:
        raise ValueError(f"{mask_key}: {mask.size} valori per {nv} vertici")
    F_mask = F[mask[F].all(axis=1)]
    # Componente piu' grande e niente degeneri, sulla forma media: gli indici restano quelli
    # della testa intera (``compact`` li rimappa per ogni identita').
    F_mask = mo.remove_degenerate(v_template, F_mask)
    F_face = mo.largest_component(F_mask)
    used = np.unique(F_face)

    return {
        "v_template": v_template,
        "shapedirs": shapedirs,
        "f": F,
        "f_face": np.ascontiguousarray(F_face, dtype=np.int32),
        "face_vertices": used.astype(np.int32),
        "info": {
            "keys": {"mean": mean_key, "basis": basis_key, "sigma": sigma_key, "tri": tri_key,
                     "mask": mask_key},
            "all_keys": sorted(mat),
            "n_verts_head": int(nv),
            "n_faces_head": int(len(F)),
            "n_basis_in_file": int(k_all),
            "n_shape_used": int(k),
            "layout": layout,
            "mean_edge_by_layout": edges,
            "scaling": scaling,
            "mode_norms_first5": [float(x) for x in norms[:5]],
            "mode_norms_last": float(norms[-1]),
            "sigma_first5": None if sigma_used is None else [float(x) for x in sigma_used[:5]],
            "mask_vertices": int(mask.sum()),
            "mask_faces_all_in": int(len(F_mask)),
            "face_faces_largest_component": int(len(F_face)),
            "face_vertices": int(len(used)),
        },
    }


def load_hifi(path: Path | str | None = None, n_shape: int = 0) -> dict:
    """Il 3DMM HIFI3D con la regione del volto; ``n_shape`` 0 = tutti i modi del .mat."""
    path = Path(path or os.environ.get("WBES_HIFI_MAT", ""))
    if not path.is_file():
        raise FileNotFoundError(f"modello HIFI3D non trovato: '{path}' (WBES_HIFI_MAT)")
    return _load_cached(str(path.resolve()), int(n_shape))


def hifi_shape_mesh(weights: np.ndarray, model: dict) -> np.ndarray:
    """Testa intera (nv, 3) per i coefficienti `weights` (gia' nella scala di shapedirs)."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    k = model["shapedirs"].shape[2]
    if not 0 < len(w) <= k:
        raise ValueError(f"servono 1..{k} coefficienti, dati {len(w)}")
    return model["v_template"] + model["shapedirs"][:, :, : len(w)] @ w


shape_mesh = hifi_shape_mesh  # nome generico, quello che usa zs_identities.py


def face_patch(V_head: np.ndarray, model: dict) -> tuple[np.ndarray, np.ndarray]:
    """Patch del volto compattata: (V (n_face, 3), F (m, 3) int32) a indici fissi."""
    used = model["face_vertices"]
    remap = -np.ones(len(V_head), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return np.ascontiguousarray(V_head[used]), remap[model["f_face"]].astype(np.int32)


def _self_check() -> None:
    import json

    m = load_hifi()
    print(json.dumps(m["info"], indent=2))
    k = m["shapedirs"].shape[2]
    V0 = hifi_shape_mesh(np.zeros(k), m)
    assert np.allclose(V0, m["v_template"]), "coefficienti nulli: deve tornare la forma media"
    bbox = V0.max(axis=0) - V0.min(axis=0)
    print(f"testa media: bbox {bbox.round(3).tolist()}")

    rng = np.random.default_rng(0)
    z = rng.normal(size=k)
    V1 = hifi_shape_mesh(z, m)
    d = np.linalg.norm(V1 - V0, axis=1)
    print(f"z~N(0,1): spostamento medio {d.mean():.4g}, max {d.max():.4g} "
          f"({d.mean() / np.linalg.norm(bbox):.3%} della diagonale)")
    assert d.mean() > 1e-6, "un campione N(0,1) non sposta la mesh"

    Vf, Ff = face_patch(V1, m)
    Vp, Fp = mo.prepare_open_surface(Vf, Ff)
    assert len(Fp) == len(Ff) and len(Vp) == len(Vf), "la patch non e' una superficie pulita"
    nb = len(mo.extract_boundary_vertices(Ff))
    print(f"patch volto: V={len(Vf)} F={len(Ff)} bordo={nb} vertici; "
          f"bbox {(Vf.max(0) - Vf.min(0)).round(3).tolist()}")
    print("OK hifi_model")


if __name__ == "__main__":
    _self_check()
