#!/usr/bin/env python3
"""Loader minimo del 3DMM GNM Head v3.0 (Google, Apache-2.0) + forward solo forma, stessa interfaccia di ``hifi_model``.

    aau/run.sh aau/zs3dmm/gnm_model.py            # self-check sul .npz di WBES_GNM_NPZ

Modello (``gnm_head.npz``, forward di ``repo/gnm/shape/gnm_common.py``, ~/data/gnm_head)::

    V_bind = template + identity @ vertex_identity_basis + expression @ expression_basis
    V      = LBS(V_bind + pose_correctives(rotations), joints(identity), rotations, translation)

Testa da 17.821 vertici in metri, +Y in alto e naso verso +Z (il frame di ICT e HIFI3D), mesh a
quad (17.662) con la sua triangolazione (35.324 = due triangoli per quad).

Identita' ed espressione
------------------------
``identity_names``: 170 ``head_*``, poi 3 ``eyes_*`` e 80 ``teeth_*`` (253). Si usano solo le 170
della testa, nell'ordine del file; occhi e denti restano a zero (bulbi e arcate non stanno nella
regione del volto). Le basi sono gia' scalate (norme dei modi decrescenti, nel manifest),
l'esempio di ``gnm_numpy.py`` campiona ``np.random.normal(size=gnm.identity_dim)`` e il README da'
-3..3 come intervallo tipico: N(0, 1) e' il prior, e si campiona come ICT, senza troncamento. Espressione zero per le identita'; la base d'espressione (383:
100 + 100 occhi, 150 parte bassa del volto, 32 lingua, 1 pupille) la carica
``load_gnm_expressions``, come ``load_fv_expressions``.

Posa: l'API (``gnm(identity, expression, rotations, translation)``) passa SEMPRE per LBS. Con
rotazioni e traslazione nulle le trasformazioni dei giunti sono traslazioni pure verso la loro
posizione di bind, i correttivi di posa valgono zero (dipendono da R - I) e i pesi di skinning
sommano a 1 per vertice: la mesh posata coincide con la bind pose. Non lo si assume: al
caricamento ``gnm_forward`` (port numpy di ``__call__``, ``linear_blend_skinning`` e
``joint_transforms_world``) valuta un'identita' casuale in posa neutra e la confronta con la forma
lineare ``template + shapedirs @ w``, ed esce se differiscono (scarto nel manifest). La regione
si valuta quindi sulla mesh in posa neutra, che e' la forma lineare.

Regione del volto
-----------------
Due gruppi di vertici candidati nel file:
  - ``skin_exterior`` (11.460 vertici): TUTTA la pelle esterna della testa -- cuoio capelluto,
    nuca, orecchie, collo fino all'attaccatura delle spalle (bbox 0.26 x 0.34 x 0.24 m). Non e'
    una regione del volto: la GT per vertice sarebbe dominata da cranio e collo, e il ``crop``
    (banda dal bordo esterno) taglierebbe il collo invece del contorno del viso;
  - ``hockey_mask`` (4.592 vertici): la maschera del volto, dalla fronte al mento e da guancia
    a guancia, con occhi, narici e bocca aperti (bbox l:a:p = 1 : 1.20 : 0.73, contro
    1 : 1.05 : 0.64 della patch HIFI3D e 1 : 1.47 : 0.86 della patch ICT, che arriva a orecchie
    e collo). E' la regione documentata dal modello che corrisponde a ``mask_face`` di HIFI3D e
    alla geometria "Face" di ICT: si usa questa.

La ``hockey_mask`` nativa ha pero' meta' dei vertici delle patch ICT (9.409) e HIFI3D (9.518): la
sua ``down8k`` scenderebbe a ~1.600 vertici, sotto quella di ogni altro dominio (lo stesso
motivo per cui in ``fv_model`` si e' scartata la versione semplificata di FaceVerse). La mesh e'
a quad: invece di spezzare ogni quad lungo una diagonale (``triangles`` del file) lo si divide
in 4 triangoli a ventaglio dal suo centro, la media dei 4 angoli (il centro del patch
bilineare). Il centro e' una combinazione lineare FISSA dei vertici della testa, quindi lo si
aggiunge a ``v_template``, ``shapedirs`` ed ``exprdirs`` come vertice in piu' (``n_quad_centers``),
e forward, ``face_patch`` e indici fissi restano quelli degli altri domini. Risultato: ~9.000
vertici e ~17.700 triangoli, la scala delle patch ICT e HIFI3D (numeri esatti nel manifest). Si
tengono i quad con i 4 angoli nella maschera, poi la sola componente connessa piu' grande, come
negli altri domini.
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
from hifi_model import face_patch, hifi_shape_mesh  # noqa: E402

shape_mesh = hifi_shape_mesh  # stesso forward lineare: template + shapedirs @ w (posa neutra, verificato)

REGION = "hockey_mask"
GROUP_THRESHOLD = 1e-4          # _NONZERO_THRESHOLD di gnm_xnp.vertex_group_mask
HEAD_PREFIX = "head_"
POSE_TOLERANCE = 1e-6           # metri: posa neutra contro forma lineare (float32 del file: ~1e-7)


def _rodrigues(r: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * K @ K


def gnm_forward(z: dict, identity: np.ndarray, expression: np.ndarray,
                rotations: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Port numpy di ``GNM.__call__`` (gnm_xnp.py, gnm_common.py) per un solo campione, testa nativa."""
    V = (z["template_vertex_positions"] + np.einsum("i,ivk->vk", identity, z["vertex_identity_basis"])
         + np.einsum("e,evk->vk", expression, z["expression_basis"]))
    J = z["template_joint_positions"] + np.einsum("i,ijk->jk", identity, z["joint_identity_basis"])
    R = np.stack([_rodrigues(r) for r in np.asarray(rotations, dtype=np.float64)])
    V = V + ((R - np.eye(3)).reshape(-1) @ z["pose_correctives_regressor"]).reshape(-1, 3)
    parents = [int(p) for p in z["joint_parent_indices"]]
    local = []
    for j in range(len(J)):
        M = np.eye(4)
        M[:3, :3] = R[j]
        M[:3, 3] = J[0] + translation if j == 0 else J[j] - J[parents[j]]
        local.append(M)
    world = [local[0]]
    for j in range(1, len(J)):
        world.append(world[parents[j]] @ local[j])
    world = np.stack(world)
    r_w, t_w = world[:, :3, :3], world[:, :3, 3]
    t_bind = t_w - (r_w @ J[:, :, None])[..., 0]
    W = z["skinning_weights"]
    return (np.einsum("jv,jmn->vmn", W, r_w) @ V[:, :, None])[..., 0] + np.einsum("jv,jm->vm", W, t_bind)


def _read(path: str) -> dict:
    with np.load(path) as d:
        out = {k: d[k] for k in d.files}
    for k, v in out.items():
        if v.dtype.kind == "f":
            out[k] = v.astype(np.float64)
    return out


def _quad_centers(X: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """(nv, ...) -> (nv + nq, ...): in coda la media dei 4 angoli di ogni quad."""
    return np.concatenate([X, X[quads].mean(axis=1)], axis=0)


@lru_cache(maxsize=2)
def _load_cached(path: str, n_shape: int) -> dict:
    z = _read(path)
    T = z["template_vertex_positions"]
    nv = len(T)
    id_names = [str(n) for n in z["identity_names"]]
    head = [i for i, n in enumerate(id_names) if n.startswith(HEAD_PREFIX)]
    if head != list(range(len(head))):
        raise ValueError(f"le basi {HEAD_PREFIX}* non sono le prime del file: {head[:3]}..{head[-3:]}")
    k_all = len(head)
    k = k_all if n_shape <= 0 else min(n_shape, k_all)
    B = z["vertex_identity_basis"][:k]                    # (k, nv, 3)
    if B.shape[1:] != (nv, 3):
        raise ValueError(f"vertex_identity_basis {z['vertex_identity_basis'].shape} incompatibile con {nv} vertici")

    # Posa neutra: la forma posata dall'API (LBS) deve coincidere con quella lineare.
    w_test = np.zeros(len(id_names))
    w_test[:k] = np.random.default_rng(0).normal(size=k)
    n_joints = len(z["joint_names"])
    V_posed = gnm_forward(z, w_test, np.zeros(len(z["expression_names"])), np.zeros((n_joints, 3)), np.zeros(3))
    pose_dev = float(np.abs(V_posed - (T + np.einsum("i,ivk->vk", w_test[:k], B))).max())
    if not pose_dev < POSE_TOLERANCE:
        raise ValueError(f"posa neutra diversa dalla forma lineare: scarto {pose_dev:.3g} m")
    weight_sum = z["skinning_weights"].sum(axis=0)

    # Regione: quad con i 4 angoli nel gruppo, ognuno in 4 triangoli a ventaglio dal centro.
    groups = [str(n) for n in z["vertex_group_names"]]
    mask = z["vertex_groups"][groups.index(REGION)] > GROUP_THRESHOLD
    Q_all = z["quads"].astype(np.int64)
    Q = Q_all[mask[Q_all].all(axis=1)]
    centers = nv + np.arange(len(Q))
    F_fan = np.concatenate([np.stack([Q[:, i], Q[:, (i + 1) % 4], centers], axis=1) for i in range(4)])
    v_template = _quad_centers(T, Q)
    shapedirs = np.ascontiguousarray(_quad_centers(np.moveaxis(B, 0, -1), Q))  # (nv + nq, 3, k)
    F_face = mo.largest_component(mo.remove_degenerate(v_template, F_fan))
    used = np.unique(F_face)
    # Verso dei triangoli: il ventaglio tiene l'ordine dei quad. Si esige che sia coerente (ogni
    # spigolo interno percorso una volta per verso, cfr. i due triangoli capovolti di FaceVerse)
    # e uscente come quello di ICT: normale complessiva della patch verso +Z, il naso.
    F_tri = z["triangles"].astype(np.int64)
    E = np.concatenate([F_face[:, [0, 1]], F_face[:, [1, 2]], F_face[:, [2, 0]]])
    if len(np.unique(E, axis=0)) != len(E):
        raise ValueError("verso dei triangoli della regione non coerente")
    tri = v_template[F_face]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).sum(axis=0)
    if not normal[2] > 0:
        raise ValueError(f"patch orientata verso l'interno: normale complessiva {normal}")

    norms = np.linalg.norm(B.reshape(k, -1), axis=1)
    ext = v_template[used].max(axis=0) - v_template[used].min(axis=0)
    return {
        "v_template": v_template,
        "shapedirs": shapedirs,
        "quads_face": Q.astype(np.int32),
        "f": np.ascontiguousarray(F_tri, dtype=np.int32),
        "f_face": np.ascontiguousarray(F_face, dtype=np.int32),
        "face_vertices": used.astype(np.int32),
        "info": {
            "keys": {"mean": "template_vertex_positions", "basis": "vertex_identity_basis[head_*]",
                     "sigma": None, "tri": "quads (ventaglio dal centro)", "mask": f"vertex_groups[{REGION}]"},
            "all_keys": sorted(z),
            "version": str(z["version"]),
            "variant": str(z["variant"]),
            "n_verts_head": int(nv),
            "n_faces_head": int(len(F_tri)),
            "n_quads_head": int(len(Q_all)),
            "n_basis_in_file": int(len(id_names)),
            "n_head_basis_in_file": int(k_all),
            "n_shape_used": int(k),
            "layout": "(I, V, 3)",
            "scaling": "as_is (N(0,1) sulle basi head_*, gia' scalate: norme dei modi decrescenti)",
            "mode_norms_first5": [float(x) for x in norms[:5]],
            "mode_norms_last": float(norms[-1]),
            "sigma_first5": None,
            "units": "metri",
            "frame": "+Y alto, +Z naso (frame ICT)",
            "pose": "neutra: rotazioni e traslazione nulle (LBS == bind pose, verificato)",
            "pose_neutral_max_dev_m": pose_dev,
            "skinning_weight_sum": [float(weight_sum.min()), float(weight_sum.max())],
            "region": f"{REGION}, quad divisi in 4 triangoli a ventaglio dal centro",
            "mask_vertices": int(mask.sum()),
            "mask_quads_all_in": int(len(Q)),
            "n_quad_centers": int(len(Q)),
            "face_faces_largest_component": int(len(F_face)),
            "face_vertices": int(len(used)),
            "face_bbox_m": [float(x) for x in ext],
            "face_normal_sum_unit": [float(x) for x in normal / np.linalg.norm(normal)],
        },
    }


def load_gnm(path: Path | str | None = None, n_shape: int = 0) -> dict:
    """Il 3DMM GNM Head con la regione del volto; ``n_shape`` 0 = tutte le 170 basi della testa."""
    path = Path(path or os.environ.get("WBES_GNM_NPZ", ""))
    if not path.is_file():
        raise FileNotFoundError(f"modello GNM non trovato: '{path}' (WBES_GNM_NPZ)")
    return _load_cached(str(path.resolve()), int(n_shape))


@lru_cache(maxsize=2)
def _load_expressions_cached(path: str) -> dict:
    with np.load(path) as d:
        E = d["expression_basis"].astype(np.float64)       # (383, nv, 3)
        names = [str(n) for n in d["expression_names"]]
    if E.shape[0] != len(names):
        raise ValueError(f"expression_basis {E.shape} incompatibile con {len(names)} nomi")
    Q = load_gnm(path)["quads_face"].astype(np.int64)
    exprdirs = np.ascontiguousarray(_quad_centers(np.moveaxis(E, 0, -1), Q))  # (nv+nq, 3, 383)
    groups = {}
    for i, n in enumerate(names):
        groups.setdefault(n.rsplit("_", 1)[0], []).append(i)
    return {"names": names, "exprdirs": exprdirs, "groups": groups}


def load_gnm_expressions(path: Path | str | None = None) -> dict:
    """Base d'espressione di GNM Head: ``expression_basis`` (383 componenti, ``expression_names``).

    Forward: ``V_bind = template + identity_basis @ id + expression_basis @ expr`` (posa neutra);
    gruppi ``left_eye_region`` / ``right_eye_region`` (100 + 100), ``lower_face_region`` (150),
    ``tongue`` (32), ``pupils`` (1); intervallo tipico dei coefficienti -3..3 (README). ``exprdirs``
    e' (nv + nq, 3, 383), con i centri dei quad come ``shapedirs``.
    """
    path = Path(path or os.environ.get("WBES_GNM_NPZ", ""))
    if not path.is_file():
        raise FileNotFoundError(f"modello GNM non trovato: '{path}' (WBES_GNM_NPZ)")
    return _load_expressions_cached(str(path.resolve()))


def _self_check() -> None:
    import json

    m = load_gnm()
    print(json.dumps(m["info"], indent=2))
    k = m["shapedirs"].shape[2]
    V0 = shape_mesh(np.zeros(k), m)
    assert np.allclose(V0, m["v_template"]), "coefficienti nulli: deve tornare la forma media"
    V1 = shape_mesh(np.random.default_rng(0).normal(size=k), m)
    d = np.linalg.norm(V1 - V0, axis=1)
    bbox = V0.max(axis=0) - V0.min(axis=0)
    print(f"z~N(0,1): spostamento medio {d.mean():.4g} m, max {d.max():.4g} "
          f"({d.mean() / np.linalg.norm(bbox):.3%} della diagonale)")
    assert d.mean() > 1e-6, "un campione N(0,1) non sposta la mesh"
    Vf, Ff = face_patch(V1, m)
    Vp, Fp = mo.prepare_open_surface(Vf, Ff)
    assert len(Fp) == len(Ff) and len(Vp) == len(Vf), "la patch non e' una superficie pulita"
    print(f"patch: V={len(Vf)} F={len(Ff)} bordo={len(mo.extract_boundary_vertices(Ff))} vertici; "
          f"bbox {(Vf.max(0) - Vf.min(0)).round(4).tolist()} m")
    ex = load_gnm_expressions()
    print("gruppi d'espressione: " + ", ".join(f"{g}={len(i)}" for g, i in ex["groups"].items()))
    print("OK gnm_model")


if __name__ == "__main__":
    _self_check()
