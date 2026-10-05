#!/usr/bin/env python3
"""Controllo della canonicalizzazione del frame nel pre-pass (prepass_ops.apply_frame).

    aau/run.sh aau/data_scale/check_canon.py

Su id0000 original (BFM), operatori senza e con R = Rx(180) = diag(1,-1,-1) e flip_faces (la
trasformazione misurata dal PI: BFM ha naso verso -z, alto verso -y, normali verso l'interno):
  * R e' un'isometria: autovalori e massa coincidono, i vertici sono V @ R.T;
  * con det(R) = +1 e flip_faces le normali sono MENO le normali originali ruotate;
  * orientazione, misurata: la normale media pesata per area va confrontata con la direzione
    frontale (il centro del patch sporge rispetto al bordo). Dopo il canon BFM deve avere la stessa
    direzione frontale e lo stesso verso delle normali di ICT (ict0000 original).
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepass_ops  # noqa: E402

DS = HERE.parents[1] / "datasets"
SPEC = {"R": [[1, 0, 0], [0, -1, 0], [0, 0, -1]], "flip_faces": True}


def normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    return n  # pesate per area (norma = 2 x area)


def boundary_vertices(F):
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    u, c = np.unique(e, axis=0, return_counts=True)
    return np.unique(u[c == 1])


def nose_alignment(V, F):
    """(cos fra normale media e direzione frontale, segno z della direzione frontale).

    Direzione frontale lungo z: il centro di un patch facciale sporge rispetto al suo bordo, quindi
    e' il segno di (z mediana dei vertici interni - z media del bordo). Il "vertice piu' lontano in
    z" non funziona: su ICT e' sul bordo, che si curva all'indietro (prima versione, sbagliata).
    """
    b = boundary_vertices(F)
    inner = np.setdiff1d(np.arange(len(V)), b)
    front = np.sign(np.median(V[inner, 2]) - V[b, 2].mean())
    n = normals(V, F).sum(0)
    return float(n[2] / np.linalg.norm(n)) * front, int(front)


src = DS / "REMESH/npz_data_topo_500/id0000_GTready_original.npz"
with tempfile.TemporaryDirectory() as t:
    a, b = Path(t) / "a" / src.name, Path(t) / "b" / src.name
    a.parent.mkdir(); b.parent.mkdir()
    prepass_ops.ops_areanorm(src, a)
    prepass_ops.ops_areanorm(src, b, {"bfm": SPEC})
    za, zb = np.load(a), np.load(b)
    R = np.asarray(SPEC["R"], dtype=np.float64)
    Va, Vb = za["verts"].astype(np.float64), zb["verts"].astype(np.float64)
    na, nb = normals(Va, za["faces"]), normals(Vb, zb["faces"])
    na_u = na / np.linalg.norm(na, axis=1, keepdims=True)
    nb_u = nb / np.linalg.norm(nb, axis=1, keepdims=True)
    with np.load(DS / "ICT/topo/ict0000_GTready_original.npz") as z:
        ict_align, ict_nose = nose_alignment(z["V"].astype(np.float64), z["F"])
    bfm_align, bfm_nose = nose_alignment(Va, za["faces"])
    can_align, can_nose = nose_alignment(Vb, zb["faces"])
    out = {
        "evals_rel_maxdiff": float(np.abs(za["evals"] - zb["evals"]).max() / za["evals"].max()),
        "mass_rel_maxdiff": float(np.abs(za["mass"] - zb["mass"]).max() / za["mass"].max()),
        "verts_rotated_maxdiff": float(np.abs(Va @ R.T - Vb).max()),
        "normals_are_minus_rotated_cos_min": float((-(na_u @ R.T) * nb_u).sum(1).min()),
        "nose_z_sign": {"bfm": bfm_nose, "bfm_canon": can_nose, "ict": ict_nose},
        "mean_normal_dot_nose": {"bfm": bfm_align, "bfm_canon": can_align, "ict": ict_align},
    }
print(out)
ok = (out["evals_rel_maxdiff"] < 1e-4 and out["mass_rel_maxdiff"] < 1e-5
      and out["verts_rotated_maxdiff"] < 1e-6 and out["normals_are_minus_rotated_cos_min"] > 0.999
      and can_nose == ict_nose and np.sign(can_align) == np.sign(ict_align))
print("OK" if ok else "FALLITO")
raise SystemExit(0 if ok else 1)
