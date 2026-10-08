#!/usr/bin/env python3
"""Passo 1: la regione del volto in topologia FLAME, dalla maschera ufficiale.

    v3_work/unified_gt/run.sh v3_work/unified_gt/flame_region.py

Maschere ufficiali: ``v2_work/genflame/official/FLAME_masks.pkl`` (readme: "vertex indices for
different masks for the publicly available FLAME head model"). La maschera ``face`` (1.787 vertici)
va dalla fronte al mento e lateralmente fino a davanti alle orecchie (|x| <= 75 mm, le orecchie
partono da 63 mm ma su z piu' arretrate): non contiene orecchie, collo, nuca, ne' i bulbi oculari
(intersezioni misurate: 0 vertici con ``left_ear``/``right_ear``/``neck``/``*_eyeball``/
``boundary``, 23 con ``scalp`` sul bordo della fronte).

Resta da togliere l'INTERNO di occhi e bocca, che ``face`` contiene: il bordo palpebrale che gira
dietro la palpebra verso il bulbo e la superficie interna delle labbra (la bocca della media e'
chiusa). Regola geometrica, senza indici scelti a mano: un vertice e' esterno se e' visibile da
almeno una direzione entro 75 gradi da +z (64 direzioni di Fibonacci, raggio dalla superficie
spostata di 0.05 mm lungo la normale, occlusori = tutta la testa FLAME con i bulbi) con la normale
rivolta verso quella direzione. Poi i triangoli con tre vertici esterni e la sola componente
connessa piu' grande.

Uscita: ``datasets/UNIFIED_GT/flame_region.npz`` (``vidx`` indici FLAME dei vertici della regione,
``F`` triangoli reindicizzati, ``n_face_mask``) e ``aau/runs/evidence/e8/flame_region.png``.
Nessun vertice FLAME salvato: solo indici (la licenza vale per il modello).
"""

from __future__ import annotations

import pickle

import numpy as np

import ugt as C
import domains
import render as rd

MASKS = C.REPO_ROOT / "v2_work" / "genflame" / "official" / "FLAME_masks.pkl"
OUT = C.DATA_ROOT / "flame_region.npz"
def load_masks() -> dict:
    with open(MASKS, "rb") as fh:
        return {k: np.asarray(v, dtype=np.int64) for k, v in pickle.load(fh, encoding="latin1").items()}


def region() -> dict:
    tpl = domains.flame()
    V = tpl["V"] * 1000.0                  # mm
    F = tpl["F_render"]
    masks = load_masks()
    face = np.unique(masks["face"])
    excluded = {k: int(np.isin(face, masks[k]).sum()) for k in
                ("left_ear", "right_ear", "neck", "scalp", "boundary", "left_eyeball", "right_eyeball")}
    vis = C.exterior(V, F, face)
    keep = np.zeros(len(V), dtype=bool)
    keep[face[vis]] = True
    Fr = C.largest_component(F[keep[F].all(1)])
    vidx = np.unique(Fr)
    remap = -np.ones(len(V), dtype=np.int64)
    remap[vidx] = np.arange(len(vidx))
    Fr = remap[Fr]
    interior = face[~vis]
    info = {
        "n_face_mask": int(len(face)), "n_interior": int((~vis).sum()),
        "interior_in_lips": int(np.isin(interior, masks["lips"]).sum()),
        "interior_in_eye_region": int(np.isin(interior, masks["eye_region"]).sum()),
        "n_region": int(len(vidx)), "n_region_faces": int(len(Fr)),
        "dropped_by_component": int(keep.sum() - len(vidx)),
        "face_mask_intersections": excluded,
        "area_mm2": float(C.face_areas(V[vidx], Fr).sum()),
        "n_boundary_vertices": int(C.boundary_vertices(Fr, len(vidx)).sum()),
        "bbox_mm": (V[vidx].max(0) - V[vidx].min(0)).tolist(),
    }
    return {"vidx": vidx, "F": Fr, "interior": interior, "face": face, "info": info, "V": V, "F_all": F}


def figure(r: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    V, F = r["V"], r["F_all"]
    surf = C.Surface(V, F)
    in_region = np.zeros(len(V), dtype=bool)
    in_region[r["vidx"]] = True
    is_interior = np.zeros(len(V), dtype=bool)
    is_interior[r["interior"]] = True
    tri_region = in_region[F].all(1)
    tri_interior = is_interior[F].any(1)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, (title, R) in zip(axs, (("frontale", np.eye(3)), ("3/4 (yaw 45)", C.rot_y(-45)),
                                    ("profilo (yaw 90)", C.rot_y(-90)))):
        view = rd.framing(V, R, 700)
        out = rd.render(surf, view)
        img = out["img"].astype(float) / 255
        t = out["tri"]
        m_reg = (t >= 0) & tri_region[np.maximum(t, 0)]
        m_int = (t >= 0) & tri_interior[np.maximum(t, 0)]
        img[m_reg] = 0.55 * img[m_reg] + 0.45 * np.array([0.2, 0.5, 1.0])
        img[m_int] = 0.4 * img[m_int] + 0.6 * np.array([1.0, 0.15, 0.1])
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"Regione FLAME: {r['info']['n_region']} vertici (blu); rosso = interno escluso "
                 f"({r['info']['n_interior']} vertici della maschera face)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    r = region()
    C.save_npz(OUT, vidx=r["vidx"], F=r["F"], interior=r["interior"], face_mask=r["face"])
    C.save_json(C.DATA_ROOT / "flame_region.json", r["info"])
    figure(r, C.EVID_DIR / "flame_region.png")
    print(r["info"])


if __name__ == "__main__":
    main()
