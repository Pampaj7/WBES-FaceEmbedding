#!/usr/bin/env python3
"""Crop del volto FLAME per WS-FLAME, sopra il loader di ``v2_work/genflame/flame_model.py``.

Il modello si carica con ``load_flame``/``model_path`` di genflame, importati e non
riscritti: ``$WBES_FLAME_MODEL`` se impostata, altrimenti
``v2_work/genflame/official/FLAME2020/generic_model.pkl``, e nient'altro. LICENZA: nessuno
dei due file e' nel repo; li mette l'utente con la sua licenza MPI.

Il crop storico (``v2_work/genflame/make_flame_topologies.face_crop_faces``) passa dalla
corrispondenza ``BFM_to_FLAME_corr.npz``, che e' un asset FLAME e qui non si usa (e quello
script non gira comunque in questo checkout: vuole open3d e
``datasets.expand_remesh_topologies``, che non c'e'). Si usa invece la regione ``face`` della
maschera ufficiale ``FLAME_masks.pkl`` (``$WBES_FLAME_MASKS``, altrimenti ``OFFICIAL_MASKS``):
triangoli con tutti e tre i vertici nella regione, poi la sola componente connessa piu'
grande. L'insieme di indici dipende solo dalla topologia e dalla maschera, non dai
coefficienti: e' lo stesso per ogni identita', cioe' e' una topologia.
``head`` (testa intera, senza i bulbi oculari) e' una prova senza maschera, NON il
protocollo: quei numeri non si confrontano con BFM e ICT.

    aau/run.sh aau/flame/flame_mask.py       # self-check di genflame + crop
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "v2_work" / "genflame"))

import flame_model as genflame  # noqa: E402
from flame_model import _dechumpy, _Unpickler, load_flame, model_path  # noqa: E402,F401

OFFICIAL_MASKS = genflame.OFFICIAL_MODEL.parents[1] / "FLAME_masks.pkl"
CROPS = ("mask", "head")


def masks_path() -> Path:
    """$WBES_FLAME_MASKS se impostata, altrimenti OFFICIAL_MASKS; nient'altro (licenza)."""
    env = os.environ.get("WBES_FLAME_MASKS", "").strip()
    path = Path(env) if env else OFFICIAL_MASKS
    if not path.is_file():
        raise FileNotFoundError(f"maschere FLAME non trovate in {path}: imposta WBES_FLAME_MASKS "
                                f"o metti il FLAME_masks.pkl della licenza in {OFFICIAL_MASKS}")
    return path


def load_mask_region(path: Path, region: str, n_verts: int) -> np.ndarray:
    """Indici dei vertici di una regione di ``FLAME_masks.pkl`` (dict regione -> indici)."""
    with open(path, "rb") as fh:
        masks = _Unpickler(fh, encoding="latin1").load()
    if region not in masks:
        raise SystemExit(f"ERRORE: regione '{region}' assente in {path}; "
                         f"regioni disponibili: {sorted(masks)}")
    idx = np.unique(np.asarray(_dechumpy(masks[region]), dtype=np.int64).ravel())
    if idx.size == 0 or idx.min() < 0 or idx.max() >= n_verts:
        raise SystemExit(f"ERRORE: regione '{region}' con indici fuori da [0, {n_verts})")
    return idx


def _largest_component(F: np.ndarray) -> np.ndarray:
    import igl

    n_comp, comp = igl.facet_components(np.asarray(F, dtype=np.int64))
    if n_comp <= 1:
        return F
    comp = np.asarray(comp)
    return np.ascontiguousarray(F[comp == int(np.argmax(np.bincount(comp)))])


def crop_faces(F: np.ndarray, n_verts: int, crop: str, region: str = "face") -> np.ndarray:
    """Triangoli della regione di valutazione, in indici della topologia FLAME completa."""
    F = np.asarray(F, dtype=np.int32)
    if crop == "mask":
        keep = np.zeros(n_verts, dtype=bool)
        keep[load_mask_region(masks_path(), region, n_verts)] = True
        F = F[keep[F].all(axis=1)]
    elif crop != "head":
        raise ValueError(f"crop sconosciuto: {crop} (attesi {CROPS})")
    if len(F) == 0:
        raise SystemExit("ERRORE: il crop non lascia nessun triangolo")
    return _largest_component(F)


def compact(V: np.ndarray, F_crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """La mesh ristretta a ``F_crop``, vertici ricompattati (ordine originale)."""
    used = np.unique(F_crop)
    remap = np.full(len(V), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return np.ascontiguousarray(V[used]), np.ascontiguousarray(remap[F_crop])


if __name__ == "__main__":
    genflame._self_check()
    m = load_flame(model_path())
    Fc = crop_faces(m["f"], len(m["v_template"]), "mask",
                    os.environ.get("WBES_FLAME_MASK_REGION", "face"))
    Vc, Fc = compact(m["v_template"], Fc)
    print(f"crop mask ({masks_path()}): {len(Vc)} vertici, {len(Fc)} triangoli")
