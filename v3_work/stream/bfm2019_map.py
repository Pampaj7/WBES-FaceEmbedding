#!/usr/bin/env python3
"""Mappa della GT unificata per BFM 2019, l'unico dominio di training dello stream senza mappa in
datasets/UNIFIED_GT/unified_space.npz (che ha il BFM 3DDFA dei dati REMESH, un'altra topologia).

    v3_work/unified_gt/run.sh v3_work/stream/bfm2019_map.py        (venv .venv_ugt: open3d e mediapipe)

Stessa procedura degli altri domini, chiamata cosi' com'e': ``v3_work/unified_gt/correspond.py::correspond``
(landmark col detector sul render frontale, similarita' robusta, NICP della regione FLAME guidato dai landmark,
punto piu' vicino sul template -> mappa baricentrica, copertura) con un template in piu' iniettato in
``domains.template``: la media di model2019_bfm (47.439 vertici, mm), tutti i triangoli.

Poi, come unified.py per gli altri: la mappa ristretta alla regione unificata (``ridx``) e la similarita'
canonica (pesata per l'area FLAME) dai punti mappati della media BFM 2019 alla media FLAME in mm. La regione e
mu NON cambiano (BFM 2019 non entra nella GPA): le GT degli altri domini restano quelle di oggi.

Uscite: ``datasets/STREAM/maps/bfm2019.npz`` (``vidx``, ``bary`` sulla regione unificata, numerazione di
model2019_bfm; ``covered``, ``dist_mm``; ``canon_s``/``canon_R``/``canon_t``/``canon_flip``) e ``.json`` coi
controlli di correspond (residuo sui landmark, anche tenuti fuori; Chamfer; copertura), figura in
``aau/runs/evidence/stream/corr_bfm2019.png``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "v3_work" / "unified_gt"))

import ugt as C  # noqa: E402
import domains  # noqa: E402

BFM2019_NPZ = REPO_ROOT / "external_data" / "bfm2019" / "model2019_bfm.npz"
OUT = REPO_ROOT / "datasets" / "STREAM" / "maps" / "bfm2019.npz"
EVID = REPO_ROOT / "aau" / "runs" / "evidence" / "stream"


def template_bfm2019() -> dict:
    """La media e i triangoli come li legge v3_work/mm/loaders.py::load_bfm2019 (mm, +y alto, +z naso)."""
    with np.load(BFM2019_NPZ) as z:
        mean = z["shape__model__mean"].astype(np.float64).reshape(-1, 3)
        F = z["shape__representer__cells"].T.astype(np.int64)
    return domains._pack("bfm2019", mean, F, F, None, units="millimetri", frame="BFM 2019: +y alto, +z naso",
                         note="media di model2019_bfm.npz (bfm2019_convert.py), 47.439 vertici, con orecchie")


def main() -> None:
    import correspond as CO
    import render as rd

    orig = domains.template
    domains.template = lambda name: template_bfm2019() if name == "bfm2019" else orig(name)
    C.EVID_DIR = EVID
    fl = CO.flame_frame()
    r = CO.correspond("bfm2019", fl, rd.face_oval_indices())
    a, info = r["arrays"], r["info"]
    sp = dict(np.load(C.DATA_ROOT / "unified_space.npz"))
    ridx = sp["ridx"]
    vidx, bary, covered = a["vidx"][ridx], a["bary"][ridx], a["covered"][ridx]
    # similarita' canonica come unified.py: punti mappati della media -> media FLAME (mm), pesi d'area FLAME
    tpl = template_bfm2019()
    M = C.bary_interp(tpl["V"], vidx, bary)
    P_flame = domains.flame()["V"][sp["flame_vidx"]] * 1000.0
    w_flame = C.vertex_areas(P_flame, sp["F"])
    s, R, t = C.umeyama(M, P_flame, w_flame)
    res = np.sqrt(((C.apply_sim(M, s, R, t) - P_flame) ** 2).sum(1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    C.save_npz(OUT, vidx=vidx.astype(np.int64), bary=bary, covered=covered, dist_mm=a["dist_mm"][ridx],
               canon_s=s, canon_R=R, canon_t=t, canon_flip=bool(info["canonical"]["flip_faces"]))
    out = {"domain": "bfm2019", "template": str(BFM2019_NPZ), "n_unified": int(len(ridx)),
           "unified_covered": int(covered.sum()), "unified_uncovered": int((~covered).sum()),
           "unified_dist_mm": {"median": float(np.median(a["dist_mm"][ridx])), "max": float(a["dist_mm"][ridx].max())},
           "canonical": {"s": float(s), "R": R.tolist(), "t": t.tolist(), "flip_faces": bool(info["canonical"]["flip_faces"]),
                         "residual_to_flame_mean_mm": {"rms_area_weighted": float(np.sqrt(np.average(res ** 2,
                                                                                                    weights=w_flame))),
                                                       "max": float(res.max())}},
           "correspond": info}
    C.save_json(OUT.with_suffix(".json"), out)
    L = info["landmarks"]
    print(json.dumps({k: out[k] for k in ("unified_covered", "unified_uncovered", "unified_dist_mm")}), flush=True)
    print(f"[bfm2019] landmark {L['n_used']}: residuo mediano {L['residual_after_similarity_mm']['median']:.2f} -> "
          f"{L['residual_after_nicp_mm']['median']:.2f} mm, tenuti fuori {L['held_out_after_nicp_mm']['median']:.2f} mm; "
          f"Chamfer {info['chamfer_mm']['symmetric_mean']:.3f} mm; canonica s={s:.5f} "
          f"rms {out['canonical']['residual_to_flame_mean_mm']['rms_area_weighted']:.2f} mm", flush=True)


if __name__ == "__main__":
    main()
