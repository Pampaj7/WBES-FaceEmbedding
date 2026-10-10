#!/usr/bin/env python3
"""Regione del volto su Ava-256: la regione FLAME e la regione unificata portate sulla topologia a 7306 vertici
con la procedura di ``v3_work/unified_gt/correspond.py``, la patch delle viste e il controllo con Multiface.

    v3_work/unified_gt/run.sh aau/ava256/ava_corr.py          (dopo ava_neutral.py template)

CONFERMATIVO: non valutare prima del protocollo confermativo (README). Solo dati e GT.

README, scelte 3 e 7. ``correspond.correspond`` e' importata e non riscritta: il template di Ava-256
(``ava_neutral.py template``, compattato sui 5741 vertici usati come la topologia di Multiface in
``domains.multiface``) le arriva registrando il dominio ``ava256`` in ``domains.template``; la figura dei landmark
(render del volto medio) va in ``datasets/AVA256/evidence`` invece che in ``aau/runs/evidence/e8``
(``ugt.EVID_DIR``). ``unified_space.npz`` e ``datasets/UNIFIED_GT/corr/`` non si toccano.

Patch delle viste (scelta 7): template nel frame FLAME con la similarita' dei landmark di ``correspond``, triangoli
esterni (``correspond.exterior_target``: via cavita' di bocca e orbite), vertici il cui punto piu' vicino sulla
regione FLAME deformata dal NICP non e' sul bordo ed e' entro ``COVER_MM`` (2 mm); triangoli con tre vertici dentro,
componente connessa piu' grande.

Controllo indipendente (scelta 3): la topologia di Ava-256 estende quella di Multiface (stessi indici per i 5471
vertici di Multiface), quindi la mappa di Multiface di ``unified_space.npz`` vale anche qui per identita' di indice;
si misura la distanza fra i punti delle due mappe sul template (mm nel frame FLAME).

Uscite (``datasets/AVA256/corr/``, indici di vertice ORIGINALI a 7306):
  - ``ava256.npz``: le uscite di ``correspond`` (``vidx``, ``bary``, ``covered``, ``Y_mm``, ...), i 1478 punti della
    regione unificata (``vidx_unified``, ``bary_unified``, ``covered_unified``), la mappa di Multiface
    (``vidx_multiface``, ``bary_multiface``), la patch (``patch_tri`` sui triangoli della topologia, ``patch_F``);
  - ``ava256.json``: i controlli di ``correspond`` (landmark, Chamfer, copertura, trasformazione canonica);
  - ``aau/ava256/corr_summary.json``: numeri aggregati, nessuna geometria.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
for _p in (THIS_DIR, THIS_DIR.parents[1] / "v3_work" / "unified_gt", THIS_DIR.parent / "multiface"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ava_common as ac  # noqa: E402
import correspond as CO  # noqa: E402
import domains  # noqa: E402
import nicp as NI  # noqa: E402
import render as rd  # noqa: E402
import ugt as C  # noqa: E402
from prepare_multiface import read_obj_faces  # noqa: E402
from shapes import Space  # noqa: E402

TEMPLATE = ac.DATA_ROOT / "template.npz"
CORR_DIR = ac.DATA_ROOT / "corr"
SUMMARY = ac.SUMMARY_DIR / "corr_summary.json"
MULTIFACE_FULL = domains.DATASETS / "Multiface" / "full"
NAME = "ava256"


def template() -> tuple[dict, np.ndarray, np.ndarray]:
    """Template compattato sui vertici usati (formato di ``domains``), indici originali dei vertici e topologia."""
    with np.load(TEMPLATE) as z:
        V, F = z["V"], z["F"]
    Vc, Fc, used = C.compact(V, F)
    tpl = domains._pack(NAME, Vc, Fc, Fc, None, units="unita' del file (verificate in ava_gt.py)",
                        frame="frame della testa del tracking di Meta (+y alto, +z avanti), prima cattura",
                        note="media GPA delle medie di EXP_neutral_peak (ava_neutral.py template), 5741 vertici usati")
    if len(tpl["F_target"]) != len(Fc):
        raise SystemExit("la topologia di Ava-256 non e' una sola componente connessa")
    return tpl, used, F


def multiface_map(sp: Space, used_ava: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Mappa unificata di Multiface in indici originali (7306): la compattazione di prepare_multiface
    (``prepare_open_surface``) tiene i vertici usati in ordine, controllata su una npz tracked preparata."""
    obj = sorted(MULTIFACE_FULL.glob("*/tracked_mesh/*/*.obj"))[0]
    Fm = read_obj_faces(obj).astype(np.int64)
    used_mf = np.unique(Fm)
    remap = -np.ones(ac.N_VERTS, dtype=np.int64)
    remap[used_mf] = np.arange(len(used_mf))
    paths = domains.multiface_neutral_paths()
    prep = paths[sorted(paths)[0]][0]
    with np.load(prep) as z:
        same = np.array_equal(np.asarray(z["F"], np.int64), remap[Fm])
    if not same or len(used_mf) != 5471:
        raise SystemExit("la compattazione della topologia Multiface non e' quella attesa")
    vidx = used_mf[sp.z["vidx_multiface"]]
    check = {"multiface_obj": str(obj.relative_to(domains.DATASETS)), "multiface_used_vertices": int(len(used_mf)),
             "multiface_used_in_ava": bool(np.isin(used_mf, used_ava).all()),
             "faces_equal_to_prep_npz": bool(same)}
    return vidx, sp.z["bary_multiface"], check


def patch(tpl: dict, fl: dict, arrays: dict) -> tuple[np.ndarray, dict]:
    """Triangoli (indici compattati) della patch delle viste (README, scelta 7) e i suoi controlli."""
    T = (float(arrays["T_landmark_s"]), arrays["T_landmark_R"], arrays["T_landmark_t"])
    V_t = C.apply_sim(tpl["V"], *T)
    F_ext, n_interior = CO.exterior_target(V_t, tpl["F_target"], tpl["F_render"])
    Y = arrays["Y_mm"]
    cl = C.Surface(Y, fl["Fr"]).closest(V_t)
    bnd = NI.on_boundary(fl["Fr"], cl["tri"], cl["bary"], C.boundary_edge_mask(fl["Fr"]),
                         C.boundary_vertices(fl["Fr"], len(Y)))
    inside = ~bnd & (cl["dist"] < CO.COVER_MM)
    Fp = C.largest_component(F_ext[inside[F_ext].all(1)])
    vp = np.unique(Fp)
    info = {"n_vertices": int(len(vp)), "n_faces": int(len(Fp)),
            "area_mm2_flame_frame": float(C.face_areas(V_t, Fp).sum()),
            "area_flame_region_deformed_mm2": float(C.face_areas(Y, fl["Fr"]).sum()),
            "bbox_mm_flame_frame": (V_t[vp].max(0) - V_t[vp].min(0)).tolist(),
            "n_boundary_vertices": int(C.boundary_vertices(Fp, len(V_t)).sum()),
            "template_vertices_dropped_interior": int(n_interior)}
    return Fp, info


def main() -> None:
    sp = Space()
    tpl, used, F = template()
    _orig = domains.template
    domains.template = lambda name: tpl if name == NAME else _orig(name)   # dominio in piu', solo qui
    C.EVID_DIR = ac.EVID_DIR                                               # render del volto medio: fuori da git
    fl = CO.flame_frame()
    oval = rd.face_oval_indices()
    r = CO.correspond(NAME, fl, oval)
    info, arrays = r["info"], r["arrays"]

    # indici compattati -> originali; restrizione ai punti della regione unificata
    vidx = used[arrays["vidx"]]
    ridx = sp.z["ridx"]
    out = {k: v for k, v in arrays.items() if k != "vidx"}
    out.update(vidx=vidx, vidx_unified=vidx[ridx], bary_unified=arrays["bary"][ridx],
               covered_unified=arrays["covered"][ridx], used=used)
    vidx_mf, bary_mf, mf_check = multiface_map(sp, used)
    out.update(vidx_multiface=vidx_mf, bary_multiface=bary_mf)

    # le due mappe sul template, nel frame FLAME della similarita' dei landmark (mm)
    with np.load(TEMPLATE) as z:
        V_full = z["V"]
    T = (float(arrays["T_landmark_s"]), arrays["T_landmark_R"], arrays["T_landmark_t"])
    V_t = C.apply_sim(V_full, *T)
    d_maps = np.linalg.norm(C.bary_interp(V_t, vidx[ridx], arrays["bary"][ridx]) -
                            C.bary_interp(V_t, vidx_mf, bary_mf), axis=1)

    Fp_c, pinfo = patch(tpl, fl, arrays)
    patch_tri = CO.tri_lookup(tpl["F_target"], Fp_c) >= 0             # righe di F_target = righe di F
    out.update(patch_tri=patch_tri, patch_F=used[Fp_c])
    C.save_npz(CORR_DIR / f"{NAME}.npz", **out)
    C.save_json(CORR_DIR / f"{NAME}.json", info)

    st = lambda x: {"median": float(np.median(x)), "p95": float(np.percentile(x, 95)),  # noqa: E731
                    "max": float(np.max(x))}
    L = info["landmarks"]
    summ = {
        "procedure": "v3_work/unified_gt/correspond.py (correspond, importata) sul template di ava_neutral.py",
        "landmarks": {"n_used": L["n_used"], "n_outliers_similarity": L["n_outliers_similarity"],
                      "residual_after_similarity_mm": L["residual_after_similarity_mm"]["median"],
                      "residual_after_nicp_mm": L["residual_after_nicp_mm"]["median"],
                      "held_out_before_mm": L["held_out_before_mm"]["median"],
                      "held_out_after_nicp_mm": L["held_out_after_nicp_mm"]["median"]},
        "chamfer_symmetric_mean_mm": info["chamfer_mm"]["symmetric_mean"],
        "distortion": info["distortion"], "coverage_flame_region": info["coverage"],
        "coverage_unified": {"n": int(len(ridx)), "n_covered": int(arrays["covered"][ridx].sum()),
                             "dist_mm_max": float(arrays["dist_mm"][ridx].max())},
        "canonical_similarity_to_flame_mean": {"s": info["canonical"]["s"], "rms_mm": info["canonical"]["rms_mm"],
                                               "flip_faces": info["canonical"]["flip_faces"]},
        "multiface_index_map": {**mf_check, "distance_to_new_map_mm_on_template": st(d_maps)},
        "reference_multiface_corr": "datasets/UNIFIED_GT/corr/multiface.json: landmark 337, NICP 0.56 mm, "
                                    "tenuti fuori 0.69 mm, Chamfer 0.264 mm, coperti 1669/1671",
        "patch": pinfo, "seconds": info["seconds"]}
    C.save_json(SUMMARY, summ)
    print(f"[ava-corr] landmark {L['n_used']}: residuo {L['residual_after_similarity_mm']['median']:.2f} -> "
          f"{L['residual_after_nicp_mm']['median']:.2f} mm (tenuti fuori {L['held_out_after_nicp_mm']['median']:.2f}); "
          f"Chamfer {info['chamfer_mm']['symmetric_mean']:.3f} mm; regione unificata coperta "
          f"{summ['coverage_unified']['n_covered']}/{len(ridx)}; s canonica {info['canonical']['s']:.4f}; "
          f"mappa Multiface: distanza mediana {np.median(d_maps):.2f} mm, max {d_maps.max():.2f}; patch "
          f"{pinfo['n_vertices']} vertici / {pinfo['n_faces']} triangoli", flush=True)


if __name__ == "__main__":
    main()
