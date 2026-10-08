#!/usr/bin/env python3
"""GT unificata di un set di valutazione, nel formato della pipeline ``aau/zs3dmm`` (``gt_matrix.npz``).

    v3_work/unified_gt/run.sh v3_work/unified_gt/make_eval_gt.py --domain hifi3d
    v3_work/unified_gt/run.sh v3_work/unified_gt/make_eval_gt.py --domain facescape --ids id930001,id930004
    # in python:  from make_eval_gt import unified_gt;  D_mm, names = unified_gt("faceverse", ["id910001", ...])

Formato: npz con ``D_orig`` float32 (n, n), diviso per il massimo, e ``names`` (``idNNNNNN``), come
``<vista>/gt_matrix.npz`` e letto da ``load_gt_distance_matrix``; accanto un json con ``mm_per_unit``.
Senza ``--ids`` i nomi sono QUELLI DELLA VISTA, nello stesso ordine (il pool di 500): il file si puo'
mettere al posto di ``ZS_DIST_NPZ`` / ``--gt`` dei summarizer, o passare a ``zs_summarize.with_gt`` per
sostituire la GT nelle righe gia' calcolate (come fa ``eval_methods.py``).

Domini e identita' (pesi delle identita' su disco, stesso modello della vista):
  - hifi3d     id900000+i -> datasets/HIFI3D/identities/hifiNNNN.npz, testa intera dai pesi (la patch
               della vista non contiene tutti i vertici della mappa: 16 punti su 1.478 cadono fuori);
  - faceverse  id910000+i -> datasets/FACEVERSE_ZS/identities/fvNNNN.npz (mesh piena);
  - facescape  id930000+i -> datasets/DEV_FACESCAPE/identities/fsNNNN.npz (z standardizzati): forma
               neutra ``core[:, 0, :] @ (id_mean + sqrt(id_var) z)`` sui soli vertici della mappa; la
               base ristretta si calcola una volta dal core (4.9 GB) e resta in
               ``datasets/UNIFIED_GT/cache/facescape_unified_basis.npz`` (solo locale: licenza FaceScape).
Controllo per ogni chiamata: per 3 identita' la forma ricostruita contro la patch salvata, sui vertici in
comune (max |diff| nel json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import ugt as C
import domains
from shapes import Space, unified_distances

EVAL_DIR = C.DATA_ROOT / "eval"
FS_BASIS = C.DATA_ROOT / "cache" / "facescape_unified_basis.npz"
SETS = {"hifi3d": ("HIFI3D", "hifi", 900000), "faceverse": ("FACEVERSE_ZS", "fv", 910000),
        "facescape": ("DEV_FACESCAPE", "fs", 930000)}


def _identity_file(domain: str, sid: str) -> Path:
    root, prefix, off = SETS[domain]
    return domains.DATASETS / root / "identities" / f"{prefix}{int(sid[2:]) - off:04d}.npz"


def _facescape_basis(sp: Space) -> dict:
    """Base neutra di FaceScape ristretta ai vertici della mappa (cache)."""
    vidx = sp.z["vidx_facescape"]
    need = np.unique(vidx)
    if FS_BASIS.exists():
        with np.load(FS_BASIS) as z:
            if np.array_equal(z["vertices"], need):
                return {k: z[k] for k in z.files}
    with np.load(domains.FS_NPZ, allow_pickle=True) as z:
        core = z["shape_bm_core"]
        rows = (3 * need[:, None] + np.arange(3)[None]).ravel()
        Cn = core[rows, 0, :].astype(np.float64)                       # (3m, k), neutra
        id_mean = z["id_mean"].astype(np.float64)
        id_sigma = np.sqrt(z["id_var"].astype(np.float64))
    out = {"vertices": need, "mean": (Cn @ id_mean).reshape(-1, 3),
           "basis": (Cn * id_sigma).reshape(len(need), 3, -1)}
    FS_BASIS.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FS_BASIS, **out)
    return out


def mapped_points(domain: str, sids: list[str], sp: Space) -> tuple[np.ndarray, dict]:
    """Punti della regione unificata (n, m, 3) per le identita' ``sids`` del dominio, e controlli."""
    W = np.stack([np.load(_identity_file(domain, s))["weights"].astype(np.float64) for s in sids])
    checks = {}
    if domain in ("hifi3d", "faceverse"):
        tpl = domains.template(domain)
        M0, B = sp.linear(domain, tpl)
        P = M0[None] + np.einsum("nk,kvd->nvd", W[:, : B.shape[0]], B)
        from hifi_model import face_patch
        import hifi_model
        import fv_model
        m = hifi_model.load_hifi(domains.HIFI_MAT) if domain == "hifi3d" else fv_model.load_fv(domains.FV_NPY)
        diff = 0.0
        for j in range(min(3, len(sids))):
            Vp, _ = face_patch(tpl["V"] + tpl["basis"][:, :, : W.shape[1]] @ W[j], m)
            diff = max(diff, float(np.abs(Vp - np.load(_identity_file(domain, sids[j]))["V"]).max()))
        checks["weights_vs_stored_max_abs"] = diff
    elif domain == "facescape":
        fb = _facescape_basis(sp)
        pos = {v: i for i, v in enumerate(fb["vertices"])}
        loc = np.vectorize(pos.get)(sp.z["vidx_facescape"])
        Vn = fb["mean"][None] + np.einsum("nk,vdk->nvd", W, fb["basis"])  # (n, m_vert, 3)
        P = np.einsum("nvkd,vk->nvd", Vn[:, loc], sp.z["bary_facescape"])
        # controllo: patch salvata contro la ricostruzione, sui vertici della mappa dentro la patch
        root = domains.DATASETS / SETS["facescape"][0] / "identities"
        fvi = np.load(root / "face_vertex_indices.npy")
        inv = {int(v): i for i, v in enumerate(fvi)}
        common = [k for k, v in enumerate(fb["vertices"]) if int(v) in inv]
        diff = 0.0
        for j in range(min(3, len(sids))):
            Vs = np.load(_identity_file(domain, sids[j]))["V"].astype(np.float64)
            diff = max(diff, float(np.abs(Vn[j, common] - Vs[[inv[int(fb["vertices"][k])] for k in common]]).max()))
        checks["weights_vs_stored_max_abs_mm"] = diff
        checks["map_vertices_inside_dev_patch"] = f"{len(common)}/{len(fb['vertices'])}"
    else:
        raise SystemExit(f"dominio non gestito: {domain}")
    return P, checks


def unified_gt(domain: str, sids: list[str], sp: Space | None = None) -> tuple[np.ndarray, list[str], dict]:
    """(D in mm (n, n), nomi, controlli): g_ij = ||s_i - s_j|| / sqrt(A), come shapes.py."""
    sp = sp or Space()
    P, checks = mapped_points(domain, sids, sp)
    a, _, _ = sp.align(P)
    S = sp.svec(a)
    D = unified_distances(S, S, sp.A)
    np.fill_diagonal(D, 0.0)
    return 0.5 * (D + D.T), list(sids), checks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--domain", required=True, choices=sorted(SETS))
    p.add_argument("--ids", default="", help="id separati da virgole; vuoto = i nomi della vista, nel loro ordine")
    p.add_argument("--out", type=Path, default=None, help="default datasets/UNIFIED_GT/eval/<dominio>_gt_matrix.npz")
    a = p.parse_args()
    view_gt = domains.DATASETS / SETS[a.domain][0] / "eval_view" / "gt_matrix.npz"
    if a.ids:
        names = np.array([s.strip() for s in a.ids.split(",") if s.strip()])
    else:
        with np.load(view_gt) as z:
            names = z["names"]
    sids = [str(s).split("_GTready")[0] for s in names]
    D, _, checks = unified_gt(a.domain, sids)
    gmax = float(D.max())
    out = a.out or EVAL_DIR / f"{a.domain}_gt_matrix.npz"
    C.save_npz(out, D_orig=(D / gmax).astype(np.float32), names=names)
    iu = np.triu_indices(len(D), 1)
    man = {"domain": a.domain, "n": len(sids), "mm_per_unit": gmax, "names_from": "--ids" if a.ids else str(view_gt),
           "definition": "GT unificata (v3_work/unified_gt): RMS pesato per area in mm sulla regione FLAME comune dopo "
                         "Procrustes di similarita' verso mu; D_orig * mm_per_unit = mm",
           "median_mm": float(np.median(D[iu])), "min_offdiag_mm": float(D[iu].min()), **checks}
    out.with_suffix(".json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
