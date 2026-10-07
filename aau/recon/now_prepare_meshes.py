#!/usr/bin/env python3
"""Patch volto canoniche per NoW: scansioni e ricostruzioni, stessa regola per tutte.

    aau/run.sh aau/recon/now_prepare_meshes.py --workers 16

Quattro passi, per ogni mesh (le 20 scansioni e ogni ricostruzione di ogni metodo):

1. **Frame canonico.**  Similarita' (senza riflessione) che porta i 7 landmark della mesh
   sul template T7 (``now_common.build_template``: media di Procrustes generalizzata dei 7
   landmark delle 20 scansioni, in mm).  La ricostruzione usa i SUOI landmark, mai quelli
   della scansione: e' un frame canonico, non un allineamento alla verita' a terra.
2. **Ritaglio.**  La regola di ``compute_mask`` del codice NoW (``now_common.now_mask``),
   calcolata sui landmark canonici della mesh stessa, poi ``crop_patch`` (facce coi tre
   vertici dentro, componente principale, schegge via).
3. **Decimazione** a ``TARGET_FACES`` = 5215 triangoli con ``mesh_ops.decimate_to``, il
   collasso quadrico di WS3b: le patch hanno la risoluzione di ``gt_face``.  Se il ritaglio
   ha meno triangoli del bersaglio (MICA: la topologia FLAME ne ha ~4k nel volto) prima si
   suddivide 1->4 a punto medio (``mesh_ops.subdivide_midpoint``), poi si decima.
4. **Verso dei triangoli**: coerente e verso l'esterno (``orient_outward``), dopo la
   decimazione, quindi senza toccare i vertici.

Uscite (fuori dal repo, ``now_common.WORK_ROOT``): ``template_lmk7.json``,
``scan_face/<soggetto>.npz``, ``recon_face/<metodo>/<nome>.npz``.  Nel repo: i controlli
``prep_<scan|metodo>.csv`` (scala, raggio del ritaglio, vertici, residuo dei landmark sul
template), che sono numeri e non dati.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import now_common as common  # noqa: E402

sys.path.insert(0, str(common.REPO_ROOT / "v2_work" / "genict"))
import mesh_ops as mo  # noqa: E402

PREP_FIELDS = ("name", "subject", "challenge", "scale_to_mm", "crop_radius_mm",
               "n_vertices_crop", "n_vertices", "n_faces", "lmk_residual_mm", "outward_area_before",
               "outward_area", "n_folds")

_STATE: dict = {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--methods", type=str, default=",".join(common.METHODS))
    p.add_argument("--target-faces", type=int, default=common.TARGET_FACES)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def crop_patch(V: np.ndarray, F: np.ndarray, centre: np.ndarray, radius: float):
    """Facce coi tre vertici entro ``radius`` dal centro, poi ``prepare_open_surface``.

    Non si usa ``ws3b_prepare_meshes.crop_region``: restituisce ``kept[kept2]`` con ``kept2``
    gia' nella numerazione compattata da ``remove_unreferenced``, quindi quando
    ``prepare_open_surface`` toglie una componente gli indici dei vertici slittano e la patch
    si riempie di triangoli spuri.  In WS3b gira sulle mesh medie, dove non toglie niente; qui
    sulle scansioni FaMoS_180507_03345_TA (un frammento di 23 facce) e FaMoS_180502_03341_TA
    (una faccia isolata) il bug scattava.  Qui i vertici escono gia' estratti, senza indici.
    """
    inside = np.linalg.norm(V - centre[None, :], axis=1) <= radius
    F = np.asarray(F, dtype=np.int64)
    F_in = F[inside[F].all(axis=1)]
    if len(F_in) == 0:
        raise RuntimeError(f"il ritaglio di raggio {radius} non contiene nessun triangolo")
    return mo.prepare_open_surface(V, F_in)


def make_manifold(V: np.ndarray, F: np.ndarray, max_rounds: int = 20):
    """Toglie facce duplicate, facce su spigoli con piu' di due facce e facce attorno ai
    vertici non manifold, finche' la mesh e' manifold; poi ripulisce.

    Le scansioni NoW non lo sono (spigoli non manifold, vertici a farfalla sul bordo,
    qualche faccia doppia), e ``igl.qslim`` -- che chiude il bordo su un vertice
    all'infinito -- su una mesh cosi' restituisce una mesh VUOTA senza errori.  Sulle
    ricostruzioni, topologie di modello pulite, non toglie niente.
    """
    import igl

    F = np.asarray(F, dtype=np.int64)
    _, first = np.unique(np.sort(F, axis=1), axis=0, return_index=True)
    F = F[np.sort(first)]
    for _ in range(max_rounds):
        V, F = mo.prepare_open_surface(V, F)
        F = np.asarray(F, dtype=np.int64)
        E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        _, inv, counts = np.unique(E, axis=0, return_inverse=True, return_counts=True)
        bad = (counts[inv.reshape(-1)].reshape(3, -1) > 2).any(axis=0)
        vm = igl.is_vertex_manifold(F)
        vm = np.asarray(vm[-1] if isinstance(vm, tuple) else vm).reshape(-1).astype(bool)
        bad |= ~vm[F].all(axis=1)
        if not bad.any():
            return np.asarray(V, dtype=np.float64), F
        F = F[~bad]
    raise RuntimeError(f"mesh ancora non manifold dopo {max_rounds} giri di pulizia")


def orient_outward(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, float]:
    """Facce orientate in modo coerente e con le normali verso l'esterno (+z, verso chi guarda).

    E' la convenzione ICT (frame y in su, naso +z, normali fuori), quella delle scansioni e di
    MICA.  3DDFA_V2, SynergyNet e PRNet, dopo la negazione della y, arrivano con le normali
    verso l'INTERNO (1-3% dell'area verso +z) e una scansione ha un orientamento misto (54%):
    Chamfer, ICP e render a due facce non lo vedono, gli operatori gradiente di DiffusionNet si'
    (il riferimento tangente segue la normale; vedi aau/runs/ws_frame).  Si applica DOPO la
    decimazione: i vertici non cambiano, cambia solo l'ordine dei vertici nei triangoli.
    Restituisce anche la frazione d'area verso +z prima della correzione, per il csv.
    """
    import igl

    def frac_out(G):
        n = np.cross(V[G[:, 1]] - V[G[:, 0]], V[G[:, 2]] - V[G[:, 0]])
        a = np.linalg.norm(n, axis=1)
        return float((a * (n[:, 2] > 0)).sum() / max(a.sum(), 1e-12))

    F = np.asarray(F, dtype=np.int64)
    before = frac_out(F)
    G = np.asarray(igl.bfs_orient(F)[0], dtype=np.int64)
    if frac_out(G) < 0.5:
        G = G[:, ::-1]
    return np.ascontiguousarray(G, dtype=np.int32), before


def patch_quality(V: np.ndarray, F: np.ndarray) -> tuple[float, int]:
    """(frazione d'area con normale verso +z, coppie di triangoli adiacenti con normali a piu'
    di 120 gradi): una patch volto frontale sana sta sopra 0.9 e ha poche pieghe."""
    import igl

    F = np.asarray(F, dtype=np.int64)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    a = np.linalg.norm(n, axis=1)
    nn = n / (a[:, None] + 1e-12)
    TT = np.asarray(igl.triangle_triangle_adjacency(F)[0])
    folds = sum(int(((nn[TT[:, k] >= 0] * nn[TT[TT[:, k] >= 0, k]]).sum(1) < -0.5).sum()) for k in range(3))
    return float((a * (n[:, 2] > 0)).sum() / max(a.sum(), 1e-12)), folds // 2


def canonical_patch(V, F, lmk, template, target_faces):
    """Frame canonico, ritaglio NoW, decimazione; piu' i controlli della riga di prep."""
    Vc, Lc, scale = common.to_canonical(V, lmk, template)
    centre, radius = common.now_mask(Lc)
    Vk, Fk = crop_patch(Vc, F, centre, radius)
    Vm, Fm = make_manifold(Vk, Fk)
    # MICA (FLAME, 9976 facce su tutta la testa) ha nel ritaglio meno triangoli del bersaglio:
    # suddivisione 1->4 a punto medio, che non sposta la geometria, poi la stessa decimazione.
    if len(Fm) < target_faces:
        Vm, Fm = mo.subdivide_midpoint(Vm, Fm, 1)
    Vd, Fd = mo.decimate_to(Vm, Fm, target_faces)
    if len(Fd) < 0.9 * target_faces:
        raise RuntimeError(f"decimazione a {len(Fd)} triangoli invece di {target_faces}")
    Fd, outward_before = orient_outward(Vd, Fd)
    outward, folds = patch_quality(Vd, Fd)
    return Vd, Fd, {
        "outward_area_before": outward_before, "outward_area": outward, "n_folds": folds,
        "scale_to_mm": float(scale), "crop_radius_mm": float(radius),
        "n_vertices_crop": int(len(Vk)), "n_vertices": int(len(Vd)), "n_faces": int(len(Fd)),
        "lmk_residual_mm": float(np.linalg.norm(Lc - template, axis=1).mean()),
    }


def _prep_recon(name: str) -> dict:
    method, template = _STATE["method"], _STATE["template"]
    V, F, lmk = common.load_recon(method, name)
    Vd, Fd, row = canonical_patch(V, F, lmk, template, _STATE["target_faces"])
    mo.save_variant(Vd, Fd, common.recon_face_dir(method) / f"{name}.npz")
    return {"name": name, **row}


def main() -> None:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    items = common.load_items()
    subjects = common.subjects_of(items)

    # --- scansioni e template -------------------------------------------------------
    t0 = time.time()
    scans = {s: common.load_scan(s) for s in subjects}
    template = common.build_template([scans[s][2] for s in subjects])
    common.template_path().parent.mkdir(parents=True, exist_ok=True)
    common.template_path().write_text(json.dumps({
        "landmarks_mm": template.tolist(), "names": list(common.LMK7_NAMES),
        "subjects": subjects, "ex_ex_mm": float(np.linalg.norm(template[0] - template[3])),
    }, indent=2), encoding="utf-8")
    print(f"[now-prep] template T7 da {len(subjects)} scansioni in {time.time() - t0:.0f}s: "
          f"ex-ex {np.linalg.norm(template[0] - template[3]):.1f} mm, "
          f"subnasale-radice {np.linalg.norm(template[4] - (template[1] + template[2]) / 2):.1f} mm",
          flush=True)

    out = common.scan_face_dir()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in subjects:
        V, F, lmk = scans[s]
        Vd, Fd, row = canonical_patch(V, F, lmk, template, args.target_faces)
        mo.save_variant(Vd, Fd, out / f"{s}.npz")
        rows.append({"name": s, "subject": s, "challenge": "", **row})
    common.write_rows(common.OUT_ROOT / "prep_scan.csv", PREP_FIELDS, rows,
                      target_faces=args.target_faces, template=str(common.template_path()))
    med = {k: float(np.median([r[k] for r in rows])) for k in PREP_FIELDS[3:]}
    print(f"[now-prep] scan_face: {len(rows)} scansioni, raggio mediano {med['crop_radius_mm']:.1f} mm, "
          f"{med['n_vertices_crop']:.0f} vertici nel ritaglio -> {med['n_vertices']:.0f} dopo la decimazione, "
          f"residuo landmark {med['lmk_residual_mm']:.2f} mm; verso+z minimo "
          f"{min(r['outward_area'] for r in rows):.3f}, pieghe massime {max(r['n_folds'] for r in rows)}", flush=True)

    # --- ricostruzioni -----------------------------------------------------------------
    by_name = {it.name: it for it in items}
    for method in methods:
        names = [it.name for it in items if (common.recon_dir(method) / f"{it.name}.npz").is_file()]
        if len(names) != len(items):
            print(f"[now-prep] ATTENZIONE {method}: {len(names)}/{len(items)} ricostruzioni", flush=True)
        common.recon_face_dir(method).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        _STATE.clear()
        _STATE.update(method=method, template=template, target_faces=args.target_faces)
        with mp.get_context("fork").Pool(processes=args.workers) as pool:
            rows = pool.map(_prep_recon, names, chunksize=8)
        rows = [{**r, "subject": by_name[r["name"]].subject,
                 "challenge": by_name[r["name"]].challenge} for r in rows]
        common.write_rows(common.OUT_ROOT / f"prep_{method}.csv", PREP_FIELDS, rows,
                          target_faces=args.target_faces, template=str(common.template_path()))
        med = {k: float(np.median([r[k] for r in rows])) for k in PREP_FIELDS[3:]}
        print(f"[now-prep] {method}: {len(rows)} mesh in {time.time() - t0:.0f}s; scala {med['scale_to_mm']:.3f} mm/px, "
              f"raggio mediano {med['crop_radius_mm']:.1f} mm, {med['n_vertices_crop']:.0f} vertici nel ritaglio -> "
              f"{med['n_vertices']:.0f}, residuo landmark {med['lmk_residual_mm']:.2f} mm; verso+z minimo "
              f"{min(r['outward_area'] for r in rows):.3f}, pieghe massime {max(r['n_folds'] for r in rows)}", flush=True)


if __name__ == "__main__":
    main()
