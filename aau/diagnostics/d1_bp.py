#!/usr/bin/env python3
"""D1 (PROTOCOL_D.md sez. 3.5): variante B dei concorrenti parametrici sugli insiemi FLAME e distanze B sui visti.

    aau/outlineB/run_o3d.sh aau/diagnostics/d1_bp.py static
    aau/outlineB/run_o3d.sh aau/diagnostics/d1_bp.py fit --sets flame2023_s1,flame2023 --workers 64
    (diag.sbatch, passi bp_static e bp_fit)

``aau/baselines_param`` importato, non modificato: ``bp`` (regione, contesto, distanze), ``bp_e2.fit_items`` (NICP,
variante A, variante B coi valori congelati ``bp.LOOP``, fallimenti di ``bp_fit_e1.failure``), senza espressione
(``free=False``) come gli held-out dell'emendamento 2.

``static``: i fit B gia' calcolati sugli held-out (``baselines_param/calib_e2/<modello>.npz``: beta di B e regione per
dominio), distanze ``bp.mesh_distances`` per dominio -> ``d1/bp/static_<modello>.npz`` (``<dominio>_names``,
``<dominio>_D_sr``, ``<dominio>_D_fr``, nei nomi ordinati).
``fit``: per insieme FLAME, vista in memoria (dominio del frame ``flame``, mesh di ``datasets/DIAG_D1/in``), L e CS_ref
= mediane di maxabs e cs delle original (``blmm.mesh_scalars``, mm, come fact_calib.baselines), regione con
riferimenti = le 200 original (``bp.region``), fit di entrambi i modelli -> ``d1/bp/<insieme>_<modello>.npz`` (beta,
fallimenti, tempi, ``D_sr``, ``D_fr`` nei nomi ordinati). Ripartibile: le uscite gia' su disco si saltano.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

import diag

sys.path.insert(0, str(diag.REPO / "aau/baselines_param"))
import bp  # noqa: E402
import bp_e2  # noqa: E402
import blmm  # noqa: E402  (aau/baselines_mm, messo su sys.path da bp)

OUT = diag.D1 / "bp"
HELDOUT_DOMS = ("bfm", "ict", "gnm")


def view_of(name: str, frame: str, root: Path) -> str:
    """Vista in memoria per blmm.work_coords / to_mm (come bp_e2.run_heldout)."""
    v = f"d1_{name}"
    blmm.VIEWS[v] = {"dir": root, "domain": frame, "gt": None, "template_dir": None, "fb_root": None,
                     "template_ref": None}
    return v


def run_static() -> None:
    bl = json.loads(bp_e2.BL_PARAMS.read_text())["domains"]
    prm = {"domains": {d: {"L": float(bl[d]["L"]), "cs_ref": float(bl[d]["cs_ref"])} for d in HELDOUT_DOMS}}
    for name in diag.BP_MODELS:
        out = OUT / f"static_{name}.npz"
        if out.exists():
            print(f"[d1-bp] {out}: gia' presente, salto", flush=True)
            continue
        t0 = time.time()
        with np.load(diag.CALIB_E2 / f"{name}.npz", allow_pickle=True) as z:
            names, dom, beta = [str(x) for x in z["names"]], np.asarray(z["domain"]).astype(str), z["beta"]
            regions = {d: np.asarray(z[f"region_{d}"]) for d in HELDOUT_DOMS}
        arrays = {}
        for d in HELDOUT_DOMS:
            idx = np.flatnonzero(dom == d)
            idx = idx[np.argsort([names[k] for k in idx])]
            ctx = bp.context(view_of(f"heldout_{d}", d, diag.RAW), name, regions[d], prm)
            D = bp.mesh_distances(np.asarray(beta[idx], np.float64), ctx)
            arrays.update({f"{d}_names": np.asarray([names[k] for k in idx]), f"{d}_D_sr": D["sr"], f"{d}_D_fr": D["fr"],
                           f"{d}_n_failed": int((~np.isfinite(beta[idx]).all(1)).sum())})
            print(f"[d1-bp] static {name} {d}: {len(idx)} mesh, beta non finiti {arrays[f'{d}_n_failed']}", flush=True)
        diag.atomic_savez(out, source=str(diag.CALIB_E2 / f"{name}.npz"), loop=json.dumps(bp.LOOP[name]), **arrays)
        print(f"[d1-bp] static {name}: {time.time() - t0:.0f}s -> {out}", flush=True)


def _scalars(path: str):
    V, F = blmm.load_raw(Path(path))
    return blmm.mesh_scalars(blmm.to_mm(_VIEW[0], V), F)


_VIEW: list = []


def run_fit(sets: list, workers: int) -> None:
    for s in sets:
        names = diag.set_names(s)
        view = view_of(s, diag.SETS[s]["frame"], diag.RAW_IN)
        _VIEW[:] = [view]
        orig = [n for n in names if n.endswith("_GTready_original.npz")]
        with mp.get_context("fork").Pool(workers) as pool:
            sc = pool.map(_scalars, [str(diag.RAW_IN / n) for n in orig], chunksize=2)
        prm = {"domains": {diag.SETS[s]["frame"]: {"L": float(np.median([x["maxabs"] for x in sc])),
                                                  "cs_ref": float(np.median([x["cs"] for x in sc]))}}}
        refs = []
        for n in orig:
            V, F = blmm.load_raw(diag.RAW_IN / n)
            refs.append((n, blmm.to_mm(view, V), F))
        items = [(diag.split_name(n)[0], diag.split_name(n)[1], diag.RAW_IN / n) for n in names]
        for name in diag.BP_MODELS:
            out = OUT / f"{s}_{name}.npz"
            if out.exists():
                print(f"[d1-bp] {out}: gia' presente, salto", flush=True)
                continue
            t0 = time.time()
            reg = bp.region(view, bp.load_model(name), refs)
            ctx = bp.context(view, name, reg["vertices"], prm)
            arr = bp_e2.fit_items(view, name, items, ctx, prm, bp.LOOP[name], False, workers)
            bp_e2.summary("d1", view, name, arr, t0)
            D = bp.mesh_distances(arr["vb_beta"], ctx)
            diag.atomic_savez(out, names=np.asarray(names), vb_beta=arr["vb_beta"], failed_vb=arr["failed_vb"],
                              failed_va=arr["failed_va"], t_stage=arr["t_stage"], stages=arr["stages"],
                              nicp_resid_mm=arr["nicp_resid_mm"], vb_kept=arr["vb_kept"], vb_rms=arr["vb_rms"],
                              D_sr=D["sr"], D_fr=D["fr"], region_vertices=reg["vertices"], loop=json.dumps(bp.LOOP[name]),
                              params=json.dumps(prm), model_sha256=arr["model_sha256"],
                              region_diag=json.dumps({"n_refs": reg["n_refs"], "n_vertices": int(len(reg["vertices"])),
                                                      "kept_median": float(np.median([x["kept"] for x in reg["diag"]])),
                                                      "median_mm_median": float(np.median([x["median_mm"]
                                                                                           for x in reg["diag"]]))}),
                              wall_s=time.time() - t0, workers=workers)
            print(f"[d1-bp] {s} {name}: regione {len(reg['vertices'])} v., B fallite {len(arr['failed_vb'])}, "
                  f"{time.time() - t0:.0f}s -> {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=("static", "fit"))
    ap.add_argument("--sets", default=",".join(diag.FLAME_SETS))
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    if a.step == "static":
        run_static()
    else:
        run_fit(a.sets.split(","), a.workers)


if __name__ == "__main__":
    main()
