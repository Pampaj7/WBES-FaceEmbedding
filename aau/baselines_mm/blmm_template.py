#!/usr/bin/env python3
"""NICP su template nei tre modi: template del dominio e iscrizione delle 600 mesh valutate di una vista.

    aau/outlineB/run_o3d.sh aau/baselines_mm/blmm_template.py --views hifi3d,faceverse --modes mm,cs --workers 32
    (blmm.sbatch, passo template; definizione: blmm.py, ``build_template`` ed ``enroll``)

Come ``aau/competitors/comp_template.py`` (stessi soggetti del template, stessi 4096 vertici ``rng(0)``, stesso
seme per mesh ``ir_template.mesh_seed``), con le coordinate del modo e, in ``mm`` e ``cs``, il ritorno al template
con la rigida invece della similarita'. Scrive ``<OUT_ROOT>/<vista>/<modo>/template.npz``: ``R`` (600, 4096, 3)
float32 in mm (``maxabs``: unita' maxabs), ``subjects``, ``topologies``, ``seeds``, ``seconds``, ``failed``, ``T``,
``template_subjects``, ``n_template_vertices``, ``unit_mm``. FaMoS escluso: le patch non hanno una topologia comune
da cui fare il template.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np

import blmm

_T: dict = {}


def _one(task):
    key, seed = task
    t0 = time.perf_counter()
    try:
        R = blmm.enroll(_T["X"][key], seed, _T["T"], _T["mode"])
        return R.astype(np.float32), time.perf_counter() - t0, ""
    except Exception as exc:  # noqa: BLE001  (la mesh resta NaN, contata)
        return np.full((blmm.N_POINTS, 3), np.nan, np.float32), time.perf_counter() - t0, f"{type(exc).__name__}: {exc}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--views", default="hifi3d,faceverse,facescape,facescape_expr")
    p.add_argument("--modes", default=",".join(blmm.MODES))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    prm = blmm.params()
    for view in a.views.split(","):
        subjects = blmm.subjects(view)
        keys = [(s, t) for s in subjects for t in blmm.TOPOLOGIES]
        for mode in a.modes.split(","):
            out = blmm.view_root(view) / mode / "template.npz"
            if out.exists() and not a.overwrite:
                print(f"[blmm-tpl] {out}: gia' presente, salto", flush=True)
                continue
            t0 = time.time()
            T, chosen, n_vert = blmm.build_template(view, mode, prm)
            if set(chosen) & set(subjects):
                raise SystemExit("template costruito con soggetti valutati")
            _T.update(mode=mode, T=T, X={k: blmm.work_coords(view, mode, blmm.mesh_path(view, *k), prm)[0] for k in keys})
            tasks = [(k, blmm.mesh_seed(*k)) for k in keys]
            with mp.get_context("fork").Pool(a.workers) as pool:
                res = pool.map(_one, tasks, chunksize=4)
            unit = blmm.unit(view, mode, prm)
            failed = [f"{k[0]}|{k[1]}|{r[2]}" for k, r in zip(keys, res) if r[2]]
            blmm.atomic_savez(out, R=np.stack([r[0] for r in res]) * np.float32(unit),
                              subjects=np.asarray([s for s, _ in keys]), topologies=np.asarray([t for _, t in keys]),
                              seeds=np.asarray([q for _, q in tasks]), seconds=np.asarray([r[1] for r in res]),
                              failed=np.asarray(failed), T=T * unit, template_subjects=np.asarray(chosen),
                              n_template_vertices=n_vert, unit_mm=unit, mode=mode)
            print(f"[blmm-tpl] {view} {mode}: {len(keys)} iscrizioni in {time.time() - t0:.0f}s, mediana "
                  f"{np.median([r[1] for r in res]):.2f}s, fallite {len(failed)}" + (f" ({failed[:2]})" if failed else ""),
                  flush=True)


if __name__ == "__main__":
    main()
