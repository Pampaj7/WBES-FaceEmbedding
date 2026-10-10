#!/usr/bin/env python3
"""Per mesh: centroid size e sqrt(area) robuste, altezza (mm, frame canonico); per dominio L_d e CS_ref.

    aau/outlineB/run_o3d.sh aau/baselines_mm/blmm_scalars.py --workers 32          (blmm.sbatch, passo scalars)

  - ``<OUT_ROOT>/<vista>/scalars.npz``: ``paths`` (relativi alla radice: le mesh valutate della vista, le original
    dei soggetti del template e quelle di riferimento del dominio), colonne di ``blmm.mesh_scalars`` (mm).
  - ``<OUT_ROOT>/params.json``: per dominio ``L`` = mediana di ``maxabs`` e ``cs_ref`` = mediana di ``cs`` sulle
    original dei 100 soggetti valutati della vista neutra (FaMoS: le 30 mesh di galleria). Una costante per
    dominio: unita' di lavoro dei modi ``mm`` e ``cs`` (``blmm.py``). Non si sovrascrive con valori diversi.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

import blmm


def _one(task):
    view, path = task
    V, F = blmm.load_raw(Path(path))
    return blmm.mesh_scalars(blmm.to_mm(view, V, Path(path).stem), F)


def view_paths(view: str) -> tuple[list[Path], list[Path]]:
    """(mesh della vista + template, original di riferimento del dominio)."""
    d = blmm.VIEWS[view]
    if view == "famos":
        rows = blmm.famos_manifest()
        paths = [d["dir"] / f"{r['name']}.npz" for r in rows]
        return paths, [d["dir"] / f"{r['name']}.npz" for r in rows if r["role"] == "gallery"]
    subj = blmm.subjects(view)
    paths = [blmm.mesh_path(view, s, t) for s in subj for t in blmm.TOPOLOGIES]
    ref = [blmm.mesh_path(view, s, "original", template=True) for s in subj]
    tpl = [blmm.mesh_path(view, s, "original", template=True) for s in blmm.template_subjects(view)]
    return list(dict.fromkeys(paths + ref + tpl)), ref


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--views", default=",".join(blmm.VIEWS))
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    prm = {"definition": "L = mediana di max|V_mm - centro dei vertici|, cs_ref = mediana della centroid size robusta "
                         "(blmm.mesh_scalars), mm, sulle original dei 100 soggetti valutati della vista neutra (FaMoS: "
                         "galleria)", "frames": str(blmm.FRAMES_JSON), "smooth_k": blmm.SMOOTH_K, "domains": {}}
    with mp.get_context("fork").Pool(a.workers) as pool:
        for view in a.views.split(","):
            paths, ref = view_paths(view)
            res = pool.map(_one, [(view, str(q)) for q in paths], chunksize=2)
            arr = {k: np.asarray([r[k] for r in res]) for k in res[0]}
            blmm.atomic_savez(blmm.OUT_ROOT / view / "scalars.npz", paths=np.asarray([blmm.rel(q) for q in paths]), **arr)
            pos = {q: k for k, q in enumerate(paths)}
            info = {"n": len(paths)}
            if view != "famos":
                ii = np.asarray([pos[blmm.mesh_path(view, s, t)] for s in blmm.subjects(view) for t in blmm.TOPOLOGIES])
                for c in ("cs", "cs_mass", "sqrt_area", "sqrt_area_mass"):
                    x = arr[c][ii].reshape(-1, len(blmm.TOPOLOGIES))
                    info[f"{c}_topo_cv_median"] = float(np.median(x.std(1, ddof=1) / x.mean(1)))
                    info[f"{c}_by_topology_median"] = {t: round(float(v), 2) for t, v in zip(blmm.TOPOLOGIES, np.median(x, 0))}
            print(f"[blmm-scalars] {view}: {json.dumps(info)}", flush=True)
            dom = blmm.VIEWS[view]["domain"]
            if dom not in prm["domains"]:
                rr = [res[pos[q]] for q in ref]
                prm["domains"][dom] = {"L": float(np.median([r["maxabs"] for r in rr])),
                                       "cs_ref": float(np.median([r["cs"] for r in rr])), "n_ref": len(rr),
                                       "ref_dir": blmm.rel(ref[0].parent)}
                print(f"[blmm-scalars] {dom}: {prm['domains'][dom]}", flush=True)
    old = json.loads(blmm.PARAMS_JSON.read_text()) if blmm.PARAMS_JSON.exists() else None
    if old and any(old["domains"].get(d) not in (None, v) for d, v in prm["domains"].items()):
        raise SystemExit(f"{blmm.PARAMS_JSON}: L o cs_ref diversi da quelli gia' usati: non sovrascrivo")
    if old:
        prm["domains"] = {**old["domains"], **prm["domains"]}
    blmm.PARAMS_JSON.write_text(json.dumps(prm, indent=1) + "\n")


if __name__ == "__main__":
    main()
