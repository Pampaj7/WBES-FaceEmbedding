#!/usr/bin/env python3
"""Mesh di training di ULRSSM: SOLO i generatori del nostro encoder (preset ``validated`` dello stream: BFM 2019, ICT,
GNM), con le stesse discretizzazioni e perturbazioni e la ricetta dei bracci decisivi C3F/C3M (``STREAM_RECIPE c3m``
di ``v3_work/stream/slurm/massive_node.sh``). Nessun dominio di valutazione, nessuna GT: ULRSSM e' non supervisionato.

    aau/run.sh aau/competitors/ulrssm/ulr_data.py --n-train 200 --n-val 10 --workers 32
    (ulr.sbatch, passo data; CPU)

Per dominio e gruppo g un generatore ``SeedSequence([seme, dominio, split, g])`` (split 0 = train, 1 = val: identita'
diverse) e le estrazioni del produttore, importate e non riscritte: ``src.identity`` (identita' fresca a code larghe,
``sources.MMSource``), poi ``producer.view_recipe`` con 6 viste per identita', quota d'espressioni del C3M per dominio
(``bfm2019=0,ict=0.2315,gnm=0.1968``), discretizzazioni senza reinserimento nel gruppo (``--label-draw perm``: una
vista per ognuna delle 6 etichette di ``views.LABELS``), mesh da ``views.discretize`` (original, remesh, down8k, noisy,
crop, up60k). Coordinate nel frame canonico dei generatori (scala della libreria: ULRSSM normalizza all'area 1).

Uscita ``datasets/ULRSSM_FACES/<split>/<dominio>_<g>_<vista>.npz`` (``V`` float32, ``F`` int32, etichetta,
espressione, seme del rumore) e ``datasets/ULRSSM_FACES/manifest.json`` (ricetta, semi, conteggi, vertici). Ripartibile:
i file gia' scritti si saltano (stessa sequenza di estrazioni, quindi stesse mesh).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

import ulr_common as U  # noqa: E402

sys.path.insert(0, str(U.REPO / "v3_work" / "stream"))
import producer  # noqa: E402  (sola lettura: view_recipe, parse_expr_frac)
import sources as S  # noqa: E402
import views as VW  # noqa: E402

DOMAINS = S.SOURCE_PRESETS["validated"]                    # bfm2019, ict, gnm
EXPR_FRAC = "bfm2019=0,ict=0.2315,gnm=0.1968"              # massive_node.sh, ricetta c3m
VIEWS_PER_ID = 6                                           # STREAM_VIEWS = STREAM_MESHES_PER_ID (C3M)
SPLITS = {"train": 0, "val": 1}
_C: dict = {}


def recipe_cfg() -> SimpleNamespace:
    """I campi della cfg del produttore che ``view_recipe`` legge (come ``producer.main`` con la ricetta c3m)."""
    w = np.asarray([VW.LABEL_WEIGHTS[k] for k in VW.LABELS])
    return SimpleNamespace(views=VIEWS_PER_ID, p_expr=0.6, expr_frac=producer.parse_expr_frac(EXPR_FRAC, DOMAINS),
                           labels=list(VW.LABELS), label_p=w / w.sum(), label_draw="perm")


def _group(task):
    """Le 6 viste del gruppo ``g`` del dominio ``d`` nello split: [(nome, n vertici, etichetta, espressione)]."""
    d, split, g = task
    src, cfg, out = _C["srcs"][d], _C["cfg"], U.DATA / split
    rng = np.random.default_rng(np.random.SeedSequence([_C["seed"], DOMAINS.index(d), SPLITS[split], g]))
    ident = src.identity(rng)
    rows = []
    for i, V, F, tag, label, noise_seed, _ in producer.view_recipe(src, ident, rng, cfg, d):
        name = f"{d}_{g:04d}_{i}"
        path = out / f"{name}.npz"
        if path.exists():
            with np.load(path) as z:
                if str(z["label"]) != label or int(z["noise_seed"]) != noise_seed:
                    raise SystemExit(f"{path}: gia' presente ma diverso dalla ricetta")
                rows.append((name, int(len(z["V"])), label, tag))
            continue
        Vd, Fd = VW.discretize(V, F, label, noise_seed)
        tmp = path.with_name(f".{name}.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, V=Vd.astype(np.float32), F=Fd.astype(np.int32), label=label, expr=tag,
                            noise_seed=noise_seed, domain=d, group=g, view=i)
        os.replace(tmp, path)
        rows.append((name, int(len(Vd)), label, tag))
    return d, split, g, rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-train", type=int, default=200, help="identita' di training per dominio")
    p.add_argument("--n-val", type=int, default=10, help="identita' di validazione per dominio (diverse)")
    p.add_argument("--seed", type=int, default=20261012)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    t0 = time.time()
    for s in SPLITS:
        (U.DATA / s).mkdir(parents=True, exist_ok=True)
    uni = S.Unified()
    _C.update(srcs=S.build_sources(list(DOMAINS), uni), cfg=recipe_cfg(), seed=a.seed)
    tasks = [(d, s, g) for s, n in (("train", a.n_train), ("val", a.n_val)) for d in DOMAINS for g in range(n)]
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = pool.map(_group, tasks, chunksize=1)
    files = {s: [] for s in SPLITS}
    for d, s, g, rows in res:
        files[s] += [{"name": n, "domain": d, "group": g, "n_vertices": nv, "label": lab, "expr": tag}
                     for n, nv, lab, tag in rows]
    cfg = _C["cfg"]
    man = {"domains": list(DOMAINS), "seed": a.seed, "n_train": a.n_train, "n_val": a.n_val,
           "recipe": {"views": cfg.views, "expr_frac": cfg.expr_frac, "labels": cfg.labels,
                      "label_draw": cfg.label_draw, "source": "v3_work/stream (sources.MMSource, producer.view_recipe, "
                                                              "views.discretize), ricetta c3m di massive_node.sh"},
           "sources": {d: _C["srcs"][d].describe() for d in DOMAINS}, "files": files,
           "n_meshes": {s: len(v) for s, v in files.items()},
           "vertices": {s: {"median": float(np.median([r["n_vertices"] for r in v])),
                            "max": int(max(r["n_vertices"] for r in v))} for s, v in files.items()},
           "seconds": time.time() - t0, **U.stamp()}
    U.atomic_json(U.DATA / "manifest.json", man)
    print(f"[ulr-data] {man['n_meshes']} mesh in {man['seconds']:.0f}s, vertici {man['vertices']}", flush=True)


if __name__ == "__main__":
    main()
