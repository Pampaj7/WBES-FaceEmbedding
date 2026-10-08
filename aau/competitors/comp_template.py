#!/usr/bin/env python3
"""Baseline "NICP su template" di indomain_recog sulle 600 mesh HIFI3D valutate: template e iscrizioni.

    aau/outlineB/run_o3d.sh aau/competitors/comp_template.py --view-dir datasets/HIFI3D/eval_view/npz \\
        --out aau/runs/competitors_hifi3d/template_hifi3d.npz --workers 32
    (comp_template.sbatch; protocollo: aau/runs/competitors_hifi3d/protocol.md, aggiunta delle 14:28)

Iscrizione e distanza sono quelle di ``aau/indomain/ir_template.py``, importate (``_init``,
``_enroll_one`` -> ``enroll``; ``template_distances`` nel summary), con lo stesso seme per mesh
(``mesh_seed``). Cambia solo da dove viene il template: HIFI3D non ha soggetti di training, quindi la
media delle ``original`` (maxabs) e' su 100 soggetti del pool che NON sono fra i 100 valutati, scelti con
``rng(1234)``; poi maxabs e 4096 vertici con ``rng(0)``, come ``build_template``.

Scrive ``--out`` con ``R`` (600, 4096, 3) float32, ``subjects``, ``topologies``, ``seconds``, ``failed``,
e il template ``T`` con i soggetti da cui viene.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "indomain"))
sys.path.insert(0, str(THIS_DIR.parent / "zs3dmm"))

import ir_template as irt  # noqa: E402  (mette faceBench sul path via ir_simicp)
from zs_stage import TOPOLOGIES, select_subjects  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--view-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--eval-seed", type=int, default=1234, help="WBES_EVAL_SEED: scelta dei soggetti")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="solo le prime N mesh (prova)")
    return p.parse_args()


def build_template(view_dir: Path, evaluated: list[str]) -> tuple[np.ndarray, list[str], int]:
    """``ir_template.build_template`` con i soggetti non valutati del pool al posto di quelli di training."""
    from mesh_npz_utils import normalize_vertices

    pool = sorted({p.name.split("_GTready_")[0] for p in view_dir.glob("id*_GTready_original.npz")})
    others = sorted(set(pool) - set(evaluated))
    chosen = sorted(np.random.default_rng(1234).choice(others, irt.N_TEMPLATE_SUBJECTS, replace=False).tolist())
    Vs = []
    for s in chosen:
        with np.load(view_dir / f"{s}_GTready_original.npz") as d:
            Vs.append(normalize_vertices(np.asarray(d["V"])))
    if len({V.shape for V in Vs}) != 1:
        raise SystemExit("le original dei soggetti del template hanno numeri di vertici diversi")
    mean = normalize_vertices(np.mean(np.stack(Vs), axis=0))
    idx = np.sort(np.random.default_rng(0).choice(len(mean), irt.N_POINTS, replace=False))
    print(f"[comp-tpl] template: media di {len(chosen)} original non valutate (pool {len(pool)}, non valutati "
          f"{len(others)}), {len(mean)} vertici, {irt.N_POINTS} punti; primi {chosen[:3]}", flush=True)
    return mean[idx], chosen, len(mean)


def main() -> None:
    args = parse_args()
    subjects = select_subjects(args.view_dir, args.eval_seed)
    T, chosen, n_vertices = build_template(args.view_dir, subjects)
    if set(chosen) & set(subjects):
        raise SystemExit("template costruito con soggetti valutati")
    keys = [(s, t) for s in subjects for t in TOPOLOGIES]
    if args.limit:
        keys = keys[: args.limit]
    tasks = [(str(args.view_dir / f"{s}_GTready_{t}.npz"), irt.mesh_seed(s, t)) for s, t in keys]
    t0 = time.time()
    # spawn come ir_template: open3d non e' fork-safe.
    with mp.get_context("spawn").Pool(args.workers, initializer=irt._init, initargs=(T,)) as pool:
        res = pool.map(irt._enroll_one, tasks, chunksize=4)
    failed = [(k, e) for k, (_, _, e) in zip(keys, res) if e]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, R=np.stack([r[0] for r in res]), subjects=np.asarray([s for s, _ in keys]),
             topologies=np.asarray([t for _, t in keys]), seconds=np.asarray([r[1] for r in res]),
             failed=np.asarray([f"{k[0]}|{k[1]}|{e}" for k, e in failed]), seeds=np.asarray([t[1] for t in tasks]),
             T=T, template_subjects=np.asarray(chosen), n_template_vertices=n_vertices)
    print(f"[comp-tpl] {len(keys)} mesh iscritte in {time.time() - t0:.0f}s, mediana "
          f"{np.median([r[1] for r in res]):.2f}s per mesh nel worker, fallite {len(failed)}"
          + (f" (p.es. {failed[:2]})" if failed else ""), flush=True)
    print(f"[comp-tpl] scritto {args.out}", flush=True)


if __name__ == "__main__":
    main()
