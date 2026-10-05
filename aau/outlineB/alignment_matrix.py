#!/usr/bin/env python3
"""Matrici 100x100 di tab:alignment_effect con la pipeline faceBench vera, su qualunque set.

Le quattro righe geometriche della Tabella 2 del paper -- Chamfer, rigid ICP + Chamfer,
rigid ICP + NICP + P2P, rigid ICP + NICP + P2Tri -- escono da
``run_facebench_remesh.run_geometry_pipeline`` con ``stages=raw,rigid,nicp``, **importata e
non riscritta**: vertici normalizzati maxabs per mesh (``mesh_npz_utils``), 4096 punti
campionati con seme = indice della coppia (X) e seme + 1 (Y), ``icp_align`` con
``prealign_by_bbox``, ``nonrigid_icp_align`` sul risultato.  Cambia solo l'insieme di
soggetti (``--subject-set``, vedi ``aau/baselines/common.py``).

Il seme della coppia e' quello di ``run_facebench_remesh.main``: l'indice ``i`` della coppia
di soggetti dentro la sua coppia di topologie, nell'ordine i<j dei soggetti ordinati, cioe'
``np.triu_indices``.  Per questo ``chamfer`` qui deve coincidere con le matrici di
``aau/baselines/chamfer_matrix.py`` (stessi punti, stessa formula): e' il primo controllo.

Le colonne della pipeline diventano metriche nel formato di ``common`` (solo i<j):

    raw_chamfer -> chamfer        rigid_p2p  -> rigid_icp_chamfer
    nicp_p2p    -> nicp_p2p       nicp_p2tri -> nicp_p2tri

Una coppia che fallisce resta NaN (``status != ok`` in faceBench) e viene contata nel log e
nell'npz.  Riprendibile per coppia di topologie: salta quelle con tutte e quattro le matrici
su disco.  ``--shard k/n`` divide le coppie di topologie fra job indipendenti.

  aau/outlineB/run_o3d.sh aau/outlineB/alignment_matrix.py --subject-set heldout \
      --out-root aau/runs/outlineB/bfm_heldout --workers 32 --shard 0/3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "baselines"))

import common  # noqa: E402

FB_DIR = common.REPO_ROOT / "faceBench" / "latentVSpipeline"
# run_facebench_remesh mette da se' faceBench/ e la sua dir su sys.path, ma i worker spawn
# devono poterlo importare prima ancora di eseguirlo.
sys.path.insert(0, str(FB_DIR))

# colonna della pipeline -> nome della metrica nelle matrici
PIPELINE_METRICS = {
    "raw_chamfer": "chamfer",
    "rigid_p2p": "rigid_icp_chamfer",
    "nicp_p2p": "nicp_p2p",
    "nicp_p2tri": "nicp_p2tri",
}
STAGES = ["raw", "rigid", "nicp"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--subject-set", type=str, required=True, choices=common.SUBJECT_SETS)
    p.add_argument("--settings", type=str,
                   default="original_to_original,all_cross_topology",
                   help="default: original->original + le 30 coppie cross (crop compreso, "
                        "serve alla compressione)")
    p.add_argument("--max-sample-points", type=int, default=4096,
                   help="default di run_facebench_remesh.py")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk", type=int, default=16, help="coppie per task inviato ai worker")
    p.add_argument("--shard", type=str, default="0/1", help="k/n: questo job fa le coppie "
                   "di topologie k, k+n, k+2n, ...")
    p.add_argument("--max-subjects", type=int, default=0, help="0 = tutti; >0 per un test rapido")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _run_chunk(task):
    """Worker: la pipeline faceBench su un blocco di coppie, nell'ordine ricevuto."""
    import run_facebench_remesh as rfr

    paths_a, paths_b, seeds, max_pts = task
    out = []
    for path_a, path_b, seed in zip(paths_a, paths_b, seeds):
        metrics = rfr.run_geometry_pipeline(path_a, path_b, STAGES, max_pts, seed)
        out.append((
            [float(metrics.get(column, np.nan)) for column in PIPELINE_METRICS],
            float(metrics.get("nicp_seconds", np.nan)),
            metrics.get("status", "ok"),
            metrics.get("error", ""),
        ))
    return out


def output_paths(out_root: Path, topology_a: str, topology_b: str) -> dict[str, Path]:
    return {metric: common.matrix_path(metric, topology_a, topology_b, out_root)
            for metric in PIPELINE_METRICS.values()}


def main() -> None:
    args = parse_args()
    import multiprocessing as mp

    subjects = common.subject_set(args.subject_set)
    if args.max_subjects > 0:
        subjects = subjects[: args.max_subjects]
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    topology_pairs = common.all_topology_pairs(settings)
    shard_k, shard_n = (int(x) for x in args.shard.split("/"))
    mine = topology_pairs[shard_k::shard_n]

    n = len(subjects)
    pair_i, pair_j = common.subject_pair_indices(n)
    paths = {(s, t): str(common.mesh_path(s, t)) for s in subjects for (a, b) in mine for t in (a, b)}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"[align] mancano {len(missing)} mesh, p.es. {missing[:3]}")
    print(f"[align] soggetti={args.subject_set} ({n}, primi {subjects[:3]}) mesh_root={common.MESH_ROOT}\n"
          f"[align] coppie di topologie: {len(topology_pairs)} in tutto, {len(mine)} in questo "
          f"shard {args.shard}: {mine}\n"
          f"[align] coppie di soggetti per topologia={len(pair_i)} workers={args.workers} "
          f"max_sample_points={args.max_sample_points}", flush=True)

    # spawn come run_facebench_remesh: open3d non e' fork-safe.
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(processes=args.workers) as pool:
        for topology_a, topology_b in mine:
            out_paths = output_paths(args.out_root, topology_a, topology_b)
            if all(p.exists() for p in out_paths.values()) and not args.overwrite:
                print(f"[align] {topology_a}->{topology_b}: gia' presente, salto", flush=True)
                continue
            t1 = time.time()
            chunks = []
            for start in range(0, len(pair_i), args.chunk):
                stop = min(start + args.chunk, len(pair_i))
                chunks.append((
                    [paths[(subjects[i], topology_a)] for i in pair_i[start:stop]],
                    [paths[(subjects[j], topology_b)] for j in pair_j[start:stop]],
                    list(range(start, stop)),       # seme = indice della coppia, come in main()
                    args.max_sample_points,
                ))
            results = [r for block in pool.map(_run_chunk, chunks) for r in block]
            values = np.asarray([r[0] for r in results], dtype=np.float64)   # (n_pairs, 4)
            nicp_seconds = np.asarray([r[1] for r in results], dtype=np.float64)
            failed = [(k, r[3]) for k, r in enumerate(results) if r[2] != "ok"]
            for col, metric in enumerate(PIPELINE_METRICS.values()):
                D = common.empty_matrix(n)
                D[pair_i, pair_j] = values[:, col]
                # Scrittura atomica: un kill a meta' non lascia un npz che il resume
                # scambierebbe per finito.
                tmp = out_paths[metric].with_name(out_paths[metric].stem + ".tmp.npz")
                common.save_matrix(tmp, D, subjects, metric, topology_a, topology_b,
                                   max_sample_points=args.max_sample_points,
                                   pipeline="run_facebench_remesh.run_geometry_pipeline",
                                   n_failed=len(failed),
                                   nicp_seconds_mean=float(np.nanmean(nicp_seconds)))
                os.replace(tmp, out_paths[metric])
            print(f"[align] {topology_a}->{topology_b}: {len(results)} coppie in "
                  f"{time.time() - t1:.0f}s, NICP {np.nanmean(nicp_seconds):.2f}s/coppia, "
                  f"fallite {len(failed)}" + (f" (p.es. {failed[:2]})" if failed else ""),
                  flush=True)
    print(f"[align] fine in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
