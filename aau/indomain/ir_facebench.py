#!/usr/bin/env python3
"""Baseline faceBench (Chamfer, ICP rigido + Chamfer, NICP P2P/P2Tri) su TUTTE le coppie del riconoscimento in dominio.

    aau/outlineB/run_o3d.sh aau/indomain/ir_facebench.py --out-root aau/runs/indomain_recog/facebench \\
        --stage-dir /tmp/ir_fb --workers 64 --shard 0/6
    (ir_facebench.sbatch)

Retrieval e verifica vogliono la distanza fra ogni coppia di mesh con etichette diverse (topologie,
o espressioni per ``rexpr``), stesso soggetto compreso. ``alignment_matrix.py`` riempie solo i<j
per coppia ordinata di topologie e ``zs_bl_same.py`` aggiunge la diagonale; qui, per ogni coppia
NON ordinata di etichette (a, b) con a prima di b nell'ordine di ``sets.json``, si calcola la
matrice piena N x N: riga i = soggetto i in a (X), colonna j = soggetto j in b (Y). Ogni coppia
non ordinata di mesh con etichette diverse esce cosi' una volta sola; NICP e' asimmetrico e la sua
orientazione e' quella delle etichette, non query -> galleria (come in zs3dmm).

Pipeline importata, non riscritta: ``alignment_matrix._run_chunk`` -> ``run_geometry_pipeline`` di
faceBench, stadi raw/rigid/nicp, 4096 punti; seme = indice della coppia nella sua matrice (i*N + j).

Unita' di lavoro = (insieme, coppia di etichette): 15 per insieme, 45 in tutto; ``--shard k/n``
prende le unita' k, k+n, ... Riprendibile per unita'. Prima di partire la geometria (V/F, senza
operatori) delle mesh che servono va copiata in ``--stage-dir`` (/tmp del nodo): ~200 MB, e la
pipeline poi legge da RAM invece che da CephFS.

Output: ``<out-root>/<insieme>/<metrica>/<a>__<b>.npz`` con ``values`` (N, N), ``subjects``,
``n_failed``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
sys.path.insert(0, str(AAU_DIR / "baselines"))
sys.path.insert(0, str(AAU_DIR / "outlineB"))

import alignment_matrix as am  # noqa: E402

SETS_JSON = AAU_DIR / "runs" / "indomain_recog" / "sets.json"
FB_SETS = ("bfm", "ict", "rexpr")   # bfm19 e' un sottoinsieme di bfm: stesse matrici


def units(sets: dict) -> list[tuple[str, str, str]]:
    out = []
    for name in FB_SETS:
        labels = sets[name]["labels"]
        out += [(name, a, b) for i, a in enumerate(labels) for b in labels[i + 1:]]
    return out


def unit_paths(out_root: Path, name: str, a: str, b: str) -> dict[str, Path]:
    return {m: out_root / name / m / f"{a}__{b}.npz" for m in am.PIPELINE_METRICS.values()}


def stage(view_dir: Path, subjects, labels, dest: Path) -> dict[tuple[str, str], str]:
    """Copia V/F di ogni mesh in ``dest``; chiave (soggetto, etichetta) -> percorso locale."""
    dest.mkdir(parents=True, exist_ok=True)
    paths = {}
    for s in subjects:
        for t in labels:
            out = dest / f"{s}_GTready_{t}.npz"
            if not out.exists():
                with np.load(view_dir / out.name) as d:
                    V, F = (d["V"], d["F"]) if "V" in d else (d["verts"], d["faces"])
                    np.savez(out, V=V, F=F)
            paths[(s, t)] = str(out)
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--stage-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--max-sample-points", type=int, default=4096)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--max-subjects", type=int, default=0, help="0 = tutti; >0 per un test rapido")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    sets = json.loads(SETS_JSON.read_text())["sets"]
    k, n_shards = (int(x) for x in a.shard.split("/"))
    mine = units(sets)[k::n_shards]
    todo = [u for u in mine if a.overwrite or not all(q.exists() for q in unit_paths(a.out_root, *u).values())]
    print(f"[ir-fb] shard {a.shard}: {len(mine)} unita', {len(todo)} da fare: {todo}", flush=True)
    if not todo:
        return

    t0 = time.time()
    paths, subjects = {}, {}
    for name in sorted({u[0] for u in todo}):
        subj = sets[name]["subjects"][: a.max_subjects or None]
        subjects[name] = subj
        paths[name] = stage(Path(sets[name]["view_dir"]), subj, sets[name]["labels"], a.stage_dir / name)
    print(f"[ir-fb] geometria copiata in {a.stage_dir} in {time.time() - t0:.0f}s", flush=True)

    import multiprocessing as mp

    # spawn come alignment_matrix: open3d non e' fork-safe.
    with mp.get_context("spawn").Pool(processes=a.workers) as pool:
        for name, la, lb in todo:
            t1 = time.time()
            subj, P = subjects[name], paths[name]
            n = len(subj)
            ii, jj = np.divmod(np.arange(n * n), n)
            chunks = [([P[(subj[i], la)] for i in ii[s:s + a.chunk]], [P[(subj[j], lb)] for j in jj[s:s + a.chunk]],
                       list(range(s, min(s + a.chunk, n * n))), a.max_sample_points)
                      for s in range(0, n * n, a.chunk)]
            results = [r for block in pool.map(am._run_chunk, chunks) for r in block]
            values = np.asarray([r[0] for r in results], dtype=np.float64).reshape(n, n, -1)
            nicp_seconds = np.asarray([r[1] for r in results], dtype=np.float64)
            failed = [(q, r[3]) for q, r in enumerate(results) if r[2] != "ok"]
            for col, (metric, out) in enumerate(unit_paths(a.out_root, name, la, lb).items()):
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_name(out.stem + ".tmp.npz")
                np.savez_compressed(tmp, values=values[:, :, col], subjects=np.asarray(subj, dtype="U16"),
                                    metric=metric, label_a=la, label_b=lb, n_failed=len(failed),
                                    max_sample_points=a.max_sample_points,
                                    pipeline="run_facebench_remesh.run_geometry_pipeline")
                os.replace(tmp, out)
            print(f"[ir-fb] {name} {la}-{lb}: {len(results)} coppie in {time.time() - t1:.0f}s, "
                  f"NICP {np.nanmean(nicp_seconds):.2f}s/coppia, fallite {len(failed)}"
                  + (f" (p.es. {failed[:2]})" if failed else ""), flush=True)
    print(f"[ir-fb] fine in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
