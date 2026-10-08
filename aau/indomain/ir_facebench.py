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

``--variant sim`` (revisione 1 del protocollo, punto A): al posto di ``alignment_matrix._run_chunk``
il worker di ``ir_simicp.py`` (ICP di SIMILARITA'), metriche ``sim_icp_chamfer``, ``sim_nicp_p2tri`` e
``chamfer_sim`` (la Chamfer grezza rifatta, controllo), sugli insiemi bfm e ict e sulla galleria
grande ``ict992`` (punto D). Le unita' di ict992 sono rettangolari: righe = le 100 query del
protocollo nell'etichetta di query, colonne = tutti i 992 soggetti nell'etichetta di galleria
(X = query); file ``<q>__to__<g>.npz`` con ``values`` (100, 992), ``subjects_a`` (query) e
``subjects_b`` (galleria).
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
import ir_simicp  # noqa: E402

SETS_JSON = AAU_DIR / "runs" / "indomain_recog" / "sets.json"
# bfm19 e' un sottoinsieme di bfm: stesse matrici. variante -> (insiemi quadrati, worker, metriche)
VARIANTS = {"rigid": (("bfm", "ict", "rexpr"), am._run_chunk, tuple(am.PIPELINE_METRICS.values())),
            "sim": (("bfm", "ict"), ir_simicp._run_chunk, tuple(ir_simicp.SIM_METRICS.values()))}
# Blocchi della galleria grande (protocollo, punto D): (etichetta delle query, etichetta della galleria).
GALLERY_BLOCKS = (("noisy", "original"), ("crop", "original"), ("remesh", "original"))


def units(sets: dict, variant: str = "rigid") -> list[tuple[str, str, str, str]]:
    """(insieme, a, b, tipo): ``square`` = N x N sulle coppie non ordinate, ``gallery`` = query x 992."""
    out = []
    for name in VARIANTS[variant][0]:
        labels = sets[name]["labels"]
        out += [(name, a, b, "square") for i, a in enumerate(labels) for b in labels[i + 1:]]
    if variant == "sim":
        out += [("ict992", q, g, "gallery") for q, g in GALLERY_BLOCKS]
    return out


def unit_paths(out_root: Path, name: str, a: str, b: str, kind: str = "square", variant: str = "rigid") -> dict[str, Path]:
    fname = f"{a}__{b}.npz" if kind == "square" else f"{a}__to__{b}.npz"
    return {m: out_root / name / m / fname for m in VARIANTS[variant][2]}


def stage(view_dir: Path, keys, dest: Path) -> dict[tuple[str, str], str]:
    """Copia V/F delle mesh ``keys`` (soggetto, etichetta) in ``dest``; chiave -> percorso locale."""
    dest.mkdir(parents=True, exist_ok=True)
    paths = {}
    for s, t in keys:
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
    p.add_argument("--variant", choices=sorted(VARIANTS), default="rigid")
    p.add_argument("--only", default="", help="solo queste unita', p.es. ict992:remesh:original (ignora --shard)")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    sets = json.loads(SETS_JSON.read_text())["sets"]
    _, worker, _ = VARIANTS[a.variant]
    k, n_shards = (int(x) for x in a.shard.split("/"))
    mine = units(sets, a.variant)[k::n_shards]
    if a.only:
        wanted = {tuple(u.split(":")) for u in a.only.split(",")}
        mine = [u for u in units(sets, a.variant) if u[:3] in wanted]
    todo = [u for u in mine
            if a.overwrite or not all(q.exists() for q in unit_paths(a.out_root, *u, variant=a.variant).values())]
    print(f"[ir-fb] shard {a.shard}: {len(mine)} unita', {len(todo)} da fare: {todo}", flush=True)
    if not todo:
        return

    t0 = time.time()
    paths, subjects = {}, {}
    for name in sorted({u[0] for u in todo}):
        subj = sets[name]["subjects"][: a.max_subjects or None]
        queries = [q for q in sets[name].get("queries", subj) if q in subj][: a.max_subjects or None]
        subjects[name] = (subj, queries)
        mine_here = [u for u in todo if u[0] == name]
        keys = {(s, t) for _, la, lb, kind in mine_here
                for t, ss in ((la, subj if kind == "square" else queries), (lb, subj)) for s in ss}
        paths[name] = stage(Path(sets[name]["view_dir"]), sorted(keys), a.stage_dir / name)
    print(f"[ir-fb] geometria copiata in {a.stage_dir} in {time.time() - t0:.0f}s", flush=True)

    import multiprocessing as mp

    # spawn come alignment_matrix: open3d non e' fork-safe.
    with mp.get_context("spawn").Pool(processes=a.workers) as pool:
        for name, la, lb, kind in todo:
            t1 = time.time()
            subj, queries = subjects[name]
            P = paths[name]
            rows = subj if kind == "square" else queries
            nr, nc = len(rows), len(subj)
            ii, jj = np.divmod(np.arange(nr * nc), nc)
            chunks = [([P[(rows[i], la)] for i in ii[s:s + a.chunk]], [P[(subj[j], lb)] for j in jj[s:s + a.chunk]],
                       list(range(s, min(s + a.chunk, nr * nc))), a.max_sample_points)
                      for s in range(0, nr * nc, a.chunk)]
            results = [r for block in pool.map(worker, chunks) for r in block]
            values = np.asarray([r[0] for r in results], dtype=np.float64).reshape(nr, nc, -1)
            nicp_seconds = np.asarray([r[1] for r in results], dtype=np.float64)
            failed = [(q, r[3]) for q, r in enumerate(results) if r[2] != "ok"]
            ids = ({"subjects": np.asarray(subj, dtype="U16")} if kind == "square" else
                   {"subjects_a": np.asarray(rows, dtype="U16"), "subjects_b": np.asarray(subj, dtype="U16")})
            for col, (metric, out) in enumerate(unit_paths(a.out_root, name, la, lb, kind, a.variant).items()):
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_name(out.stem + ".tmp.npz")
                np.savez_compressed(tmp, values=values[:, :, col], **ids,
                                    metric=metric, label_a=la, label_b=lb, n_failed=len(failed),
                                    max_sample_points=a.max_sample_points, variant=a.variant,
                                    pipeline="run_facebench_remesh.run_geometry_pipeline")
                os.replace(tmp, out)
            print(f"[ir-fb] {name} {la}-{lb}: {len(results)} coppie in {time.time() - t1:.0f}s, "
                  f"NICP {np.nanmean(nicp_seconds):.2f}s/coppia, fallite {len(failed)}"
                  + (f" (p.es. {failed[:2]})" if failed else ""), flush=True)
    print(f"[ir-fb] fine in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
