#!/usr/bin/env python3
"""Le baseline geometriche su NoW: Chamfer grezza e ICP + Chamfer in mm.

    aau/run.sh aau/recon/now_geometric.py --workers 32

Le mesh sono le patch canoniche di ``now_prepare_meshes.py`` (frame del template T7 in mm,
ritaglio NoW, 5215 triangoli).  Due criteri, con le definizioni di ``ws3b_geometric.py``:

``chamfer_raw``
    Chamfer simmetrica fra le due patch, ognuna centrata sulla media e divisa per il proprio
    maxabs, 4096 punti per lato (variante facebench).  Nessun allineamento oltre il frame
    canonico, che ogni mesh si e' dato coi propri landmark.

``icp_chamfer_mm``
    Similarita' ICP (``fg_metrics.rigid_icp_align``, 30 iterazioni, 4096 punti) del
    riferimento sull'altra mesh, similarita' riletta con ``procrustes_transform`` e
    invertita, cosi' l'altra mesh finisce nel frame del riferimento; poi la Chamfer in mm,
    senza altra normalizzazione.  E' ``chamfer_icp_mm`` di WS3b.  Riferimento = la
    scansione nel modo gt, la ricostruzione di indice minore nel modo pairs.

``--mode gt``     una riga per ricostruzione contro ``scan_face`` del soggetto (csv nel repo).
``--mode pairs``  matrice (n, n) fra tutte le ricostruzioni di un metodo, per il protocollo
                  d'identita' di (d) (npz fuori dal repo: i nomi sono dati NoW).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import now_common as common  # noqa: E402
from ws3b_geometric import maxabs_normalize  # noqa: E402

sys.path.insert(0, str(common.REPO_ROOT / "faceBench" / "latentVSpipeline"))

METRICS = ("chamfer_raw", "icp_chamfer_mm")
ITEM_FIELDS = ("name", "subject", "challenge")

# Stato condiviso coi worker via fork.
_STATE: dict = {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", type=str, default="all", choices=("all", "gt", "pairs"))
    p.add_argument("--methods", type=str, default=",".join(common.METHODS))
    p.add_argument("--sample-points", type=int, default=4096)
    p.add_argument("--icp-points", type=int, default=4096)
    p.add_argument("--icp-iter", type=int, default=30)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_V(path: Path) -> np.ndarray:
    with np.load(path) as z:
        return np.asarray(z["V"], dtype=np.float64)


def distances(V_ref: np.ndarray, V_other: np.ndarray, seed: int) -> tuple[float, float]:
    """(chamfer_raw, icp_chamfer_mm) fra il riferimento e l'altra mesh, semi da ``seed``."""
    import fg_metrics as fg

    n, k = _STATE["sample_points"], _STATE["icp_points"]
    raw = fg.symmetric_chamfer(fg.sample_vertices(maxabs_normalize(V_other), n, seed=seed),
                               fg.sample_vertices(maxabs_normalize(V_ref), n, seed=seed + 1))
    # Come ws3b_geometric: il riferimento va sull'altra mesh, la similarita' si rilegge e si
    # inverte, e l'altra mesh torna nei mm del riferimento.
    Y = fg.sample_vertices(V_other, k, seed=seed + 1)
    ref_aligned = fg.rigid_icp_align(V_ref, Y, max_points=k, max_iter=_STATE["icp_iter"], seed=seed)
    scale, R, trans = fg.procrustes_transform(V_ref, ref_aligned, scaling=True)
    other_mm = (V_other - trans) @ R.T / max(scale, 1e-12)
    icp = fg.symmetric_chamfer(fg.sample_vertices(other_mm, n, seed=seed),
                               fg.sample_vertices(V_ref, n, seed=seed + 1))
    return float(raw), float(icp)


def _gt_chunk(tasks):
    return [distances(_STATE["scans"][subject], _STATE["verts"][name], index)
            for index, name, subject in tasks]


def _pairs_chunk(tasks):
    verts = _STATE["verts"]
    return [(i, j, *distances(verts[_STATE["names"][i]], verts[_STATE["names"][j]], index))
            for index, i, j in tasks]


def run_pool(fn, tasks, args):
    blocks = [tasks[s:s + args.chunk] for s in range(0, len(tasks), args.chunk)]
    with mp.get_context("fork").Pool(processes=args.workers) as pool:
        return [r for block in pool.map(fn, blocks) for r in block]


def main() -> None:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    items = common.load_items()
    _STATE.update(sample_points=args.sample_points, icp_points=args.icp_points, icp_iter=args.icp_iter)
    _STATE["scans"] = {s: load_V(common.scan_face_dir() / f"{s}.npz") for s in common.subjects_of(items)}
    params = dict(sample_points=args.sample_points, icp_points=args.icp_points, icp_iter=args.icp_iter,
                  variant="facebench", units="chamfer_raw adimensionale (maxabs), icp_chamfer_mm in mm")

    for method in methods:
        mine = [it for it in items if (common.recon_face_dir(method) / f"{it.name}.npz").is_file()]
        if len(mine) != len(items):
            print(f"[now-geom] ATTENZIONE {method}: {len(mine)}/{len(items)} patch", flush=True)
        _STATE["verts"] = {it.name: load_V(common.recon_face_dir(method) / f"{it.name}.npz") for it in mine}

        path = common.gt_csv_path("geometric", method)
        if args.mode in ("all", "gt") and (args.overwrite or not path.exists()):
            t0 = time.time()
            # Il seme e' l'indice dell'immagine nella lista ufficiale: non dipende dai worker.
            index = {it.name: k for k, it in enumerate(items)}
            res = run_pool(_gt_chunk, [(index[it.name], it.name, it.subject) for it in mine], args)
            rows = [{**{k: getattr(it, k) for k in ITEM_FIELDS}, "chamfer_raw": r[0], "icp_chamfer_mm": r[1]}
                    for it, r in zip(mine, res)]
            common.write_rows(path, ITEM_FIELDS + METRICS, rows, method=method, seconds=time.time() - t0, **params)
            print(f"[now-geom] {method} gt: {len(rows)} in {time.time() - t0:.0f}s, mediane chamfer_raw "
                  f"{np.median([r[0] for r in res]):.4f}, icp_chamfer_mm {np.median([r[1] for r in res]):.3f} mm",
                  flush=True)

        out = common.pair_matrix_path("geometric", method)
        if args.mode in ("all", "pairs") and (args.overwrite or not out.exists()):
            t0 = time.time()
            names = [it.name for it in mine]
            _STATE["names"] = names
            iu, ju = np.triu_indices(len(names), 1)
            res = run_pool(_pairs_chunk, [(k, int(i), int(j)) for k, (i, j) in enumerate(zip(iu, ju))], args)
            D = {m: np.zeros((len(names), len(names))) for m in METRICS}
            for i, j, raw, icp in res:
                D["chamfer_raw"][i, j] = D["chamfer_raw"][j, i] = raw
                D["icp_chamfer_mm"][i, j] = D["icp_chamfer_mm"][j, i] = icp
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, names=np.asarray(names), **D)
            print(f"[now-geom] {method} pairs: {len(res)} coppie in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
