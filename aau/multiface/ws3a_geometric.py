#!/usr/bin/env python3
"""Chamfer, rigid ICP + Chamfer, varifold e currents sulle coppie del protocollo WS3a.

Le quattro metriche stanno in un solo script perche' condividono il costo dominante, che non
e' il calcolo ma il caricamento delle mesh: 2840 npz per topologia, letti una volta nel
processo padre e condivisi coi worker via fork, esattamente come in
``aau/baselines/chamfer_matrix.py``.

Chi calcola cosa (nessuna formula riscritta qui):

``chamfer`` / ``rigid_icp``
    ``faceBench/latentVSpipeline/fg_metrics.compute_pair_metrics`` con
    ``stages=("raw", "rigid")``: restituisce in un colpo la Chamfer simmetrica grezza e
    quella dopo allineamento rigido ICP (30 iterazioni, Procrustes con scala).  I vertici
    sono normalizzati maxabs e sottocampionati a 4096 punti con ``fg_metrics.sample_vertices``,
    seme = indice della coppia per il lato A e seme+1 per il lato B: e' il protocollo
    ``--variant facebench`` gia' usato per la Tabella 2 estesa di WS1.

``varifold`` / ``currents``
    ``v2_work/phase0/measure_distances.{varifold,currents}_distance``, ``normalize="maxabs"``
    e sigma 0.5/0.2/0.1/0.05 come WS1, ma **senza** il sottocampionamento a 4000 triangoli
    del default di phase0.  Il default era un difetto: ``mesh_measure`` pesca 4000
    triangoli su ``len(F)`` con un seme diverso per mesh, quindi la stessa mesh misurata due
    volte da' due misure diverse e la distanza fra due sottocampionamenti della SUA tracked
    valeva 0.125, piu' della distanza fra due soggetti.  Qui ``--max-tris 12000`` copre per
    intero tutte le topologie tranne ``up`` (27385 triangoli), che viene ridotta con un seme
    FISSO e uguale per tutte le mesh: la misura torna una funzione deterministica della
    mesh e la self-distance e' esattamente 0 (verificato, vedi ``--self-check``).
    Varifold e currents sono lo stesso kernel su misure diverse: il varifold usa il modulo
    del prodotto scalare fra le normali, currents il prodotto con segno, quindi currents
    "vede" l'orientamento e il varifold no.
    La normalizzazione e' ``maxabs`` e non piu' ``area``, il default di phase0: dividere
    per ``sqrt(area totale)`` rende la misura cieca proprio alle perturbazioni che
    cambiano l'area senza cambiare l'ingombro, che e' il caso di ``noisy`` (su REMESH ha
    2.28x l'area della sua original, e le distanze si accorciano del 34%).  ``maxabs`` e'
    anche la normalizzazione di Chamfer, del latent e dei render, quindi tutte le righe
    della tabella guardano finalmente la stessa mesh.

``bbox_proxy``
    Riga di CONTROLLO, non una metrica di forma: distanza euclidea fra i due vettori di
    quattro numeri (centro del bounding box e diagonale del bounding box) presi sulla mesh
    **gia' normalizzata maxabs**, cioe' sulla stessa mesh che vedono le metriche vere.
    Serve a leggere tutte le altre righe: se il proxy separa i soggetti quanto una metrica
    vera, quella cella non sta misurando la forma.
    Nelle unita' GREZZE del file il proxy dava AUC 0.996 sul protocollo pulito, ma era un
    controllo su un altro esperimento: nessuna metrica della tabella vede le unita' grezze,
    perche' tutte normalizzano.  Il proxy sulle mesh normalizzate e' il controllo giusto --
    dice quanto resta di banale DOPO la normalizzazione -- e per questo la riga e' stata
    rifatta cosi'.

Nota sulle auto-correlazioni di varifold e currents: ``_distance`` calcola <X,X> e <Y,Y> e
li mette in cache dentro il dict della misura.  Con la cache vuota al fork ogni worker se
le ricalcolerebbe per conto suo, fino a 24 volte le stesse 8520 auto-correlazioni contro
le 32000 correlazioni incrociate utili.  Per questo c'e' uno stage 0 che le calcola una
volta in parallelo, le riporta nel padre e solo dopo apre la pool di calcolo.  La cache e'
per (kind, sigmas, block), quindi lo stage 0 gira una volta per ognuna delle due misure.

  aau/submit.sh multiface/ws3a_geometric.sbatch
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ws3a_common as common  # noqa: E402

sys.path.insert(0, str(common.REPO_ROOT / "faceBench" / "latentVSpipeline"))
sys.path.insert(0, str(common.REPO_ROOT / "v2_work" / "phase0"))

GEOMETRIC_METRICS = ("chamfer", "rigid_icp", "varifold", "currents", "bbox_proxy")
# Le due che passano da `mesh_measure` invece che dai vertici campionati.
MEASURE_METRICS = ("varifold", "currents")

# Stato condiviso coi worker via fork: riempito nel padre prima di aprire la pool.
_VERTS: dict[tuple[str, str], np.ndarray] = {}
_MEASURES: dict[tuple[str, str], dict] = {}
_BBOX: dict[tuple[str, str], np.ndarray] = {}
_ICP_POINTS = 4096
_ICP_ITER = 30
_MAX_SAMPLE_POINTS = 4096
_SIGMAS: tuple[float, ...] = ()
_BLOCK = 0


def parse_args() -> argparse.Namespace:
    from measure_distances import BLOCK, DEFAULT_SIGMAS

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", type=Path, default=common.OUT_ROOT)
    p.add_argument("--metrics", type=str, default=",".join(GEOMETRIC_METRICS))
    p.add_argument("--max-sample-points", type=int, default=4096,
                   help="punti per lato di chamfer/rigid_icp; default del protocollo facebench")
    p.add_argument("--icp-points", type=int, default=4096, help="punti di lavoro dell'ICP")
    p.add_argument("--icp-iter", type=int, default=30)
    p.add_argument("--max-tris", type=int, default=12000,
                   help="solo varifold/currents: sopra questo numero mesh_measure "
                        "sottocampiona. 12000 copre tutte le topologie tranne up (27385)")
    p.add_argument("--measure-seed", type=int, default=0,
                   help="seme del sottocampionamento di mesh_measure, UGUALE per tutte le "
                        "mesh: e' quello che rende la misura deterministica e d(X,X)=0")
    p.add_argument("--normalize", type=str, default="maxabs", choices=("area", "maxabs"),
                   help="solo varifold/currents; maxabs = la normalizzazione di chamfer e "
                        "del latent, area = il default di phase0 (vedi il docstring)")
    p.add_argument("--self-check", type=int, default=8,
                   help="mesh su cui verificare d(X,X)=0 prima di calcolare; 0 = salta")
    p.add_argument("--sigmas", type=str, default=",".join(str(s) for s in DEFAULT_SIGMAS))
    p.add_argument("--block", type=int, default=BLOCK)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk", type=int, default=64, help="coppie per task inviato ai worker")
    p.add_argument("--max-pairs-per-class", type=int, default=0,
                   help="0 = protocollo intero; 50 = le 200 coppie del test di velocita'")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_shared(topologies: dict[str, list[str]], args, need_verts: bool, need_measures: bool,
                need_bbox: bool = False) -> None:
    """Vertici normalizzati, misure geometriche e/o bbox per ogni (topologia, nome)."""
    from measure_distances import mesh_measure

    t0 = time.time()
    n_done = 0
    for topology, names in topologies.items():
        for name in names:
            with np.load(common.mesh_path(topology, name), allow_pickle=False) as data:
                V = np.asarray(data["V"], dtype=np.float64)
                F = np.asarray(data["F"], dtype=np.int64) if need_measures else None
            if need_verts:
                # Stessa normalizzazione geometrica di dataset_gtready e di chamfer_matrix.
                Vc = V - V.mean(axis=0, keepdims=True)
                scale = float(np.max(np.abs(Vc)))
                _VERTS[(topology, name)] = Vc / scale if scale > 1e-6 else Vc * 0.0
            if need_bbox:
                # Il proxy si calcola sulla mesh NORMALIZZATA, la stessa che vedono chamfer,
                # il latent e i render: e' il residuo banale dopo la normalizzazione, non
                # la posizione assoluta della testa nel frame tracciato (che nessuna delle
                # metriche della tabella riceve).
                Vp = V - V.mean(axis=0, keepdims=True)
                scale_p = float(np.max(np.abs(Vp)))
                Vp = Vp / scale_p if scale_p > 1e-6 else Vp * 0.0
                lo, hi = Vp.min(axis=0), Vp.max(axis=0)
                _BBOX[(topology, name)] = np.append((lo + hi) / 2.0, np.linalg.norm(hi - lo))
            if need_measures:
                # seed uguale per tutte le mesh: vedi il docstring del modulo.
                _MEASURES[(topology, name)] = mesh_measure(
                    (V, F), max_tris=args.max_tris, seed=args.measure_seed,
                    normalize=args.normalize,
                )
            n_done += 1
            if n_done % 1000 == 0:
                print(f"[ws3a-geom] caricate {n_done} mesh ({n_done / max(time.time() - t0, 1e-9):.0f}/s)",
                      flush=True)
    measure_bytes = sum(4 * (m["centroids"].numel() + m["normals"].numel() + m["areas"].numel())
                        for m in _MEASURES.values())
    total_mb = (sum(v.nbytes for v in _VERTS.values()) + measure_bytes) / (1024.0 ** 2)
    print(f"[ws3a-geom] {n_done} mesh caricate in {time.time() - t0:.0f}s (~{total_mb:.0f} MB, "
          f"condivise coi worker via fork)", flush=True)


def self_check(kinds, sigmas, args) -> None:
    """d(X, X) deve essere 0: e' cio' che il seme per mesh di phase0 rompeva.

    Il controllo si fa su una misura RICOSTRUITA da zero, non sulla copia gia' in
    ``_MEASURES``: passare due volte lo stesso dict darebbe 0 anche con un
    sottocampionamento casuale, perche' xx, yy e xy verrebbero dagli stessi triangoli.
    """
    from measure_distances import _distance, mesh_measure

    keys = sorted(_MEASURES)
    picked = keys[:: max(1, len(keys) // args.self_check)][: args.self_check]
    worst = 0.0
    for topology, name in picked:
        with np.load(common.mesh_path(topology, name), allow_pickle=False) as data:
            V = np.asarray(data["V"], dtype=np.float64)
            F = np.asarray(data["F"], dtype=np.int64)
        twin = mesh_measure((V, F), max_tris=args.max_tris, seed=args.measure_seed,
                            normalize=args.normalize)
        for kind in kinds:
            worst = max(worst, abs(_distance(_MEASURES[(topology, name)], twin,
                                             sigmas, kind, args.block)))
    print(f"[ws3a-geom] self-check {kinds} su {len(picked)} mesh: d(X,X) max = {worst:.3e}",
          flush=True)
    if worst > 1e-6:
        raise SystemExit(f"self-check fallito: d(X,X) = {worst:.3e}, la misura non e' "
                         "deterministica (max_tris troppo basso o seme per mesh)")


def _init_worker(max_sample_points: int, icp_points: int, icp_iter: int,
                 sigmas: tuple[float, ...], block: int) -> None:
    import torch

    torch.set_num_threads(1)  # il parallelismo e' sui processi, non dentro BLAS
    global _MAX_SAMPLE_POINTS, _ICP_POINTS, _ICP_ITER, _SIGMAS, _BLOCK
    _MAX_SAMPLE_POINTS, _ICP_POINTS, _ICP_ITER = max_sample_points, icp_points, icp_iter
    _SIGMAS, _BLOCK = sigmas, block


def _self_inner_chunk(task):
    """Stage 0: auto-correlazioni di una misura, una per mesh invece che una per worker."""
    from measure_distances import _self_inner

    kind, keys = task
    return [(key, _self_inner(_MEASURES[key], _SIGMAS, kind, _BLOCK)) for key in keys]


def _run_chunk(task):
    """Le distanze di un blocco di coppie, per una metrica e una coppia di topologie."""
    from fg_metrics import compute_pair_metrics, sample_vertices
    from measure_distances import currents_distance, varifold_distance

    measure_distance = {"varifold": varifold_distance, "currents": currents_distance}
    metric, topology_a, topology_b, items = task
    out = []
    for pair_index, name_a, name_b in items:
        if metric == "bbox_proxy":
            out.append(float(np.linalg.norm(_BBOX[(topology_a, name_a)]
                                            - _BBOX[(topology_b, name_b)])))
            continue
        if metric in measure_distance:
            out.append(measure_distance[metric](_MEASURES[(topology_a, name_a)],
                                                _MEASURES[(topology_b, name_b)],
                                                sigmas=_SIGMAS, block=_BLOCK))
            continue
        # Stessa regola di chamfer_matrix: X con il seme della coppia, Y con seme+1.
        X = sample_vertices(_VERTS[(topology_a, name_a)], _MAX_SAMPLE_POINTS, seed=pair_index)
        Y = sample_vertices(_VERTS[(topology_b, name_b)], _MAX_SAMPLE_POINTS, seed=pair_index + 1)
        result = compute_pair_metrics(
            X, Y, stages=("raw",) if metric == "chamfer" else ("rigid",),
            icp_points=_ICP_POINTS, icp_iter=_ICP_ITER, seed=pair_index,
        )
        out.append(result.raw_chamfer if metric == "chamfer" else result.rigid_chamfer)
    return out


def main() -> None:
    import multiprocessing as mp

    args = parse_args()
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = [m for m in metrics if m not in GEOMETRIC_METRICS]
    if unknown:
        raise SystemExit(f"metrica sconosciuta: {unknown} (attese {list(GEOMETRIC_METRICS)})")
    sigmas = tuple(float(s) for s in args.sigmas.split(",") if s.strip())

    records = common.load_pairs(max_pairs_per_class=args.max_pairs_per_class)
    todo = [
        (metric, topology_a, topology_b)
        for metric in metrics
        for topology_a, topology_b in common.TOPOLOGY_PAIRS
        if args.overwrite or not common.csv_path(metric, topology_a, topology_b, args.out_root).exists()
    ]
    print(f"[ws3a-geom] coppie={len(records)} metriche={metrics} "
          f"da calcolare={len(todo)}/{len(metrics) * len(common.TOPOLOGY_PAIRS)} "
          f"workers={args.workers} max_tris={args.max_tris} sigmas={sigmas}", flush=True)
    if not todo:
        print("[ws3a-geom] niente da fare")
        return

    need_verts = any(metric in ("chamfer", "rigid_icp") for metric, _, _ in todo)
    need_bbox = any(metric == "bbox_proxy" for metric, _, _ in todo)
    measure_kinds = [k for k in MEASURE_METRICS if any(metric == k for metric, _, _ in todo)]
    need_measures = bool(measure_kinds)
    topologies = common.topology_names(records, [(ta, tb) for _, ta, tb in todo])
    print(f"[ws3a-geom] mesh da caricare: "
          + ", ".join(f"{t}={len(n)}" for t, n in topologies.items()), flush=True)
    load_shared(topologies, args, need_verts=need_verts, need_measures=need_measures,
                need_bbox=need_bbox)
    if need_measures and args.self_check:
        self_check(measure_kinds, sigmas, args)

    t0 = time.time()
    ctx = mp.get_context("fork")
    init_args = (args.max_sample_points, args.icp_points, args.icp_iter, sigmas, args.block)

    if need_measures:
        keys = sorted(_MEASURES)
        with ctx.Pool(processes=args.workers, initializer=_init_worker, initargs=init_args) as pool:
            for kind in measure_kinds:
                t1 = time.time()
                blocks = [(kind, keys[start:start + 64]) for start in range(0, len(keys), 64)]
                for done in pool.map(_self_inner_chunk, blocks):
                    for key, value in done:
                        _MEASURES[key]["_cache"][(kind, sigmas, args.block)] = value
                print(f"[ws3a-geom] auto-correlazioni {kind} per {len(keys)} mesh in "
                      f"{time.time() - t1:.0f}s", flush=True)

    with ctx.Pool(processes=args.workers, initializer=_init_worker, initargs=init_args) as pool:
        for metric, topology_a, topology_b in todo:
            t1 = time.time()
            chunks = [
                (metric, topology_a, topology_b,
                 [(rec.pair_index, rec.name_a, rec.name_b) for rec in records[start:start + args.chunk]])
                for start in range(0, len(records), args.chunk)
            ]
            values = np.concatenate([np.asarray(v, dtype=np.float64) for v in pool.map(_run_chunk, chunks)])
            seconds = time.time() - t1
            out_path = common.csv_path(metric, topology_a, topology_b, args.out_root)
            common.write_distances(
                out_path, records, values, metric, topology_a, topology_b, seconds=seconds,
                max_sample_points=args.max_sample_points, icp_points=args.icp_points,
                icp_iter=args.icp_iter, max_tris=args.max_tris, normalize=args.normalize,
                measure_seed=args.measure_seed,
                sigmas=list(sigmas), block=args.block, variant="facebench",
            )
            print(f"[ws3a-geom] {metric} {topology_a}->{topology_b}: {len(values)} coppie in "
                  f"{seconds:.0f}s (finite={int(np.isfinite(values).sum())}) -> {out_path}", flush=True)

    print(f"[ws3a-geom] fine in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
