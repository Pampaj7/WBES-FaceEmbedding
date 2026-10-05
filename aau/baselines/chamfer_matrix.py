#!/usr/bin/env python3
"""Matrici 100x100 di Chamfer raw sui soggetti held-out.

E' la baseline di riferimento: serve a validare tutta la pipeline riproducendo lo
0.729 di Chamfer original->original della Tabella 2 del paper.  Attenzione, perche' nel
repo di "raw Chamfer" ce ne sono **due**, e non sono una trasformazione monotona l'una
dell'altra (media di distanze contro media di distanze al quadrato), quindi danno due
Spearman diversi:

``--variant facebench`` (default, metrica ``chamfer``)
    ``faceBench/latentVSpipeline``: vertici normalizzati maxabs, sottocampionati a 4096
    punti, ``0.5 * (mean d(X->Y) + mean d(Y->X))`` con distanze euclidee.  E' quella che
    ha prodotto i numeri della Tabella 2 (`facebench_original_original_100subj_norm`) e
    la diagonale della Tabella 1.  Gira su CPU (cKDTree).

``--variant eval_utils`` (metrica ``chamfer_sq``)
    ``robustness/eval_utils.symmetric_chamfer_same_shape_batch``: tutti i vertici,
    ``mean d^2(X->Y) + mean d^2(Y->X)``.  E' quella dell'eval del repo e delle celle
    fuori diagonale della Tabella 1 (`table1_pairlevel_exact/*/pair_metrics.csv`).
    Gira su GPU.

In entrambi i casi la normalizzazione dei vertici e' la stessa di ``dataset_gtready``
(centratura sulla media dei vertici, divisione per il max valore assoluto).

  aau/run_baselines.sh aau/baselines/chamfer_matrix.py --variant facebench --workers 24
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

VARIANT_METRIC = {"facebench": "chamfer", "eval_utils": "chamfer_sq"}

# Vertici normalizzati, caricati nel processo padre e condivisi con i worker via fork.
_VERTS: dict[tuple[str, str], np.ndarray] = {}
_MAX_SAMPLE_POINTS = 4096


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", type=Path, default=common.OUT_ROOT)
    p.add_argument("--variant", type=str, default="facebench", choices=sorted(VARIANT_METRIC))
    p.add_argument("--settings", type=str, default=",".join(common.SETTINGS))
    p.add_argument("--max-sample-points", type=int, default=4096,
                   help="solo --variant facebench; e' il default di run_facebench_remesh.py")
    p.add_argument("--workers", type=int, default=8, help="solo --variant facebench")
    p.add_argument("--chunk", type=int, default=128, help="coppie per task inviato ai worker")
    p.add_argument("--device", type=str, default="cuda", help="solo --variant eval_utils")
    p.add_argument("--batch-pairs", type=int, default=64, help="come --chamfer_batch_pairs dell'eval")
    p.add_argument("--subject-set", type=str, default="heldout", choices=common.SUBJECT_SETS,
                   help="heldout = split del repo; facebench_first100 = i soggetti della Tabella 2")
    p.add_argument("--max-subjects", type=int, default=0, help="0 = tutti; >0 per un test rapido")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def normalized_verts(subject: str, topology: str) -> np.ndarray:
    """Vertici con la stessa normalizzazione geometrica di dataset_gtready."""
    return common.maxabs_normalize(common.load_verts_faces(subject, topology)[0])


# ------------------------------------------------------------------ variante facebench

def sample_pts(V: np.ndarray, max_pts: int, seed: int = 0) -> np.ndarray:
    """Copia dichiarata di ``run_facebench_remesh.sample_pts``.

    Non e' importabile: ``run_facebench_remesh`` fa ``import facebench``, che a sua volta
    importa ``open3d`` per l'ICP, assente dal container e inutile qui.  La funzione e'
    riportata identica; l'altra meta' del protocollo (``symmetric_chamfer``) viene invece
    importata davvero, da ``fg_metrics``.
    """
    if max_pts <= 0 or len(V) <= max_pts:
        return V
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(V), size=max_pts, replace=False)
    return V[np.sort(idx)]


def _run_chunk(task):
    from fg_metrics import symmetric_chamfer

    topology_a, topology_b, subjects_a, subjects_b, seeds = task
    out = []
    for subject_a, subject_b, seed in zip(subjects_a, subjects_b, seeds):
        # Stessa regola di run_facebench_remesh: X con il seme della coppia, Y con seme+1.
        Xs = sample_pts(_VERTS[(subject_a, topology_a)], _MAX_SAMPLE_POINTS, seed)
        Ys = sample_pts(_VERTS[(subject_b, topology_b)], _MAX_SAMPLE_POINTS, seed + 1)
        out.append(symmetric_chamfer(Xs, Ys))
    return out


def run_facebench(args, subjects, topologies, topology_pairs) -> None:
    import multiprocessing as mp

    # fg_metrics.symmetric_chamfer e' la stessa 0.5*(mean d_xy + mean d_yx) che
    # run_facebench_remesh calcola via facebench, ma con solo numpy+scipy dentro.
    # L'import qui nel padre serve a fallire subito se qualcosa manca.
    sys.path.insert(0, str(common.REPO_ROOT / "faceBench" / "latentVSpipeline"))
    from fg_metrics import symmetric_chamfer  # noqa: F401

    global _MAX_SAMPLE_POINTS
    _MAX_SAMPLE_POINTS = args.max_sample_points

    n = len(subjects)
    pair_i, pair_j = common.subject_pair_indices(n)
    names = np.asarray(subjects)

    t0 = time.time()
    for topology in topologies:
        for subject in subjects:
            _VERTS[(subject, topology)] = normalized_verts(subject, topology)
    total_mb = sum(v.nbytes for v in _VERTS.values()) / (1024.0**2)
    print(f"[chamfer] {len(_VERTS)} mesh caricate in {time.time() - t0:.0f}s ({total_mb:.0f} MB, "
          f"condivise coi worker via fork)", flush=True)

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=args.workers) as pool:
        for topology_a, topology_b in topology_pairs:
            out_path = common.matrix_path("chamfer", topology_a, topology_b, args.out_root)
            if out_path.exists() and not args.overwrite:
                print(f"[chamfer] {topology_a}->{topology_b}: gia' presente, salto", flush=True)
                continue
            t1 = time.time()
            chunks = [
                (topology_a, topology_b,
                 names[pair_i[start:start + args.chunk]].tolist(),
                 names[pair_j[start:start + args.chunk]].tolist(),
                 list(range(start, min(start + args.chunk, len(pair_i)))))
                for start in range(0, len(pair_i), args.chunk)
            ]
            values = np.concatenate([np.asarray(v, dtype=np.float64)
                                     for v in pool.map(_run_chunk, chunks)])
            D = common.empty_matrix(n)
            D[pair_i, pair_j] = values
            common.save_matrix(out_path, D, subjects, "chamfer", topology_a, topology_b,
                               max_sample_points=args.max_sample_points, variant="facebench")
            print(f"[chamfer] {topology_a}->{topology_b}: {len(values)} coppie in "
                  f"{time.time() - t1:.0f}s -> {out_path}", flush=True)


# ----------------------------------------------------------------- variante eval_utils

def run_eval_utils(args, subjects, topologies, topology_pairs) -> None:
    import torch

    sys.path.insert(0, str(common.REPO_ROOT / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic"))
    from robustness.eval_utils import compute_pairwise_chamfer_values

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    n = len(subjects)
    pair_i, pair_j = common.subject_pair_indices(n)

    t0 = time.time()
    verts = {
        topology: [torch.as_tensor(normalized_verts(s, topology), dtype=torch.float32, device=device)
                   for s in subjects]
        for topology in topologies
    }
    total_mb = sum(v.numel() * v.element_size() for vs in verts.values() for v in vs) / (1024.0**2)
    print(f"[chamfer] device={device} {len(topologies) * n} mesh caricate in "
          f"{time.time() - t0:.0f}s ({total_mb:.0f} MB)", flush=True)

    for topology_a, topology_b in topology_pairs:
        out_path = common.matrix_path("chamfer_sq", topology_a, topology_b, args.out_root)
        if out_path.exists() and not args.overwrite:
            print(f"[chamfer] {topology_a}->{topology_b}: gia' presente, salto", flush=True)
            continue
        # Lista piatta: i primi n indici sono la topologia A, i successivi n la B.
        t1 = time.time()
        values = compute_pairwise_chamfer_values(
            vertex_sets=verts[topology_a] + verts[topology_b],
            pair_i=pair_i,
            pair_j=pair_j + n,
            batch_pairs=args.batch_pairs,
            progress_desc=f"chamfer_sq {topology_a}->{topology_b}",
            show_progress=True,
        )
        D = common.empty_matrix(n)
        D[pair_i, pair_j] = values
        common.save_matrix(out_path, D, subjects, "chamfer_sq", topology_a, topology_b,
                           variant="eval_utils")
        print(f"[chamfer] {topology_a}->{topology_b}: {len(values)} coppie in "
              f"{time.time() - t1:.0f}s -> {out_path}", flush=True)


def main() -> None:
    args = parse_args()
    subjects = common.subject_set(args.subject_set)
    if args.max_subjects > 0:
        subjects = subjects[: args.max_subjects]

    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    topology_pairs = common.all_topology_pairs(settings)
    topologies = sorted({t for pair in topology_pairs for t in pair})
    print(f"[chamfer] variante={args.variant} ({VARIANT_METRIC[args.variant]}) "
          f"soggetti={len(subjects)} coppie/topologia={len(common.subject_pair_indices(len(subjects))[0])} "
          f"topologie={topologies} coppie-topologia={len(topology_pairs)}", flush=True)

    t0 = time.time()
    if args.variant == "facebench":
        run_facebench(args, subjects, topologies, topology_pairs)
    else:
        run_eval_utils(args, subjects, topologies, topology_pairs)
    print(f"[chamfer] fine in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
