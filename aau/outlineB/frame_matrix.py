#!/usr/bin/env python3
"""Chamfer e D_GT sotto frame diversi, sui soggetti held-out BFM (OUTLINE_B §5, Tabella 3).

Stessa Chamfer della Tabella 2 (variante faceBench di ``aau/baselines/chamfer_matrix.py``:
4096 punti, seme = indice della coppia per X e seme + 1 per Y, ``fg_metrics.symmetric_chamfer``),
cambia solo il frame in cui la mesh intera e' messa PRIMA del campionamento:

    maxabs  media dei vertici, diviso il max |coord|         (quello della Tabella 2)
    area    centroide pesato per area, diviso sqrt(area)     (gt_frames.reframe)
    rms     centroide pesato per area, diviso raggio rms     (gt_frames.reframe)
    global  (V - c0) / s0, una sola similarita' per tutto il dataset (global_frame.fit:
            c0 = media dei centroidi, s0 = mediana dei raggi rms, sulle 400 identita' di
            TRAINING dello split seed 1234, topologia original)

Due frame ibridi spezzano il global in posizione e taglia, a parita' di punti campionati:

    pm_translation  (V - media dei vertici) / s0: traslazione per mesh, scala globale
    pm_scale        (V - c0) / max|V - media dei vertici|: centro globale, scala per mesh
                    (il divisore di maxabs); la scala e' attorno a c0, quindi sposta
                    anche la posizione relativa a c0

Con maxabs e global formano un 2x2 (centro per mesh / globale x scala per mesh / globale).
Accanto, il controllo ``bbox_global``: centro del bounding box e diagonale (4 numeri) nel
frame globale, distanza euclidea fra i due vettori; nessun campionamento.

Le funzioni di frame sono importate da ``v2_work/xdomain`` (``gt_frames.reframe``,
``global_frame.fit/apply``), non riscritte.  ``area`` e ``rms`` sono quelli di
``v2_work/pointnet/frames.py``: la massa per vertice e' l'area baricentrica, che
``gt_frames`` ha verificato coincidere con la massa di DiffusionNet.  Il campionamento
sceglie gli indici dei vertici e non dipende dal frame, quindi tutte le metriche vedono
gli stessi punti: ``chamfer_maxabs`` deve coincidere con ``chamfer`` di chamfer_matrix.py.

In piu', D_GT ricalcolato sulle mesh ``original`` dei 100 soggetti (media sui vertici della
distanza L2, corrispondenza densa del 3DMM) nei frame raw / maxabs / area / rms / global,
salvato come metrica ``gt_<frame>``: ``rank_from_matrix`` lo confronta con il D_GT del repo,
che e' in coordinate grezze (commit 0281a3f).  ``gt_raw`` deve dare Spearman 1.

Con ``--subject-set ict_heldout`` lo stesso sui 100 held-out ICT (seed 1234), frame globale
stimato sulle altre 400 identita' dello stesso pool, D_GT ICT in coordinate grezze
(``alignment_table.ict_raw_gt``), perche' quella di train_ready e' maxabs.  Le mesh NON sono
quelle di ``datasets/ICT/eval_view_heldout``: puntano a ``topo_withops``, gia' normalizzate
maxabs per mesh (c0 = 0, s0 = 0.3, e gt_raw su di esse da' 0.570 contro la D_GT grezza, job
1054947).  Si usa una vista a symlink con gli stessi nomi verso ``datasets/ICT/topo``, da
cui ``v2_work/genict/build_ict_gt_matrix.py`` ha calcolato la D_GT grezza (``ict_raw_view``).
``gt_raw`` deve dare Spearman 1 anche qui.

  aau/run.sh aau/outlineB/frame_matrix.py --out-root aau/runs/outlineB/bfm_heldout --workers 30
  aau/run.sh aau/outlineB/frame_matrix.py --subject-set ict_heldout \
      --out-root aau/runs/outlineB/ict_heldout --frames maxabs,global,pm_translation,pm_scale
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "baselines"))

import common  # noqa: E402

XDOMAIN_DIR = common.REPO_ROOT / "v2_work" / "xdomain"
FRAMES = ("maxabs", "area", "rms", "global", "pm_translation", "pm_scale")
# D_GT solo nei frame della Tabella 3: i due ibridi servono a spezzare la Chamfer.
GT_FRAMES = ("raw", "maxabs", "area", "rms", "global")
ICT_DIR = common.REPO_ROOT / "datasets" / "ICT"

# Vertici nel frame, caricati nel padre e condivisi coi worker via fork (come chamfer_matrix).
_VERTS: dict[tuple[str, str, str], np.ndarray] = {}
_MAX_SAMPLE_POINTS = 4096


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--subject-set", type=str, default="heldout", choices=("heldout", "ict_heldout"),
                   help="il frame globale si stima sul complemento di training dello split")
    p.add_argument("--frames", type=str, default=",".join(FRAMES))
    p.add_argument("--settings", type=str, default="original_to_original,all_cross_topology")
    p.add_argument("--max-sample-points", type=int, default=4096)
    p.add_argument("--split-seed", type=int, default=1234,
                   help="seed dello split train/held-out del v1 (training = i 400 restanti)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def training_subjects(heldout: list[str], seed: int) -> list[str]:
    """Le 400 identita' di training dello split del v1, con la funzione del trainer."""
    sys.path.insert(0, str(common.REPO_ROOT / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic"))
    from robustness.data_utils import rebuild_subject_split

    pool = sorted(p.stem.split("_GTready_")[0] for p in common.MESH_ROOT.glob("*_GTready_original.npz"))
    train, evaluated = rebuild_subject_split(pool, eval_fraction=0.2, seed=seed, max_subjects=500)
    # Il set held-out letto dai pair table e lo split rifatto qui devono essere lo stesso:
    # se non lo sono, il frame globale sarebbe stimato anche su soggetti valutati.
    if sorted(evaluated) != sorted(heldout):
        raise SystemExit(f"lo split seed {seed} non riproduce i 100 held-out dei pair table")
    assert not set(train) & set(heldout)
    return train


def ict_raw_view(out_dir: Path) -> Path:
    """Vista ``id1NNNN_GTready_<t>.npz`` -> ``datasets/ICT/topo/ictNNNN_GTready_<t>.npz``.

    Stessi file di ``eval_view_heldout`` (500 identita' x 6 topologie), ma le mesh grezze.
    """
    from alignment_table import ICT_ID_OFFSET

    view = out_dir / "raw_view"
    view.mkdir(parents=True, exist_ok=True)
    for entry in sorted((ICT_DIR / "eval_view_heldout").glob("id1*_GTready_*.npz")):
        sid, rest = entry.name.split("_GTready_")
        src = ICT_DIR / "topo" / f"ict{int(sid[2:]) - ICT_ID_OFFSET:04d}_GTready_{rest}"
        if not src.is_file():
            raise SystemExit(f"mesh ICT grezza assente: {src}")
        dst = view / entry.name
        if not dst.is_symlink():
            dst.symlink_to(src)
    return view


def framed(V: np.ndarray, F: np.ndarray, frame: str, global_fit: dict) -> np.ndarray:
    import gt_frames
    import global_frame

    if frame == "global":
        return global_frame.apply(V, global_fit["c0"], global_fit["s0"])
    if frame == "pm_translation":
        return global_frame.apply(V, V.mean(0), global_fit["s0"])
    if frame == "pm_scale":
        # Stesso divisore di gt_frames.reframe(..., "maxabs"), ma attorno a c0.
        return global_frame.apply(V, global_fit["c0"],
                                  max(float(np.abs(V - V.mean(0, keepdims=True)).max()), 1e-12))
    return gt_frames.reframe(V, F, frame)


def _run_chunk(task):
    from chamfer_matrix import sample_pts
    from fg_metrics import symmetric_chamfer

    frame, topology_a, topology_b, subjects_a, subjects_b, seeds = task
    out = []
    for subject_a, subject_b, seed in zip(subjects_a, subjects_b, seeds):
        Xs = sample_pts(_VERTS[(frame, subject_a, topology_a)], _MAX_SAMPLE_POINTS, seed)
        Ys = sample_pts(_VERTS[(frame, subject_b, topology_b)], _MAX_SAMPLE_POINTS, seed + 1)
        out.append(symmetric_chamfer(Xs, Ys))
    return out


def gt_matrices(args, subjects, global_fit, frames) -> None:
    """D_GT nei frame richiesti sulle mesh original, e il controllo che raw sia quello del repo."""
    from scipy.stats import spearmanr

    n = len(subjects)
    pair_i, pair_j = common.subject_pair_indices(n)
    G = common.load_gt_submatrix(subjects)[pair_i, pair_j]
    raw = [common.load_verts_faces(s, "original") for s in subjects]
    for frame in (f for f in GT_FRAMES if f == "raw" or f in frames):
        out_path = common.matrix_path(f"gt_{frame}", "original", "original", args.out_root)
        if out_path.exists() and not args.overwrite:
            print(f"[frame] gt_{frame}: gia' presente, salto", flush=True)
            continue
        Vs = [V if frame == "raw" else framed(V, F, frame, global_fit) for V, F in raw]
        values = np.asarray([np.linalg.norm(Vs[i] - Vs[j], axis=1).mean()
                             for i, j in zip(pair_i, pair_j)])
        D = common.empty_matrix(n)
        D[pair_i, pair_j] = values
        common.save_matrix(out_path, D, subjects, f"gt_{frame}", "original", "original",
                           frame=frame)
        ratio = values / G
        print(f"[frame] gt_{frame}: rho vs D_GT del repo = {spearmanr(G, values).statistic:.6f}, "
              f"rapporto max/min - 1 = {ratio.max() / ratio.min() - 1:.2e}", flush=True)


def bbox_matrices(args, subjects, global_fit, topology_pairs) -> None:
    """Controllo bbox nel frame globale: (centro del bbox, diagonale), distanza euclidea."""
    n = len(subjects)
    pair_i, pair_j = common.subject_pair_indices(n)
    feats = {}
    for topology in sorted({t for pair in topology_pairs for t in pair}):
        rows = []
        for s in subjects:
            V = framed(*common.load_verts_faces(s, topology), "global", global_fit)
            lo, hi = V.min(axis=0), V.max(axis=0)
            rows.append(np.append((lo + hi) / 2.0, np.linalg.norm(hi - lo)))
        feats[topology] = np.asarray(rows)
    for topology_a, topology_b in topology_pairs:
        out_path = common.matrix_path("bbox_global", topology_a, topology_b, args.out_root)
        if out_path.exists() and not args.overwrite:
            continue
        D = common.empty_matrix(n)
        D[pair_i, pair_j] = np.linalg.norm(feats[topology_a][pair_i] - feats[topology_b][pair_j], axis=1)
        common.save_matrix(out_path, D, subjects, "bbox_global", topology_a, topology_b,
                           frame="global")
    print(f"[frame] bbox_global: {len(topology_pairs)} coppie di topologie", flush=True)


def main() -> None:
    args = parse_args()
    import multiprocessing as mp

    sys.path.insert(0, str(XDOMAIN_DIR))
    sys.path.insert(0, str(common.REPO_ROOT / "faceBench" / "latentVSpipeline"))
    import global_frame

    global _MAX_SAMPLE_POINTS
    _MAX_SAMPLE_POINTS = args.max_sample_points

    if args.subject_set == "ict_heldout":
        # Come alignment_table: common legge MESH_ROOT e GT_MATRIX al momento della chiamata.
        sys.path.insert(0, str(THIS_DIR))
        from alignment_table import ict_raw_gt

        common.MESH_ROOT = ict_raw_view(args.out_root)
        common.GT_MATRIX = ict_raw_gt(args.out_root)
    print(f"[frame] mesh {common.MESH_ROOT}, D_GT {common.GT_MATRIX}", flush=True)

    subjects = common.subject_set(args.subject_set)
    train = training_subjects(subjects, args.split_seed)
    global_fit = global_frame.fit(common.MESH_ROOT, train)
    print(f"[frame] frame globale su {global_fit['n_train_meshes']} mesh di training: "
          f"c0={np.round(global_fit['c0'], 1).tolist()} s0={global_fit['s0']:.1f}", flush=True)

    frames = [f.strip() for f in args.frames.split(",") if f.strip()]
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    topology_pairs = common.all_topology_pairs(settings)
    topologies = sorted({t for pair in topology_pairs for t in pair})

    gt_matrices(args, subjects, global_fit, frames)
    bbox_matrices(args, subjects, global_fit, topology_pairs)

    # Solo i frame con almeno una matrice da scrivere: gli altri non vanno nemmeno caricati.
    frames = [f for f in frames if args.overwrite or not all(
        common.matrix_path(f"chamfer_{f}", a, b, args.out_root).exists() for a, b in topology_pairs)]
    t0 = time.time()
    for topology in topologies:
        for subject in subjects:
            V, F = common.load_verts_faces(subject, topology)
            for frame in frames:
                _VERTS[(frame, subject, topology)] = framed(V, F, frame, global_fit)
    print(f"[frame] {len(_VERTS)} mesh nei frame {frames} in {time.time() - t0:.0f}s", flush=True)

    n = len(subjects)
    pair_i, pair_j = common.subject_pair_indices(n)
    names = np.asarray(subjects)
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=args.workers) as pool:
        for frame in frames:
            metric = f"chamfer_{frame}"
            for topology_a, topology_b in topology_pairs:
                out_path = common.matrix_path(metric, topology_a, topology_b, args.out_root)
                if out_path.exists() and not args.overwrite:
                    continue
                t1 = time.time()
                chunks = [
                    (frame, topology_a, topology_b,
                     names[pair_i[start:start + args.chunk]].tolist(),
                     names[pair_j[start:start + args.chunk]].tolist(),
                     list(range(start, min(start + args.chunk, len(pair_i)))))
                    for start in range(0, len(pair_i), args.chunk)
                ]
                values = np.concatenate([np.asarray(v, dtype=np.float64)
                                         for v in pool.map(_run_chunk, chunks)])
                D = common.empty_matrix(n)
                D[pair_i, pair_j] = values
                common.save_matrix(out_path, D, subjects, metric, topology_a, topology_b,
                                   max_sample_points=args.max_sample_points, frame=frame,
                                   variant="facebench")
                print(f"[frame] {metric} {topology_a}->{topology_b}: {len(values)} coppie in "
                      f"{time.time() - t1:.0f}s", flush=True)
    print(f"[frame] fine in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
