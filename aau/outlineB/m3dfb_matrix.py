#!/usr/bin/env python3
"""Matrici 100x100 di M3DFB RLR + Chamfer (stimatore E9) per tab:alignment_effect.

E9 = allineamento rigido sui landmark (Procrustes sui 5 landmark di riferimento di M3DFB),
corrispondenza Chamfer, distanza P2P densa, nessun corrector: lo stimatore che in DN
(17 agosto) faceva 0.456 su 20 soggetti.  Gira attraverso ``v2_work/m3dfb/m3dfb_adapter.py``
(importato, con le due patch documentate in ``INVENTORY.md``) sul clone di
``external/M3DFB``; mesh normalizzate maxabs per mesh come ``run_m3dfb_pairs.py``.

Landmark (M3DFB non ha un predittore, sono obbligatori), con la stessa regola di
``run_m3dfb_pairs.load_mesh``:
  - BFM: ``original`` e ``noisy`` sono BFM p23470 nell'ordine di M3DFB, quindi i 51 iBUG
    sono esatti (``m3dfb_adapter.bfm_landmark_indices``);
  - ICT: ``original`` (e ``noisy``, se ha gli stessi 9409 vertici) sono i primi 9409 vertici
    della topologia ICT, quindi i 51 iBUG sono i Multi-PIE 68 del README di ICT-FaceKit
    senza i 17 della mandibola (tutti < 9409);
  - le altre topologie: vertice piu' vicino, in coordinate grezze, al landmark della
    ``original`` dello stesso soggetto.  E' un'informazione privilegiata che una
    valutazione cross-topologia vera non avrebbe: il numero e' un limite superiore.
Il log stampa, per ogni topologia, la distanza mediana fra landmark trasferito e landmark
della original, in unita' della diagonale del bbox: se le topologie non stessero nello
stesso frame grezzo si vedrebbe li'.

  aau/outlineB/run_o3d.sh aau/outlineB/m3dfb_matrix.py --subject-set heldout \
      --out-root aau/runs/outlineB/bfm_heldout --workers 30
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "baselines"))

import common  # noqa: E402

sys.path.insert(0, str(common.REPO_ROOT / "v2_work" / "m3dfb"))
METRIC = "m3dfb_rlr_chamfer"
ESTIMATOR = "E9"   # RLR + none + Chamfer + P2P + none, m3dfb_adapter.ESTIMATORS
TEMPLATE_TOPOLOGIES = ("original", "noisy")

# ICT-FaceKit README, "Facial Landmarks", Multi-PIE 68 (0-indexed); i primi 17 sono la
# mandibola, i 51 restanti sono gli iBUG-51 nell'ordine di M3DFB.
ICT_MULTIPIE68 = (
    1225, 1888, 1052, 367, 1719, 1722, 2199, 1447, 966, 3661, 4390, 3927, 3924, 2608, 3272,
    4088, 3443, 268, 493, 1914, 2044, 1401, 3615, 4240, 4114, 2734, 2509, 978, 4527, 4942,
    4857, 1140, 2075, 1147, 4269, 3360, 1507, 1542, 1537, 1528, 1518, 1511, 3742, 3751, 3756,
    3721, 3725, 3732, 5708, 5695, 2081, 0, 4275, 6200, 6213, 6346, 6461, 5518, 5957, 5841,
    5702, 5711, 5533, 6216, 6207, 6470, 5517, 5966,
)
ICT_FACE_N_VERTS = 9409   # v2_work/genict/ict_model.py

_MESH: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--subject-set", type=str, required=True, choices=common.SUBJECT_SETS)
    p.add_argument("--settings", type=str, default="original_to_original,nocrop_cross_topology")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def template(subject_set: str) -> tuple[np.ndarray, int]:
    """Indici dei 51 landmark e numero di vertici della topologia template del dataset."""
    import m3dfb_adapter as m3

    if subject_set == "ict_heldout":
        return np.asarray(ICT_MULTIPIE68[17:], dtype=np.int64), ICT_FACE_N_VERTS
    return m3.bfm_landmark_indices(), int(m3.bfm_mm()["Npoints"])


def load_meshes(subjects, topologies, subject_set) -> None:
    template_idx, template_n = template(subject_set)
    spread: dict[str, list[float]] = {}
    for subject in subjects:
        V0, _ = common.load_verts_faces(subject, "original")
        if len(V0) != template_n:
            raise SystemExit(f"{subject} original ha {len(V0)} vertici, attesi {template_n}")
        L0 = V0[template_idx]
        diag = float(np.linalg.norm(V0.max(0) - V0.min(0)))
        for topology in topologies:
            V, F = common.load_verts_faces(subject, topology)
            if topology in TEMPLATE_TOPOLOGIES and len(V) == template_n:
                idx = template_idx
            else:
                idx = cKDTree(V).query(L0, k=1)[1].astype(np.int64)
            spread.setdefault(topology, []).append(
                float(np.median(np.linalg.norm(V[idx] - L0, axis=1))) / diag)
            _MESH[(subject, topology)] = (common.maxabs_normalize(V), F, idx)
    for topology, values in spread.items():
        print(f"[m3dfb] {topology:<9} landmark vs original: mediana {np.median(values):.2e} "
              f"max {np.max(values):.2e} (frazione della diagonale del bbox)", flush=True)


def _run_chunk(task):
    import m3dfb_adapter as m3

    topology_a, topology_b, subjects_a, subjects_b = task
    out = []
    for subject_a, subject_b in zip(subjects_a, subjects_b):
        VA, FA, ia = _MESH[(subject_a, topology_a)]
        VB, FB, ib = _MESH[(subject_b, topology_b)]
        try:
            out.append(m3.pair_distance(ESTIMATOR, VA, FA, VB, FB, lmks_a=VA[ia], lmks_b=VB[ib],
                                        lmk_indices_a=ia))
        except Exception as exc:   # come run_facebench_remesh: la coppia resta NaN e si conta
            print(f"[m3dfb] {subject_a}/{topology_a} -> {subject_b}/{topology_b}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            out.append(float("nan"))
    return out


def main() -> None:
    args = parse_args()
    import multiprocessing as mp

    subjects = common.subject_set(args.subject_set)
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    topology_pairs = common.all_topology_pairs(settings)
    topologies = sorted({t for pair in topology_pairs for t in pair})
    t0 = time.time()
    load_meshes(subjects, topologies, args.subject_set)
    print(f"[m3dfb] {args.subject_set}: {len(_MESH)} mesh in {time.time() - t0:.0f}s, "
          f"{len(topology_pairs)} coppie di topologie", flush=True)

    n = len(subjects)
    pair_i, pair_j = common.subject_pair_indices(n)
    names = np.asarray(subjects)
    # fork: niente open3d nel percorso di E9 (l'ICP di M3DFB e' solo in E1..E8).
    with mp.get_context("fork").Pool(processes=args.workers) as pool:
        for topology_a, topology_b in topology_pairs:
            out_path = common.matrix_path(METRIC, topology_a, topology_b, args.out_root)
            if out_path.exists() and not args.overwrite:
                continue
            t1 = time.time()
            chunks = [(topology_a, topology_b,
                       names[pair_i[start:start + args.chunk]].tolist(),
                       names[pair_j[start:start + args.chunk]].tolist())
                      for start in range(0, len(pair_i), args.chunk)]
            values = np.concatenate([np.asarray(v, dtype=np.float64)
                                     for v in pool.map(_run_chunk, chunks)])
            D = common.empty_matrix(n)
            D[pair_i, pair_j] = values
            tmp = out_path.with_name(out_path.stem + ".tmp.npz")
            common.save_matrix(tmp, D, subjects, METRIC, topology_a, topology_b,
                               estimator=ESTIMATOR, n_failed=int(np.sum(~np.isfinite(values))))
            os.replace(tmp, out_path)
            print(f"[m3dfb] {topology_a}->{topology_b}: {len(values)} coppie in "
                  f"{time.time() - t1:.0f}s, NaN {int(np.sum(~np.isfinite(values)))}", flush=True)
    print(f"[m3dfb] fine in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
