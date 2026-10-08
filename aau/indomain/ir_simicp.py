#!/usr/bin/env python3
"""ICP di SIMILARITA' al posto dell'ICP rigido di faceBench (revisione 1 del protocollo, punto A).

Il problema: le mesh sono ad area unitaria, il crop ha meno area e quindi esce ingrandito (~8%)
rispetto alle altre topologie dello stesso soggetto; faceBench normalizza maxabs per mesh
(``mesh_npz_utils.normalize_vertices``), prealinea col bbox (``icp.prealign_by_bbox``) e chiama
``registration_icp`` con ``TransformationEstimationPointToPoint()`` senza scala
(``facebench/rigid_aligners/icp.py:75-80``), cosi' l'errore di scala resta dentro la distanza.

Qui la stessa pipeline di ``run_facebench_remesh.run_geometry_pipeline`` -- ``load_verts``,
``sample_pts`` coi semi ``seed`` e ``seed + 1``, ``prealign_by_bbox``, soglia 1000 -- con UNA
differenza: ``TransformationEstimationPointToPoint(with_scaling=True)``. Poi le misure di
faceBench, importate: ``chamfer_correspondence`` + ``p2p_distance`` (come ``rigid_p2p``) e
``nonrigid_icp_align`` + ``p2tri_distance`` (come ``nicp_p2tri``).

``_run_chunk`` ha la firma di ``alignment_matrix._run_chunk``: ``ir_facebench.py --variant sim`` lo
usa al posto di quello.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FB_DIR = REPO_ROOT / "faceBench" / "latentVSpipeline"
sys.path.insert(0, str(FB_DIR))
sys.path.insert(0, str(FB_DIR.parent))

# colonna -> nome della metrica nelle matrici (come alignment_matrix.PIPELINE_METRICS)
# ``chamfer_sim``: la Chamfer grezza di faceBench rifatta qui con gli stessi punti; su bfm/ict deve
# coincidere con la ``chamfer`` della prima tornata (stessi semi), ed e' il controllo; su ict992 e' la
# riga Chamfer.
SIM_METRICS = {"sim_p2p": "sim_icp_chamfer", "sim_nicp_p2tri": "sim_nicp_p2tri", "raw_chamfer": "chamfer_sim"}


def similarity_icp(source: np.ndarray, target: np.ndarray, threshold: float = 1000.0) -> np.ndarray:
    """``icp.icp_align(prealign="bbox")`` con la stima di similarita' (rotazione, traslazione, scala)."""
    import open3d as o3d
    from facebench.rigid_aligners.icp import prealign_by_bbox

    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(prealign_by_bbox(source.copy(), target))
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(target)
    result = o3d.pipelines.registration.registration_icp(
        src, tgt, threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True))
    src.transform(result.transformation)
    return np.asarray(src.points)


def similarity_pipeline(path_a: str, path_b: str, max_pts: int, seed: int, nicp: bool = True) -> dict:
    """Le due righe nuove su una coppia; ``nicp=False`` solo ICP di similarita' + Chamfer."""
    import facebench as fb
    import run_facebench_remesh as rfr

    out = {"sim_p2p": np.nan, "sim_nicp_p2tri": np.nan, "raw_chamfer": np.nan, "status": "ok", "error": ""}
    try:
        Xs = rfr.sample_pts(rfr.load_verts(Path(path_a)), max_pts, seed)
        Ys = rfr.sample_pts(rfr.load_verts(Path(path_b)), max_pts, seed + 1)
        out["raw_chamfer"] = float(rfr.symmetric_chamfer(Xs, Ys))
        X_sim = similarity_icp(Xs, Ys)
        corr = fb.chamfer_correspondence(X_sim, Ys)
        out["sim_p2p"] = float(np.mean(fb.p2p_distance(X_sim, Ys, corr)))
        if nicp:
            t0 = time.time()
            X_nicp = fb.nonrigid_icp_align(X_sim, Ys)
            out["nicp_seconds"] = float(time.time() - t0)
            corr_n = fb.chamfer_correspondence(X_nicp, Ys)
            out["sim_nicp_p2tri"] = float(np.mean(fb.p2tri_distance(X_nicp, Ys, corr_n)))
    except Exception as exc:  # noqa: BLE001  (come run_geometry_pipeline: la coppia resta NaN)
        out["status"], out["error"] = "failed", f"{type(exc).__name__}: {exc}"
    return out


def _run_chunk(task):
    """Worker, stessa firma e stesso ritorno di ``alignment_matrix._run_chunk``."""
    paths_a, paths_b, seeds, max_pts = task
    res = []
    for path_a, path_b, seed in zip(paths_a, paths_b, seeds):
        m = similarity_pipeline(path_a, path_b, max_pts, seed)
        res.append(([float(m[c]) for c in SIM_METRICS], float(m.get("nicp_seconds", np.nan)), m["status"], m["error"]))
    return res
