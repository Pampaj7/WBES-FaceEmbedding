#!/usr/bin/env python3
"""Chamfer faceBench (4096 punti, ``raw``) sulle mesh canonicalizzate: la baseline a parita' di allineamento.

    aau/outlineB/run_o3d.sh aau/evidence/e2_canon/chamfer_canon.py --mesh-dir /tmp/.../in \\
        --subjects-json /tmp/.../subjects.json --out-root <dir> --workers 32 \\
        [--control-view <vista>/npz --control-root <baselines esistenti>]

Stessa misura e stessi semi della riga ``Chamfer (faceBench, 4096 pt)`` (0.477 di rank-1 su HIFI3D):
``run_facebench_remesh.run_geometry_pipeline(..., stages=["raw"], 4096, seme)`` importata, non riscritta
(vertici normalizzati maxabs per mesh, 4096 vertici estratti col seme della coppia). Semi: coppie di
soggetti diversi = indice della coppia i<j dentro la coppia ordinata di topologie (``alignment_matrix.py``);
stesso soggetto = 1000000 + indice del soggetto (``zs_bl_same.py``). Uscite nel formato di
``aau/baselines/common.py`` (``matrices/chamfer/<ta>__to__<tb>.npz``, ``matrices_same/chamfer/...``), che
``zs_expr_summarize.facebench_distances`` legge.

``--control-view``: la stessa funzione sulle mesh NON canonicalizzate della vista, per due coppie di
topologie, contro le matrici esistenti: devono coincidere (controllo che la pipeline e' quella).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
FB_DIR = REPO_ROOT / "faceBench" / "latentVSpipeline"
sys.path.insert(0, str(FB_DIR))
sys.path.insert(0, str(REPO_ROOT / "aau" / "baselines"))

TOPOLOGIES = ("crop", "down8k", "noisy", "original", "remesh", "up60k")
SEED_SAME = 1_000_000
CONTROL_PAIRS = (("original", "down8k"), ("noisy", "remesh"))


def _chunk(task):
    import run_facebench_remesh as rfr
    return [float(rfr.run_geometry_pipeline(a, b, ["raw"], 4096, s)["raw_chamfer"]) for a, b, s in task]


def matrices(mesh_dir: Path, subjects: list[str], pairs, workers: int) -> dict:
    n = len(subjects)
    iu, ju = np.triu_indices(n, 1)
    path = lambda s, t: str(mesh_dir / f"{s}_GTready_{t}.npz")  # noqa: E731
    jobs, keys = [], []
    for ta, tb in pairs:
        for k, (i, j) in enumerate(zip(iu, ju)):
            jobs.append((path(subjects[i], ta), path(subjects[j], tb), k))
            keys.append(("x", ta, tb, i, j))
        for i, s in enumerate(subjects):
            jobs.append((path(s, ta), path(s, tb), SEED_SAME + i))
            keys.append(("s", ta, tb, i, i))
    blocks = [jobs[a:a + 64] for a in range(0, len(jobs), 64)]
    with mp.get_context("spawn").Pool(workers) as pool:
        vals = [v for b in pool.map(_chunk, blocks) for v in b]
    out = {}
    for (kind, ta, tb, i, j), v in zip(keys, vals):
        if kind == "x":
            out.setdefault(("x", ta, tb), np.full((n, n), np.nan))[i, j] = v
        else:
            out.setdefault(("s", ta, tb), np.full(n, np.nan))[i] = v
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mesh-dir", type=Path, required=True)
    p.add_argument("--subjects-json", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--control-view", type=Path, default=None)
    p.add_argument("--control-root", type=Path, default=None)
    a = p.parse_args()
    import common  # noqa: E402  (solo I/O delle matrici)

    subjects = json.loads(a.subjects_json.read_text())["subjects"]
    pairs = [(x, y) for x in TOPOLOGIES for y in TOPOLOGIES if x != y]
    t0 = time.time()
    res = matrices(a.mesh_dir, subjects, pairs, a.workers)
    for (kind, ta, tb), M in res.items():
        if kind == "x":
            common.save_matrix(common.matrix_path("chamfer", ta, tb, a.out_root), M, subjects, "chamfer", ta, tb,
                               max_sample_points=4096, pipeline="run_facebench_remesh.run_geometry_pipeline[raw]",
                               mesh_dir=str(a.mesh_dir))
        else:
            q = a.out_root / "matrices_same" / "chamfer" / f"{ta}__to__{tb}.npz"
            q.parent.mkdir(parents=True, exist_ok=True)
            np.savez(q, values=M, subjects=np.asarray(subjects, dtype="U16"))
    print(f"[chamfer-canon] {len(pairs)} coppie di topologie, {len(subjects)} soggetti in {time.time() - t0:.0f}s "
          f"-> {a.out_root}", flush=True)

    if a.control_view is not None:
        ctl = matrices(a.control_view, subjects, CONTROL_PAIRS, a.workers)
        rows = []
        for (kind, ta, tb), M in ctl.items():
            if kind == "x":
                ref, subj, _, _, _ = common.load_matrix(common.matrix_path("chamfer", ta, tb, a.control_root))
                assert subj == subjects
                iu = np.triu_indices(len(subjects), 1)
                d = np.abs(M[iu] - ref[iu])
            else:
                with np.load(a.control_root / "matrices_same" / "chamfer" / f"{ta}__to__{tb}.npz") as z:
                    d = np.abs(M - z["values"])
            rows.append({"kind": kind, "pair": f"{ta}->{tb}", "max_abs_diff": float(np.nanmax(d)), "n": int(d.size)})
        (a.out_root / "control_chamfer.json").write_text(json.dumps(rows, indent=1))
        print(f"[chamfer-canon] controllo contro le matrici esistenti: {rows}", flush=True)


if __name__ == "__main__":
    main()
