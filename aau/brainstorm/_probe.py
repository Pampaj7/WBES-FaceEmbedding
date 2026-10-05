#!/usr/bin/env python3
"""Sonda per H3: dimensioni delle mesh, distanze fra topologie nel frame comune, tempi.

Serve solo a scegliere soglie e sharding di ``h3_spectral.py``; non produce risultati.

    AAU_NV="" srun -p cpu -c 4 --mem=16G aau/run.sh aau/brainstorm/_probe.py
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
import common  # noqa: E402

subjects = common.subject_set("heldout")
print("soggetti", len(subjects), subjects[:3])
for s in subjects[:3]:
    V, F = common.load_verts_faces(s, "original")
    print(s, "original", V.shape, F.shape, "F[0]", F[0], "V[0]", V[0])

s = subjects[0]
Vo, Fo = common.load_verts_faces(s, "original")
e = np.linalg.norm(Vo[Fo[:, 0]] - Vo[Fo[:, 1]], axis=1)
print("edge mediano original", np.median(e))
for t in common.TOPOLOGIES:
    V, F = common.load_verts_faces(s, t)
    tri = V[F]
    A = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
    Vc = V - V.mean(0)
    d, _ = cKDTree(V).query(Vo)
    print(f"{t:8s} nV={len(V):6d} nF={len(F):6d} A={A:.4g} maxabs={np.abs(Vc).max():.4g} "
          f"rms={np.sqrt((Vc ** 2).sum(1).mean()):.4g} NN(orig->t) med={np.median(d):.4g} "
          f"p90={np.quantile(d, .9):.4g} p99={np.quantile(d, .99):.4g} max={d.max():.4g}")

import potpourri3d as pp3d  # noqa: E402
import robust_laplacian  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as sla  # noqa: E402

for t in ("original", "up60k"):
    V, F = common.load_verts_faces(s, t)
    t0 = time.time()
    L = pp3d.cotan_laplacian(V, F, denom_eps=1e-10)
    m = pp3d.vertex_areas(V, F)
    ev = sla.eigsh((L + sp.identity(L.shape[0]) * 1e-8).tocsc(), k=65, M=sp.diags(m),
                   sigma=1e-8, return_eigenvectors=False)
    t1 = time.time()
    L2, M2 = robust_laplacian.mesh_laplacian(V, F.astype(np.int64))
    ev2 = sla.eigsh((L2 + sp.identity(L2.shape[0]) * 1e-8).tocsc(), k=65, M=M2,
                    sigma=1e-8, return_eigenvectors=False)
    t2 = time.time()
    print(f"{t}: cotan {t1 - t0:.1f}s robust {t2 - t1:.1f}s  lam1..3 cotan {np.sort(ev)[1:4]} "
          f"robust {np.sort(ev2)[1:4]}")
