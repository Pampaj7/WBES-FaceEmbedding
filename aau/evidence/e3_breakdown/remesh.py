#!/usr/bin/env python3
"""E3b: rimesh uniforme al test, a densita' comune fissa, prima dell'embedding.

    aau/run.sh aau/evidence/e3_breakdown/remesh.py stage --view-dir <vista>/npz --seed 1234 \\
        --out-dir /tmp/.../in --records <csv> [--flip-faces]

Generatore (diverso da quelli delle topologie di test, che sono tutti decimazione quadrica ``igl.qslim``
di ``v2_work/genict/mesh_ops.decimate_to``: ``remesh`` = 2 passi di smoothing umbrella + 0.7x quadrica,
``down8k`` = quadrica, ``up60k`` = suddivisione a punto medio + quadrica): clustering di Voronoi uniforme
per area con triangolazione duale, nello stile di ACVD (Valette e Chassery 2004).
  1. la mesh si porta nel frame normalizzato: baricentro per area, raggio RMS per area = 1;
  2. numero di celle K = round(A / ((sqrt(3)/2) L^2)), cioe' l'area per vertice di una triangolazione
     equilatera di lato L; L e' UNA costante per tutte le mesh di tutti i domini (``L_EDGE``);
  3. mesh fine: suddivisione a punto medio (``subdivide_midpoint``, la geometria non cambia) finche' il
     99-esimo percentile dei lati e' sotto L/3 (al massimo 4M triangoli);
  4. Lloyd pesato per area sui vertici fini (assegnazione al centroide piu' vicino, 20 iterazioni,
     inizializzazione: K vertici fini estratti con probabilita' proporzionale all'area, seme dal nome del
     file); poi le celle definitive sono GEODESICHE: Dijkstra multi-sorgente sul grafo della mesh fine dai
     vertici piu' vicini ai centroidi (celle connesse per costruzione: con le celle euclidee la superficie
     rugosa di ``noisy`` dava celle frammentate e migliaia di spigoli non-manifold); centroide per area di
     ogni cella;
  5. triangolazione duale: ogni triangolo fine coi tre vertici in tre celle diverse da' il triangolo
     (cella_a, cella_b, cella_c), col verso del triangolo fine; duplicati tolti;
  6. vertici = centroidi delle celle proiettati sulla superficie originale (``igl.point_mesh_squared_distance``),
     poi riportati nelle unita' e nel frame della mesh d'ingresso.
Il rimesh cambia SOLO la discretizzazione: frame, unita' e verso restano quelli dell'ingresso.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "aau" / "zs3dmm"))

L_EDGE = 0.035       # lato bersaglio, frame normalizzato (raggio RMS per area = 1): ~9.3k vertici sulla original ICT
N_LLOYD = 20
MAX_FINE_FACES = 4_000_000


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as d:
        if "V" in d:
            return np.asarray(d["V"], np.float64), np.asarray(d["F"], np.int64)
        return np.asarray(d["verts"], np.float64), np.asarray(d["faces"], np.int64)


def face_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)


def area_frame(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, float]:
    """Baricentro e raggio RMS della superficie (integrali per area, via i baricentri dei triangoli)."""
    a = face_areas(V, F)
    G = V[F].mean(1)
    c = (G * a[:, None]).sum(0) / a.sum()
    # E|x - c|^2 su un triangolo = |G - c|^2 + (|A-G|^2 + |B-G|^2 + |C-G|^2) / 12
    spread = ((V[F] - G[:, None, :]) ** 2).sum((1, 2)) / 12.0
    r = float(np.sqrt(((((G - c) ** 2).sum(1) + spread) * a).sum() / a.sum()))
    return c, r


def unique_edges(F: np.ndarray, n_verts: int) -> tuple[np.ndarray, np.ndarray]:
    """Spigoli unici (u < v) e, per ogni semi-spigolo (e01 | e12 | e20 impilati), l'indice del suo spigolo."""
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1).astype(np.int64)
    key, inv = np.unique(E[:, 0] * n_verts + E[:, 1], return_inverse=True)
    return np.stack([key // n_verts, key % n_verts], 1), inv


def edge_lengths(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    E, _ = unique_edges(F, len(V))
    return np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)


def subdivide_midpoint(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """1 -> 4 a punto medio: i vertici nuovi stanno sui lati, la superficie e' la stessa."""
    n, m = len(V), len(F)
    E, inv = unique_edges(F, n)
    e01, e12, e20 = n + inv[:m], n + inv[m:2 * m], n + inv[2 * m:]
    a, b, c = F[:, 0], F[:, 1], F[:, 2]
    F2 = np.concatenate([np.stack([a, e01, e20], 1), np.stack([e01, b, e12], 1),
                         np.stack([e20, e12, c], 1), np.stack([e01, e12, e20], 1)])
    return np.concatenate([V, 0.5 * (V[E[:, 0]] + V[E[:, 1]])]), F2


def topology_stats(F: np.ndarray, n_verts: int) -> dict:
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(E, axis=0, return_counts=True)
    return {"nonmanifold_edges": int((cnt > 2).sum()), "boundary_edges": int((cnt == 1).sum()),
            "unreferenced": int(n_verts - len(np.unique(F)))}


def remesh(V: np.ndarray, F: np.ndarray, seed: int, L: float = L_EDGE) -> tuple[np.ndarray, np.ndarray, dict]:
    import igl
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    t0 = time.perf_counter()
    c, r = area_frame(V, F)
    Vn = (V - c) / r
    A = float(face_areas(Vn, F).sum())
    K = int(round(A / (np.sqrt(3) / 2 * L ** 2)))
    Vf, Ff = Vn, F
    n_sub = 0
    while np.percentile(edge_lengths(Vf, Ff), 99) > L / 3 and 4 * len(Ff) <= MAX_FINE_FACES:
        Vf, Ff = subdivide_midpoint(Vf, Ff)
        n_sub += 1
    # area per vertice fine (un terzo dei triangoli incidenti)
    fa = face_areas(Vf, Ff)
    w = np.bincount(Ff.ravel(), weights=np.repeat(fa / 3.0, 3), minlength=len(Vf))
    rng = np.random.default_rng(seed)
    C = Vf[rng.choice(len(Vf), size=K, replace=False, p=w / w.sum())]
    for _ in range(N_LLOYD):
        _, lab = cKDTree(C).query(Vf)
        sw = np.bincount(lab, weights=w, minlength=len(C))
        nz = sw > 0
        C = np.stack([np.bincount(lab, weights=w * Vf[:, k], minlength=len(C)) for k in range(3)], 1)
        C = C[nz] / sw[nz, None]
    # celle geodesiche: Dijkstra multi-sorgente dal vertice fine piu' vicino a ogni centroide
    _, seeds = cKDTree(Vf).query(C)
    seeds = np.unique(seeds)
    E, _ = unique_edges(Ff, len(Vf))
    el = np.linalg.norm(Vf[E[:, 0]] - Vf[E[:, 1]], axis=1)
    G = csr_matrix((np.concatenate([el, el]), (np.concatenate([E[:, 0], E[:, 1]]), np.concatenate([E[:, 1], E[:, 0]]))),
                   shape=(len(Vf), len(Vf)))
    dist, _, src = dijkstra(G, directed=True, indices=seeds, min_only=True, return_predecessors=True)
    reach = src >= 0                      # componenti senza seme (pezzi staccati minuscoli): fuori
    pos = np.full(len(Vf), -1, np.int64)
    pos[seeds] = np.arange(len(seeds))
    lab = np.where(reach, pos[np.where(reach, src, 0)], -1)
    sw = np.bincount(lab[reach], weights=w[reach], minlength=len(seeds))
    C = np.stack([np.bincount(lab[reach], weights=(w * Vf[:, k])[reach], minlength=len(seeds)) for k in range(3)], 1)
    C = C / np.maximum(sw, 1e-30)[:, None]
    T = lab[Ff]
    keep = (T >= 0).all(1) & (T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])
    T = T[keep]
    # duplicati: stessa terna di celle (in qualunque ordine), si tiene la prima occorrenza
    _, first = np.unique(np.sort(T, axis=1), axis=0, return_index=True)
    T = T[np.sort(first)]
    used = np.unique(T)
    remap = -np.ones(len(C), np.int64)
    remap[used] = np.arange(len(used))
    Fo = remap[T]
    _, _, P = igl.point_mesh_squared_distance(C[used], Vn, F)
    Vo = np.asarray(P, np.float64) * r + c
    el = edge_lengths(Vo, Fo) / r
    st = {"K_target": K, "n_verts_out": len(Vo), "n_faces_out": len(Fo), "n_subdiv": n_sub,
          "fine_faces": len(Ff), "edge_mean_over_L": float(el.mean() / L), "edge_cv": float(el.std() / el.mean()),
          "area_ratio_out_in": float(face_areas(Vo, Fo).sum() / face_areas(V, F).sum()),
          "seconds": time.perf_counter() - t0, **topology_stats(Fo, len(Vo))}
    return Vo, Fo, st


def file_seed(name: str) -> int:
    return zlib.crc32(name.encode()) ^ 4321


def _task(task):
    src, dst, flip = task
    V, F = load_mesh(src)
    Vo, Fo, st = remesh(V, F, file_seed(src.stem))
    np.savez(dst, V=Vo, F=np.ascontiguousarray(Fo[:, ::-1]) if flip else Fo)
    sid, topo = src.stem.split("_GTready_")
    return {"subject": sid, "topology": topo, "n_verts_in": len(V), **st}


def cmd_stage(a) -> None:
    from zs_stage import TOPOLOGIES, select_subjects
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    subjects = select_subjects(a.view_dir, a.seed)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(a.view_dir / f"{s}_GTready_{t}.npz", a.out_dir / f"{s}_GTready_{t}.npz", a.flip_faces)
             for s in subjects for t in TOPOLOGIES]
    t0 = time.time()
    with mp.get_context("fork").Pool(a.workers) as pool:
        rows = pool.map(_task, tasks, chunksize=1)
    with open(a.records, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (a.out_dir.parent / "subjects.json").write_text(json.dumps(
        {"seed": a.seed, "view_dir": str(a.view_dir), "transform": f"e3b_remesh L={L_EDGE}",
         "flip_faces": a.flip_faces, "subjects": subjects}, indent=1) + "\n")
    nv = np.array([r["n_verts_out"] for r in rows])
    print(f"[remesh] {len(rows)} mesh in {time.time() - t0:.0f}s -> {a.out_dir}; vertici in uscita mediana {np.median(nv):.0f} "
          f"[{nv.min()}, {nv.max()}], lato/L medio {np.median([r['edge_mean_over_L'] for r in rows]):.3f}, "
          f"spigoli non-manifold totali {sum(r['nonmanifold_edges'] for r in rows)}", flush=True)


def cmd_probe(a) -> None:
    """Prova su poche mesh: statistiche del rimesh, nessuna eval."""
    rows = []
    for p in a.files:
        V, F = load_mesh(p)
        Vo, Fo, st = remesh(V, F, file_seed(p.stem))
        rows.append({"file": p.name, "n_verts_in": len(V), **st})
        print(json.dumps(rows[-1]), flush=True)
        if a.save_dir:
            a.save_dir.mkdir(parents=True, exist_ok=True)
            np.savez(a.save_dir / p.name, V=Vo, F=Fo)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--view-dir", type=Path, required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--records", type=Path, required=True)
    s.add_argument("--flip-faces", action="store_true", help="convenzione BFM (come zs_stage --flip-faces)")
    s.add_argument("--workers", type=int, default=16)
    q = sub.add_parser("probe")
    q.add_argument("files", type=Path, nargs="+")
    q.add_argument("--save-dir", type=Path, default=None)
    a = p.parse_args()
    {"stage": cmd_stage, "probe": cmd_probe}[a.cmd](a)


if __name__ == "__main__":
    main()
