#!/usr/bin/env python3
"""Chamfer sulla regione stabile all'espressione (e, come controllo, sulla mesh intera), tutte le coppie di mesh.

    aau/run.sh aau/zs3dmm/zs_region_chamfer.py --domain fv --view-dir <vista>/npz --out <baselines>/region_chamfer.npz
    (zs_baselines.sbatch, passo ``region``)

Baseline del protocollo dichiarato in ``aau/runs/ws_faceverse_expr/protocol.md`` (revisione 1),
la cui regola si riporta qui com'e', fissata prima di vedere i risultati e uguale per tutti i
domini:

  1. mesh normalizzata maxabs (centro sulla media dei vertici, divisione per max|coordinata|);
  2. frame canonico del dominio, assi "alto" e "avanti" dalla tabella delle convenzioni
     (aau/runs/ws_frame): BFM e FaceVerse alto -y / avanti -z; ICT e HIFI3D alto +y / avanti +z;
  3. punta del naso = vertice con la coordinata "avanti" massima;
  4. regione = vertici con coordinata "alto" >= quella della punta del naso (fronte,
     sopracciglia, occhi, dorso del naso, zigomi superiori; fuori bocca, mandibola, guance
     inferiori); triangoli con tutti e tre i vertici nella regione;
  5. regione ricentrata sul suo baricentro e riscalata maxabs su se stessa;
  6. 4096 punti campionati per area sui triangoli, seme fisso per mesh (crc32 del nome);
  7. Chamfer simmetrico = media delle distanze al quadrato punto -> vicino piu' prossimo, nei
     due versi, mediata.

``chamfer_full``: passi 1, 6, 7 sulla mesh intera, stessa implementazione: la differenza con
``chamfer_stable`` isola l'effetto della regione.

I soggetti sono quelli di ``zs_stage.select_subjects`` (gli stessi 100 dello zero-shot), tutte e 6
le topologie: matrici (600, 600) simmetriche, diagonale zero, con ``subjects`` / ``topologies``
per riga. Nessuna GPU.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import zlib
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from zs_stage import TOPOLOGIES, select_subjects  # noqa: E402

# (asse, segno) di "alto" e "avanti" nel frame dei dati del dominio
FRAMES = {
    "bfm": {"up": (1, -1.0), "forward": (2, -1.0)},
    "fv": {"up": (1, -1.0), "forward": (2, -1.0)},
    "ict": {"up": (1, 1.0), "forward": (2, 1.0)},
    "hifi": {"up": (1, 1.0), "forward": (2, 1.0)},
}
N_POINTS = 4096

_PTS: list[np.ndarray] = []
_TREES: list[cKDTree] = []


def maxabs_normalize(V: np.ndarray) -> np.ndarray:
    Vc = np.asarray(V, dtype=np.float64)
    Vc = Vc - Vc.mean(axis=0, keepdims=True)
    scale = float(np.abs(Vc).max())
    return Vc / scale if scale > 1e-12 else Vc * 0.0


def stable_region(V: np.ndarray, F: np.ndarray, domain: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Passi 2-5: (V, F) della regione stabile e frazione di vertici tenuti."""
    (ua, us), (fa, fs) = FRAMES[domain]["up"], FRAMES[domain]["forward"]
    up, forward = us * V[:, ua], fs * V[:, fa]
    keep = up >= up[int(np.argmax(forward))]
    Fr = F[keep[F].all(axis=1)]
    used = np.unique(Fr)
    remap = -np.ones(len(V), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return maxabs_normalize(V[used]), remap[Fr], float(keep.mean())


def sample_surface(V: np.ndarray, F: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Punti uniformi per area sui triangoli."""
    rng = np.random.default_rng(seed)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    tri = rng.choice(len(F), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    return (1 - s)[:, None] * a[tri] + (s * (1 - r2))[:, None] * b[tri] + (s * r2)[:, None] * c[tri]


def load_points(task: tuple[str, str]) -> dict:
    path, domain = task
    with np.load(path) as d:
        V, F = (d["V"], d["F"]) if "V" in d else (d["verts"], d["faces"])
    V, F = maxabs_normalize(V), np.asarray(F, dtype=np.int64)
    seed = zlib.crc32(Path(path).name.encode())
    Vr, Fr, frac = stable_region(V, F, domain)
    return {"full": sample_surface(V, F, N_POINTS, seed), "stable": sample_surface(Vr, Fr, N_POINTS, seed),
            "kept_vertex_fraction": frac}


def _init(points: list[np.ndarray]) -> None:
    global _PTS, _TREES
    _PTS = points
    _TREES = [cKDTree(p) for p in points]


def chamfer_row(i: int) -> tuple[int, np.ndarray]:
    row = np.zeros(len(_PTS))
    for j in range(i + 1, len(_PTS)):
        d_ij = _TREES[j].query(_PTS[i])[0]
        d_ji = _TREES[i].query(_PTS[j])[0]
        row[j] = 0.5 * (np.mean(d_ij ** 2) + np.mean(d_ji ** 2))
    return i, row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--domain", required=True, choices=sorted(FRAMES))
    p.add_argument("--view-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1234, help="WBES_EVAL_SEED: scelta dei soggetti")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()

    subjects = select_subjects(a.view_dir, a.seed)
    names = [(s, t) for s in subjects for t in TOPOLOGIES]
    paths = [str(a.view_dir / f"{s}_GTready_{t}.npz") for s, t in names]
    print(f"[region-chamfer] {len(subjects)} soggetti x {len(TOPOLOGIES)} topologie = {len(paths)} mesh, "
          f"dominio {a.domain} {FRAMES[a.domain]}", flush=True)
    with mp.get_context("fork").Pool(a.workers) as pool:
        loaded = pool.map(load_points, [(q, a.domain) for q in paths])
    frac = np.asarray([x["kept_vertex_fraction"] for x in loaded])
    print(f"[region-chamfer] frazione di vertici nella regione: min {frac.min():.3f} "
          f"mediana {np.median(frac):.3f} max {frac.max():.3f}", flush=True)

    out = {}
    for tag in ("stable", "full"):
        D = np.zeros((len(paths), len(paths)))
        with mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=([x[tag] for x in loaded],)) as pool:
            for i, row in pool.imap_unordered(chamfer_row, range(len(paths)), chunksize=4):
                D[i] = row
        out[f"chamfer_{tag}"] = D + D.T
        print(f"[region-chamfer] {tag}: fatto", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.stem + ".tmp.npz")
    np.savez_compressed(tmp, **out, subjects=np.asarray([s for s, _ in names], dtype="U16"),
                        topologies=np.asarray([t for _, t in names], dtype="U16"),
                        kept_vertex_fraction=frac, domain=a.domain, n_points=N_POINTS,
                        frame=json.dumps(FRAMES[a.domain]))
    tmp.replace(a.out)
    print(f"[region-chamfer] scritto {a.out}", flush=True)


if __name__ == "__main__":
    main()
