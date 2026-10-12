#!/usr/bin/env python3
"""Matrice completa: prodotti interni varifold delle crop (le righe che ``bp_varifold.py`` non ha calcolato).

    aau/outlineB/run_o3d.sh aau/full_matrix/fm_varifold.py hifi3d --device cuda     (fm.sbatch, passo varifold)

Mesh = ``bp.meshes`` (le 500 senza crop, nell'ordine degli shard esistenti) + le 100 crop dei soggetti valutati
(``blmm.mesh_path(vista, s, "crop")``), misura ``bp.varifold_measure`` (stessa definizione, stessa quantizzazione),
<X_i, X_j> con ``geometric_kernel.inner_products`` (kind ``varifold``, sigma di ``bp.VARIFOLD_SIGMAS_MM``, blocco
8192 come bp.sbatch) per le coppie con almeno una crop. Le 500 x 500 restano quelle di ``bp_varifold.py`` (non
ricalcolate, non toccate). Controllo: ``--check`` coppie senza crop prese a caso ricalcolate su questo device contro
gli shard esistenti (scarto relativo massimo). Uscita ``aau/runs/evidence/full_matrix/varifold/<vista>.npz``: ``G``
(S, 600, 600) con NaN sul blocco senza crop, ``subjects``, ``topologies``, ``check_rel_max``, ``device``.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "baselines_param"))

import bp  # noqa: E402
import blmm  # noqa: E402  (aau/baselines_mm, messo su sys.path da bp)

OUT = bp.AAU_DIR / "runs" / "evidence" / "full_matrix" / "varifold"


def items600(view: str) -> list:
    """(soggetto, topologia, percorso): le 500 di ``bp.meshes`` poi le 100 crop."""
    return bp.meshes(view) + [(s, "crop", blmm.mesh_path(view, s, "crop")) for s in blmm.subjects(view)]


def _measure(task):
    view, q = task
    return bp.varifold_measure(view, q)


def existing(view: str) -> tuple[np.ndarray, list]:
    """Gram 500 x 500 dagli shard di ``bp_varifold.py`` (``bp_paired.varifold_gram``, copiata)."""
    G, keys = None, None
    for p in sorted(bp.out_dir(view).glob("varifold_*of*.npz")):
        with np.load(p) as z:
            k = list(zip([str(s) for s in z["subjects"]], [str(t) for t in z["topologies"]]))
            if keys is not None and k != keys:
                raise SystemExit(f"{p}: mesh diverse dagli altri shard")
            keys = k
            G = z["G"] if G is None else np.where(np.isfinite(G), G, z["G"])
    return G, keys


def main() -> None:
    import torch
    sys.path.insert(0, str(bp.AAU_DIR / "baselines"))
    import geometric_kernel as gk
    from geometric_matrix import to_device
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("view", choices=bp.CROP_VIEWS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--block", type=int, default=8192)
    p.add_argument("--check", type=int, default=40)
    p.add_argument("--workers", type=int, default=8, help="processi per le misure (CPU)")
    a = p.parse_args()
    out = OUT / f"{a.view}.npz"
    if out.exists():
        print(f"[fm-vf] {out}: gia' presente", flush=True)
        return
    items = items600(a.view)
    G0, keys0 = existing(a.view)
    if [(s, t) for s, t, _ in items[:500]] != keys0:
        raise SystemExit(f"{a.view}: mesh senza crop in ordine diverso dagli shard")
    t0 = time.time()
    with mp.get_context("fork").Pool(a.workers) as pool:
        M = [to_device(m, a.device) for m in pool.map(_measure, [(a.view, q) for _, _, q in items], chunksize=4)]
    print(f"[fm-vf] {a.view}: {len(M)} misure in {time.time() - t0:.0f}s, device {a.device} "
          f"({torch.cuda.get_device_name() if a.device == 'cuda' else 'cpu'})", flush=True)
    sig = bp.VARIFOLD_SIGMAS_MM

    def ip(i, j):
        return gk.inner_products(M[i], M[j], sig, kinds=("varifold",), block=a.block)["varifold"].cpu().numpy()

    rng = np.random.default_rng(0)
    rel = []
    for _ in range(a.check):
        i, j = rng.choice(500, 2, replace=False)
        rel.append(float(np.abs(ip(i, j) / G0[:, i, j] - 1).max()))
    print(f"[fm-vf] controllo su {a.check} coppie senza crop: scarto relativo max {max(rel):.2e}", flush=True)
    n = len(items)
    G = np.full((len(sig), n, n), np.nan)
    t1 = time.time()
    for r, i in enumerate(range(500, n)):
        for j in range(i + 1):
            G[:, i, j] = G[:, j, i] = ip(i, j)
        if r % 10 == 0:
            print(f"[fm-vf] crop {r + 1}/{n - 500}, {time.time() - t1:.0f}s", flush=True)
    blmm.atomic_savez(out, G=G, subjects=np.asarray([s for s, _, _ in items]),
                      topologies=np.asarray([t for _, t, _ in items]), sigmas_mm=np.asarray(sig),
                      cell_mm=bp.VARIFOLD_CELL_MM, check_rel_max=max(rel), check_rel=np.asarray(rel),
                      device=torch.cuda.get_device_name() if a.device == "cuda" else "cpu")
    print(f"[fm-vf] {a.view}: {n - 500} righe crop in {time.time() - t1:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
