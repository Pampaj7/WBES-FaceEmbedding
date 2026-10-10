#!/usr/bin/env python3
"""Prodotti interni varifold in mm fra le mesh valutate di una vista (matrice di Gram per sigma).

    aau/outlineB/run_o3d.sh aau/baselines_param/bp_varifold.py hifi3d --device cuda --shard 0/1
    (bp.sbatch, passo varifold; definizione della misura: bp.varifold_measure)

Mesh = ``bp.meshes`` (100 soggetti x 5 topologie senza crop; FaMoS: 15 scansioni di galleria). <X_i, X_j> per
ogni sigma di ``bp.VARIFOLD_SIGMAS_MM`` con ``geometric_kernel.inner_products`` (kind ``varifold``, il kernel di
phase0 a blocchi), per le coppie i <= j con i nello shard (i mod n = k). Uscita
``<OUT_ROOT>/<vista>/varifold_<k>of<n>.npz``: ``G`` (S, n, n) con NaN fuori dallo shard, ``subjects``,
``topologies``, ``atoms``, ``sigmas_mm``, ``cell_mm``. La distanza la compone ``bp_paired.varifold_matrix``:
d_ij = sqrt(sum_s (G_ii + G_jj - 2 G_ij)), la formula di ``geometric_kernel.distances``.

``--check N``: N coppie calcolate sul device e su CPU (stesso codice), scarto relativo stampato; nient'altro.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import bp
import blmm  # noqa: E402  (aau/baselines_mm, messo su sys.path da bp)


def main() -> None:
    import sys
    import torch
    sys.path.insert(0, str(bp.AAU_DIR / "baselines"))
    import geometric_kernel as gk
    from geometric_matrix import to_device
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("view", choices=bp.VIEWS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--block", type=int, default=4096)
    p.add_argument("--check", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    k, nsh = (int(x) for x in a.shard.split("/"))
    out = bp.out_dir(a.view) / f"varifold_{k}of{nsh}.npz"
    if out.exists() and not a.overwrite and not a.check:
        print(f"[bp-vf] {out}: gia' presente", flush=True)
        return
    items = bp.meshes(a.view)
    t0 = time.time()
    cpu = [bp.varifold_measure(a.view, q) for _, _, q in items]
    M = [to_device(m, a.device) for m in cpu]
    atoms = np.asarray([len(m["areas"]) for m in cpu])
    print(f"[bp-vf] {a.view}: {len(M)} misure in {time.time() - t0:.0f}s, atomi {atoms.min()}/{int(np.median(atoms))}/"
          f"{atoms.max()} (min/med/max), device {a.device}", flush=True)
    sig = bp.VARIFOLD_SIGMAS_MM
    if a.check:
        torch.set_num_threads(4)
        rng = np.random.default_rng(0)
        for _ in range(a.check):
            i, j = rng.choice(len(M), 2, replace=False)
            dv = gk.inner_products(M[i], M[j], sig, kinds=("varifold",), block=a.block)["varifold"].cpu().numpy()
            cv = gk.inner_products(cpu[i], cpu[j], sig, kinds=("varifold",), block=a.block)["varifold"].numpy()
            print(f"[bp-vf] check {i},{j}: device {dv} cpu {cv} scarto relativo max {np.abs(dv / cv - 1).max():.2e}",
                  flush=True)
        return
    n = len(M)
    G = np.full((len(sig), n, n), np.nan)
    rows = [i for i in range(n) if i % nsh == k]
    t1 = time.time()
    for r, i in enumerate(rows):
        for j in range(i, n):
            G[:, i, j] = G[:, j, i] = gk.inner_products(M[i], M[j], sig, kinds=("varifold",),
                                                        block=a.block)["varifold"].cpu().numpy()
        if r % 25 == 0:
            print(f"[bp-vf] riga {r + 1}/{len(rows)}, {time.time() - t1:.0f}s", flush=True)
    blmm.atomic_savez(out, G=G, subjects=np.asarray([s for s, _, _ in items]),
                      topologies=np.asarray([t for _, t, _ in items]), atoms=atoms, sigmas_mm=np.asarray(sig),
                      cell_mm=bp.VARIFOLD_CELL_MM, rows=np.asarray(rows))
    print(f"[bp-vf] {a.view} shard {a.shard}: {len(rows)} righe in {time.time() - t1:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
