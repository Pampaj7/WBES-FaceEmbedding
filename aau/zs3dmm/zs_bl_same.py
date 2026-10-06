#!/usr/bin/env python3
"""Baseline faceBench sulle coppie STESSO SOGGETTO, topologie diverse: la parte mancante per retrieval e verifica.

    aau/outlineB/run_o3d.sh aau/zs3dmm/zs_bl_same.py fv 910000 --out-root <baselines> --workers 32

``alignment_matrix.py`` riempie solo le coppie di soggetti i<j (servono alla GT); retrieval e
verifica d'identita' vogliono anche (soggetto i in ta, soggetto i in tb). Qui, per ogni coppia
ordinata di topologie diverse (le 30 cross, crop compreso), le 100 coppie stesso-soggetto con la
STESSA pipeline (``alignment_matrix._run_chunk``, importata: ``run_geometry_pipeline`` di
faceBench, 4096 punti, ICP rigido, NICP), seme = 1000000 + indice del soggetto (fuori dagli
indici 0..4949 delle coppie i<j). Set di soggetti: ``<dominio>_heldout`` di ``zs_bl.py``, cioe'
gli stessi 100 dello zero-shot.

Output: ``<out-root>/matrices_same/<metrica>/<ta>__to__<tb>.npz`` con ``values`` (100,) e
``subjects`` nell'ordine del set. Riprendibile per coppia di topologie.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
sys.path.insert(0, str(AAU_DIR / "baselines"))
sys.path.insert(0, str(AAU_DIR / "outlineB"))
sys.path.insert(0, str(THIS_DIR))

import common  # noqa: E402
import alignment_matrix as am  # noqa: E402
from zs_bl import register  # noqa: E402

SEED_OFFSET = 1_000_000


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("uso: zs_bl_same.py <dominio> <id_offset> --out-root <dir> [--workers N]")
    domain, id_offset = sys.argv[1], int(sys.argv[2])
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-sample-points", type=int, default=4096)
    p.add_argument("--chunk", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(sys.argv[3:])

    set_name = f"{domain}_heldout"
    register(set_name, id_offset)
    subjects = common.subject_set(set_name)
    pairs = [(x, y) for x in common.TOPOLOGIES for y in common.TOPOLOGIES if x != y]
    print(f"[bl-same] {set_name}: {len(subjects)} soggetti (primi {subjects[:3]}), {len(pairs)} coppie "
          f"ordinate di topologie, mesh_root={common.MESH_ROOT}", flush=True)

    import multiprocessing as mp
    t0 = time.time()
    with mp.get_context("spawn").Pool(processes=a.workers) as pool:
        for ta, tb in pairs:
            outs = {m: a.out_root / "matrices_same" / m / f"{ta}__to__{tb}.npz" for m in am.PIPELINE_METRICS.values()}
            if all(o.exists() for o in outs.values()) and not a.overwrite:
                print(f"[bl-same] {ta}->{tb}: gia' presente, salto", flush=True)
                continue
            idx = list(range(len(subjects)))
            chunks = [([str(common.mesh_path(subjects[i], ta)) for i in idx[s:s + a.chunk]],
                       [str(common.mesh_path(subjects[i], tb)) for i in idx[s:s + a.chunk]],
                       [SEED_OFFSET + i for i in idx[s:s + a.chunk]], a.max_sample_points)
                      for s in range(0, len(idx), a.chunk)]
            results = [r for block in pool.map(am._run_chunk, chunks) for r in block]
            values = np.asarray([r[0] for r in results], dtype=np.float64)
            failed = [r[3] for r in results if r[2] != "ok"]
            for col, metric in enumerate(am.PIPELINE_METRICS.values()):
                outs[metric].parent.mkdir(parents=True, exist_ok=True)
                tmp = outs[metric].with_name(outs[metric].stem + ".tmp.npz")
                np.savez_compressed(tmp, values=values[:, col], subjects=np.asarray(subjects, dtype="U16"),
                                    metric=metric, topology_a=ta, topology_b=tb, seed_offset=SEED_OFFSET,
                                    n_failed=len(failed))
                os.replace(tmp, outs[metric])
            print(f"[bl-same] {ta}->{tb}: {len(results)} coppie, fallite {len(failed)}", flush=True)
    print(f"[bl-same] fine in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
