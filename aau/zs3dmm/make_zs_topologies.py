#!/usr/bin/env python3
"""Le 6 topologie REMESH-2 per le identita' di ``zs_identities.py`` (HIFI3D, FaceVerse).

    aau/run.sh aau/zs3dmm/make_zs_topologies.py --prefix hifi \
        --in-dir datasets/HIFI3D/identities --out-dir datasets/HIFI3D/topo --n-cores 8

Nessuna regola nuova: ``process_subject`` e' quella di ``v2_work/genict/make_ict_topologies.py``,
importata e non riscritta (igl, niente open3d), quindi le varianti sono le stesse di ICT --
``original`` (la patch com'e'), ``remesh`` (2 smoothing + decimazione 0.7x), ``crop`` (banda
di bordo di ``datasets/remesh.py``), ``noisy`` (rumore 0.003 x diagonale bbox, seme = ultime 4
cifre del nome), ``down8k``/``up60k`` (decimazione allo stesso RAPPORTO 0.345x / 2.584x che
hanno su BFM rispetto alla ``original`` del proprio modello). Cambia solo il prefisso dei file
(``hifiNNNN`` / ``fvNNNN`` invece di ``ictNNNN``).

Output: ``<prefix>NNNN_GTready_<topologia>.npz`` con chiavi ``V``/``F``. Riprendibile: salta le
identita' gia' complete. ``--demo`` controlla dimensioni e indici fissi sulle prime 3.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
# A livello di modulo: con lo start method spawn i worker rieseguono questo file, e devono
# trovare make_ict_topologies per de-serializzare process_subject.
sys.path.insert(0, str(REPO_ROOT / "v2_work" / "genict"))

from make_ict_topologies import (  # noqa: E402
    REMESH_DECIMATION,
    VARIANTS,
    process_subject,
    triangle_targets,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prefix", required=True)
    p.add_argument("--in-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-subjects", type=int, default=0, help="0 = tutti")
    p.add_argument("--n-cores", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    subjects = sorted(a.in_dir.glob(f"{a.prefix}[0-9]*.npz"))
    if not subjects:
        raise FileNotFoundError(f"nessun {a.prefix}*.npz in {a.in_dir}")
    if a.n_subjects:
        subjects = subjects[: a.n_subjects]
    a.out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(str(q), str(a.out_dir), a.overwrite) for q in subjects]
    print(f"{len(tasks)} identita' {a.in_dir} -> {a.out_dir}  workers={max(1, a.n_cores)}", flush=True)

    if a.n_cores > 1:
        pool = mp.get_context("spawn").Pool(processes=a.n_cores)
        results = pool.imap_unordered(process_subject, tasks)
    else:
        pool, results = None, map(process_subject, tasks)

    tally = {"[ok]": 0, "[skip]": 0, "[fail]": 0}
    failures = []
    try:
        for i, (status, msg) in enumerate(results, start=1):
            tally[status] += 1
            if status == "[fail]":
                failures.append(msg)
            if status != "[ok]" or i % 100 == 0 or i <= 5:
                print(f"[{i}/{len(tasks)}] {status} {msg}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    print(f"\nFatto. ok={tally['[ok]']} skip={tally['[skip]']} fail={tally['[fail]']}")
    for msg in failures[:20]:
        print(f"  - {msg}")
    if tally["[fail]"]:
        raise SystemExit(1)


def demo(argv: list[str]) -> None:
    """Le 6 varianti esistono, hanno le dimensioni attese, original/noisy hanno indici fissi."""
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", required=True)
    p.add_argument("--in-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-subjects", type=int, default=3)
    a, _ = p.parse_known_args(argv)
    ids = sorted(a.in_dir.glob(f"{a.prefix}[0-9]*.npz"))[: a.n_subjects]
    assert ids, f"servono identita' in {a.in_dir}"

    with np.load(ids[0]) as d:
        n_orig = len(d["F"])
    down_target, up_target = triangle_targets(n_orig)
    expected = {"original": n_orig, "remesh": int(n_orig * REMESH_DECIMATION), "noisy": n_orig,
                "down8k": down_target, "up60k": up_target}
    print(f"original={n_orig} triangoli, target down8k={down_target} up60k={up_target}")

    faces_ref = {}
    for q in ids:
        for v in VARIANTS:
            path = a.out_dir / f"{q.stem}_GTready_{v}.npz"
            with np.load(path) as d:
                V, F = d["V"], d["F"]
            assert V.dtype == np.float32 and F.dtype == np.int32, (V.dtype, F.dtype)
            assert F.max() == len(V) - 1, f"{path.name}: vertici non referenziati"
            assert np.isfinite(V).all(), f"{path.name}: vertici non finiti"
            print(f"  {path.name}: V={len(V)} F={len(F)}")
            if v in expected:
                assert abs(len(F) - expected[v]) <= 0.02 * expected[v], \
                    f"{path.name}: {len(F)} triangoli, attesi ~{expected[v]}"
            if v in ("original", "noisy"):
                faces_ref.setdefault(v, F)
                assert np.array_equal(faces_ref[v], F), f"{v} non e' un insieme di indici fisso"
    print("demo OK: 6 varianti per identita', dimensioni sul target, original/noisy a indici fissi")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo([x for x in sys.argv[1:] if x != "--demo"])
    else:
        main()
