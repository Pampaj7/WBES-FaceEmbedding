#!/usr/bin/env python3
"""Le 6 topologie REMESH-2 per le patch FLAME di ``flame_crop.py``.

    aau/run.sh aau/flame/make_flame_topologies.py \
        --in-dir datasets/FLAME/identities --out-dir datasets/FLAME/topo --n-cores 8

Nessuna regola nuova: ``process_subject`` e' quella di ``v2_work/genict/make_ict_topologies.py``,
importata e non riscritta (igl, niente open3d), quindi le varianti sono le stesse di ICT --
``original`` (la patch com'e'), ``remesh`` (2 smoothing + decimazione 0.7x), ``crop`` (banda
di bordo di ``datasets/remesh.py``), ``noisy`` (rumore 0.003 x diagonale bbox, seme = indice
del soggetto), ``down8k``/``up60k`` (decimazione allo stesso RAPPORTO 0.345x / 2.584x che
hanno su BFM rispetto alla ``original`` del proprio modello). Cambia solo il prefisso dei file
(``flameNNNN`` invece di ``ictNNNN``). La pipeline FLAME storica
(``v2_work/genflame/make_flame_topologies.py``) non si usa: dipende da open3d, dal modulo
``datasets.expand_remesh_topologies`` assente nel checkout e dalla corrispondenza
BFM->FLAME, che e' un asset FLAME.

Output: ``flameNNNN_GTready_<topologia>.npz`` con chiavi ``V``/``F``. Riprendibile: salta le
identita' gia' complete. Alla fine controlla ``crop`` su TUTTE le identita' (``check_crop``: il
ripiego di ``mesh_ops.trim_far_boundary_band`` lascerebbe crop == original). ``--demo``
controlla dimensioni, indici fissi e crop sulle prime 3.
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
from mesh_ops import MIN_KEEP_RATIO  # noqa: E402

# Controllo di `crop`. mesh_ops.trim_far_boundary_band (porting di datasets/remesh.py) taglia
# una banda di bordo, ma se la parte tenuta scende sotto MIN_KEEP_RATIO dei vertici restituisce
# la mesh INTATTA: quell'identita' avrebbe crop == original e la coppia crop<->original
# misurerebbe una copia. Lo stesso ripiego succede se la banda non trova bordo. Per ogni identita'
# si richiede quindi n_vertici(crop) < n_vertici(original), con frazione tenuta in
# [CROP_KEEP_MIN, 1): il minimo e' MIN_KEEP_RATIO con un margine per la pulizia
# (prepare_open_surface) che segue il taglio. Oltre CROP_MAX_FAIL_FRACTION di identita' fuori
# regola la build fallisce; sotto, le identita' vengono elencate nel log.
CROP_KEEP_MIN = MIN_KEEP_RATIO - 0.05
CROP_MAX_FAIL_FRACTION = 0.02


def crop_keep_fraction(out_dir: Path, subject: str) -> float:
    """n_vertici(crop) / n_vertici(original) di un'identita'."""
    n = {}
    for v in ("original", "crop"):
        with np.load(out_dir / f"{subject}_GTready_{v}.npz") as d:
            n[v] = len(d["V"])
    return n["crop"] / n["original"]


def check_crop(out_dir: Path, subjects: list[str], max_fail_fraction: float) -> None:
    """Elenca le identita' con il ripiego del crop; esce 1 oltre max_fail_fraction."""
    bad = []
    fracs = []
    for subject in subjects:
        f = crop_keep_fraction(out_dir, subject)
        fracs.append(f)
        if not CROP_KEEP_MIN <= f < 1.0:
            bad.append((subject, f))
    fracs = np.asarray(fracs)
    print(f"[crop] frazione di vertici tenuta: min {fracs.min():.3f} mediana {np.median(fracs):.3f} "
          f"max {fracs.max():.3f}; fuori da [{CROP_KEEP_MIN:.2f}, 1): {len(bad)}/{len(subjects)}",
          flush=True)
    for subject, f in bad:
        print(f"[crop] ATTENZIONE {subject}: crop tiene {f:.3f} dei vertici di original"
              + (" (ripiego: crop == original)" if f >= 1.0 else ""), flush=True)
    if len(bad) > max_fail_fraction * len(subjects):
        raise SystemExit(f"[crop] ERRORE: {len(bad)}/{len(subjects)} identita' fuori regola, "
                         f"oltre il {max_fail_fraction:.0%}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--in-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-subjects", type=int, default=0, help="0 = tutti")
    p.add_argument("--n-cores", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-crop-fail", type=float, default=CROP_MAX_FAIL_FRACTION,
                   help="frazione massima di identita' col ripiego del crop")
    a = p.parse_args()

    subjects = sorted(a.in_dir.glob("flame[0-9]*.npz"))
    if not subjects:
        raise FileNotFoundError(f"nessun flame*.npz in {a.in_dir}")
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
    # Anche sulle identita' saltate: una ripresa controlla i file che trova, non solo i nuovi.
    check_crop(a.out_dir, [q.stem for q in subjects], a.max_crop_fail)


def demo(argv: list[str]) -> None:
    """Le 6 varianti esistono, hanno le dimensioni attese, original/noisy hanno indici fissi."""
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-subjects", type=int, default=3)
    a, _ = p.parse_known_args(argv)
    ids = sorted(a.in_dir.glob("flame[0-9]*.npz"))[: a.n_subjects]
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
        f = crop_keep_fraction(a.out_dir, q.stem)
        assert CROP_KEEP_MIN <= f < 1.0, \
            f"{q.stem}: crop tiene {f:.3f} dei vertici di original (atteso [{CROP_KEEP_MIN:.2f}, 1))"
    print("demo OK: 6 varianti per identita', dimensioni sul target, original/noisy a indici fissi, "
          "crop piu' piccolo di original")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo([x for x in sys.argv[1:] if x != "--demo"])
    else:
        main()
