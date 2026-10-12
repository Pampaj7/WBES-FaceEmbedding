#!/usr/bin/env python3
"""G0 (``aau/runs/evidence/diagnostics/PROTOCOL_G0.md`` sez. 2): rigenera in ``--out`` le mesh che non esistono piu' su
disco e fa il controllo K3 contro le tabelle di scala con cui i bracci sono stati valutati.

    AAU_NV="" aau/run.sh aau/diagnostics/g0_meshes.py --set flame2023_s1 --out /tmp/<job>/meshes --workers 32
    AAU_NV="" aau/run.sh aau/diagnostics/g0_meshes.py --set famos   ...
    AAU_NV="" aau/run.sh aau/diagnostics/g0_meshes.py --set heldout ...          (g0.sbatch, passo teacher)

Un insieme per processo (le sorgenti dello stream e i moduli del trainer hanno nomi in comune):
- ``flame2023_s1``: ``d1_gen.identities`` + ``d1_gen._gen`` (``work_mesh`` suddivisa 1-a-4 + ``views.discretize``, semi
  di D1), riferimento ``d1/scale_table.npz``;
- ``famos``: ``d3_famos._gen`` (le 80 persone TRAIN, semi di D3), riferimento ``d3/famos/scale_table.npz``;
- ``heldout``: ``fact_calib.stage`` (mesh grezze bfm, ict, gnm dalle sorgenti della spec), riferimento
  ``datasets/V3_OPS_CACHE/heldout_calib/scale_table.npz``.
Le funzioni importate scrivono in ``diag.RAW_IN`` / ``d3_famos.RAW_IN``: qui si ridirigono su ``--out/<insieme>``.

K3 per mesh: area grezza e max |coordinata - media| con scarto relativo <= 1e-6 contro la tabella (le tabelle non
hanno il numero di vertici: si confrontano gli intervalli per etichetta di ``gen.json``, dove c'e'); up60k fuori
tolleranza ma entro 1e-3 relativo: accettata e contata (decimazione qslim). Uscita: ``g0/meshes_<insieme>.json``.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import diag  # noqa: E402

G0 = diag.EV / "g0"
TOL, TOL_UP60K = 1e-6, 1e-3
REF = {"flame2023_s1": (diag.D1 / "scale_table.npz", diag.D1 / "gen.json"),
       "famos": (diag.EV / "d3" / "famos" / "scale_table.npz", diag.EV / "d3" / "famos" / "gen.json"),
       "heldout": (diag.CALIB_TABLE.with_suffix(".npz"), None)}


def gen_flame(out: Path, workers: int) -> list:
    import d1_gen as G
    name = "flame2023_s1"
    srcs = G.S.build_sources(["flame2023"], G.S.Unified())
    frames = json.loads(diag.FRAMES.read_text())["domains"]
    G._C.update(src={name: srcs["flame2023"]}, ident={name: G.identities(name, srcs["flame2023"])},
                u={k: float(v["u"]) for k, v in frames.items()})
    diag.RAW_IN = out                                         # d1_gen._gen scrive in diag.RAW_IN
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(G._gen, [(name, k) for k in range(len(G._C["ident"][name]))], chunksize=2)
    return [r for block in res for r in block]


def gen_famos(out: Path, workers: int) -> list:
    import d3_famos as F
    src = F.S.FamosSource(F.S.Unified())
    split = json.loads(F.S.FAMOS_SPLIT.read_text())
    persons = list(src.persons)
    if sorted(split["train"]) != persons or set(persons) & set(split["test"]):
        raise SystemExit("FaMoS: persone diverse dallo split TRAIN")
    F._C["src"] = src
    F.RAW_IN = out                                            # d3_famos._gen scrive in d3_famos.RAW_IN
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(F._gen, persons, chunksize=1)
    return [r for block in res for r in block]


def gen_heldout(out: Path) -> list:
    sys.path.insert(0, str(diag.REPO / "v3_work/trainer/tools"))
    import fact_calib as fc
    table = out / "scale_table.npz"
    fc.stage(out, table)
    with np.load(table, allow_pickle=True) as z:
        return [(str(n), str(d), float(a2), float(a), float(m), -1, -1) for n, d, a2, a, m in
                zip(z["names"], z["domain"], z["area_mm2"], z["area_raw"], z["maxabs_raw"])]


def k3(rows: list, which: str) -> dict:
    """Area grezza e maxabs per mesh contro la tabella di riferimento; intervalli di vertici per etichetta."""
    tab, gen = REF[which]
    with np.load(tab, allow_pickle=True) as z:
        ref = {str(n): (float(a), float(m)) for n, a, m in zip(z["names"], z["area_raw"], z["maxabs_raw"])}
    got = {r[0]: r for r in rows}
    want = [n for n in ref if n in got] if which != "flame2023_s1" else \
        [n for n in ref if n.startswith("id71")]              # la tabella di D1 ha tutti gli insiemi nuovi
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(ref))
    bad, up60k, worst = [], [], 0.0
    for n in want:
        a, m = ref[n]
        r = got[n]
        d = max(abs(r[3] - a) / max(abs(a), 1e-12), abs(r[4] - m) / max(abs(m), 1e-12))
        worst = max(worst, d)
        if d > TOL:
            (up60k if n.endswith("_up60k.npz") and d <= TOL_UP60K else bad).append((n, d))
    verts = {}
    if gen is not None:
        rng = json.loads(gen.read_text())
        rng = rng["sets"]["flame2023_s1"]["verts_by_label"] if which == "flame2023_s1" else rng["verts_by_label"]
        for lab in diag.LABELS:
            v = [r[5] for r in rows if r[0].endswith(f"_{lab}.npz")]
            verts[lab] = {"got": [int(min(v)), int(max(v))], "ref": rng[lab], "equal": [int(min(v)), int(max(v))] == rng[lab]}
    ok = not missing and not bad and all(x["equal"] or lab == "up60k" for lab, x in verts.items())
    return {"reference": str(tab.relative_to(diag.REPO)), "n_meshes": len(want), "missing": missing[:10],
            "n_missing": len(missing), "not_in_reference": extra[:10], "max_rel_diff": worst,
            "failed": [[n, d] for n, d in bad[:20]], "n_failed": len(bad),
            "up60k_accepted": [[n, d] for n, d in up60k[:20]], "n_up60k_accepted": len(up60k),
            "verts_by_label": verts, "pass": bool(ok)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", choices=sorted(REF), required=True)
    ap.add_argument("--out", type=Path, required=True, help="radice su /tmp: le mesh vanno in <out>/<insieme>")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    t0 = time.time()
    out = a.out / a.set
    out.mkdir(parents=True, exist_ok=True)
    rows = (gen_flame(out, a.workers) if a.set == "flame2023_s1" else
            gen_famos(out, a.workers) if a.set == "famos" else gen_heldout(out))
    rep = {"definition": "PROTOCOL_G0.md sez. 2 e K3", "set": a.set, "out": str(out), "n_meshes": len(rows),
           "K3": k3(rows, a.set), "seconds": time.time() - t0, "host": os.uname().nodename}
    diag.atomic_json(G0 / f"meshes_{a.set}.json", rep)
    print(f"[g0-meshes] {a.set}: {len(rows)} mesh in {out} ({rep['seconds']:.0f}s), K3 {rep['K3']['pass']}: "
          f"scarto massimo {rep['K3']['max_rel_diff']:.1e}, fallite {rep['K3']['n_failed']}, "
          f"up60k accettate {rep['K3']['n_up60k_accepted']}, mancanti {rep['K3']['n_missing']}", flush=True)
    if not rep["K3"]["pass"]:
        raise SystemExit(f"K3 non passa: {json.dumps(rep['K3'])[:600]}")


if __name__ == "__main__":
    main()
