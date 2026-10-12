#!/usr/bin/env python3
"""Il port di NoW (``pr_metrics.now_pair``) contro il codice NoW UFFICIALE sulle stesse coppie. Nessuna GT.

    v3_work/unified_gt/run.sh aau/competitors/protocolli/pr_official_check.py export --view hifi3d --n 20
    <venv now> python3 aau/competitors/protocolli/pr_official_check.py official --view hifi3d     (pr.sbatch, passo official)
    v3_work/unified_gt/run.sh aau/competitors/protocolli/pr_official_check.py compare --view hifi3d

``export``: coppie ordinate estratte a caso (seme 7, soggetti e topologie diverse), mesh in mm e 7 landmark di A e B
(fuori dal repo, ``$PR_EXT/official/``) e i nostri errori NoW originale e metrical (ICP su tutti i punti).
``official``: ``compute_errors`` di ``external/now_evaluation/scan2mesh_computations.py`` e di
``scan2mesh_computations_metrical.py`` (``predicted_mesh_unit='mm'``), clone ufficiale non modificato (commit
7bd1498), nel venv ``external/venvs/now`` (psbody.mesh, chumpy, sbody), con lo stub del MeshViewer di
``aau/recon/now_official.py`` (libGL assente; il protocollo non lo usa). Errore = media delle distanze restituite.
``compare``: scarto fra i due per coppia; ``aau/runs/evidence/protocolli/checks/official_<vista>.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
REPO = THIS.parents[2]
EXT = Path(os.environ.get("PR_EXT", Path.home() / "ext" / "protocolli"))
OUT = EXT / "official"
TIMEOUT_S = 600


def export(vname: str, n: int) -> None:
    sys.path.insert(0, str(THIS))
    import pr_common as pc
    import pr_metrics as pm
    import pr_run as prn
    v = pc.view(vname)
    pr = pc.read_prep(v["name"])
    rng = np.random.default_rng(7)
    rows, arrays = [], {}
    for k in range(n):
        a = int(rng.integers(len(v["items"])))
        b = int(rng.choice(prn.partners(v["items"], a)))
        VA, FA, _ = pc.mm_mesh(v, v["items"][a][2])
        VB, FB, _ = pc.mm_mesh(v, v["items"][b][2])
        LA, LB = pr["L68"][a][list(pc.LMK7)], pr["L68"][b][list(pc.LMK7)]
        surf = pm.Surf(VA, FA)
        Bc = pm.crop(VB, LB, 1.4)
        t0 = time.perf_counter()
        o = pm.now_pair(surf, LA, Bc, LB, metrical=False)
        t1 = time.perf_counter()
        m = pm.now_pair(surf, LA, Bc, LB, metrical=True)
        t2 = time.perf_counter()
        arrays.update({f"VA{k}": VA, f"FA{k}": FA, f"VB{k}": VB, f"FB{k}": FB, f"LA{k}": LA, f"LB{k}": LB})
        rows.append({"k": k, "a": v["items"][a][:2], "b": v["items"][b][:2], "ours_original": o["mean"],
                     "ours_metrical": m["mean"], "ours_original_s": t1 - t0, "ours_metrical_s": t2 - t1,
                     "ours_scale": o["scale"], "ours_n_ref": o["n_ref"]})
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{vname}.npz", **arrays)
    (OUT / f"{vname}_ours.json").write_text(json.dumps(rows, indent=1))
    print(f"[pr-official] export {vname}: {n} coppie", flush=True)


def install_meshviewer_stub() -> None:
    """Come ``aau/recon/now_official.install_meshviewer_stub``."""
    stub = types.ModuleType("psbody.mesh.meshviewer")

    class MeshViewer:  # noqa: D401 - stub
        def __init__(self, *a, **k):
            raise RuntimeError("MeshViewer non disponibile (niente libGL): il protocollo non lo usa")

    stub.MeshViewer = stub.MeshViewers = MeshViewer
    sys.modules["psbody.mesh.meshviewer"] = stub


def official(vname: str) -> None:
    install_meshviewer_stub()
    sys.path.insert(0, str(REPO / "external" / "now_evaluation"))
    import scan2mesh_computations as s2m
    import scan2mesh_computations_metrical as s2mm
    z = np.load(OUT / f"{vname}.npz")
    rows = json.loads((OUT / f"{vname}_ours.json").read_text())
    out = []
    import signal

    def alarm(*_):
        raise TimeoutError

    signal.signal(signal.SIGALRM, alarm)

    def timed(fn, *args, **kw):
        """Errore medio e secondi; NaN se non termina entro ``TIMEOUT_S`` (il dogleg di chumpy con un residuo NaN, es.
        triangoli di area nulla delle up60k, non riduce mai il raggio e con ``max_fevals`` infinito non si ferma)."""
        t0 = time.perf_counter()
        signal.alarm(TIMEOUT_S)
        try:
            d = fn(*args, **kw)
            return float(np.mean(d)), int(len(d)), time.perf_counter() - t0, ""
        except TimeoutError:
            return float("nan"), -1, time.perf_counter() - t0, f"non termina in {TIMEOUT_S} s"
        finally:
            signal.alarm(0)

    for r in rows:
        k = r["k"]
        args = (z[f"VB{k}"], z[f"FB{k}"], z[f"LB{k}"], z[f"VA{k}"], z[f"FA{k}"], z[f"LA{k}"])
        e_o, n_o, s_o, w_o = timed(s2m.compute_errors, *args)
        e_m, _, s_m, w_m = timed(s2mm.compute_errors, *args, predicted_mesh_unit="mm")
        out.append({"k": k, "official_original": e_o, "official_metrical": e_m, "official_n_ref": n_o,
                    "official_original_s": s_o, "official_metrical_s": s_m, "official_note": "; ".join(x for x in (w_o, w_m) if x)})
        print(f"[pr-official] {k}: originale {out[-1]['official_original']:.4f} (nostro {r['ours_original']:.4f}), "
              f"metrical {out[-1]['official_metrical']:.4f} (nostro {r['ours_metrical']:.4f})", flush=True)
    (OUT / f"{vname}_official.json").write_text(json.dumps(out, indent=1))


def compare(vname: str) -> None:
    sys.path.insert(0, str(THIS))
    import pr_common as pc
    from scipy.stats import spearmanr
    ours = {r["k"]: r for r in json.loads((OUT / f"{vname}_ours.json").read_text())}
    off = {r["k"]: r for r in json.loads((OUT / f"{vname}_official.json").read_text())}
    res = {"view": vname, "n": len(off), "clone": pc.CLONE_COMMITS["now_evaluation"], "pairs": []}
    for m in ("original", "metrical"):
        ks = [k for k in off if np.isfinite(off[k][f"official_{m}"])]
        x = np.asarray([ours[k][f"ours_{m}"] for k in ks])
        y = np.asarray([off[k][f"official_{m}"] for k in ks])
        rel = np.abs(x - y) / y
        res[m] = {"n_finite_official": len(ks), "abs_diff_mm_median": float(np.median(np.abs(x - y))), "abs_diff_mm_max": float(np.abs(x - y).max()),
                  "rel_diff_median": float(np.median(rel)), "rel_diff_max": float(rel.max()),
                  "spearman": float(spearmanr(x, y).correlation),
                  "seconds_ours_median": float(np.median([ours[k][f"ours_{m}_s"] for k in off])),
                  "seconds_official_median": float(np.median([off[k][f"official_{m}_s"] for k in ks]))}
    res["n_ref_equal"] = int(sum(ours[k]["ours_n_ref"] == off[k]["official_n_ref"] for k in off))
    res["official_not_finished"] = {str(k): off[k]["official_note"] for k in off if off[k].get("official_note")}
    res["pairs"] = [{**ours[k], **off[k]} for k in off]
    pc.atomic_json(pc.EVID / "checks" / f"official_{vname}.json", res)
    print(f"[pr-official] {vname}: originale {res['original']}, metrical {res['metrical']}, "
          f"stessi punti di riferimento {res['n_ref_equal']}/{res['n']}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("export", "official", "compare"))
    p.add_argument("--view", default="hifi3d")
    p.add_argument("--n", type=int, default=20)
    a = p.parse_args()
    {"export": lambda: export(a.view, a.n), "official": lambda: official(a.view),
     "compare": lambda: compare(a.view)}[a.step]()


if __name__ == "__main__":
    main()
