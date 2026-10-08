#!/usr/bin/env python
"""Il pre-pass con grad_vec (prepass_v3.py) scrive gli STESSI npz del pre-pass originale (prepass_ops.py).

Due soggetti dagli shard del run su scala (un ICT_SCALE con espressioni, un GNM): tutte le mesh, entrambi
i pre-pass, confronto array per array (verts, faces, mass, evals, evecs, indici e valori di L/gradX/gradY)
e tempo di parete.

    aau/run.sh v3_work/trainer/tests/test_prepass_vec.py --work /tmp/$SLURM_JOB_ID/pv --out prepass_vec.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
REPO = THIS.parents[2]
SPEC = REPO / "aau/runs/data_scale_runs/scale_bfm_ict_gnm_s1234_nocanon_noaug_20261007_1411/spec.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--subjects", default="id20000,id100004")
    ap.add_argument("--n-proc", type=int, default=4)
    a = ap.parse_args()
    spec = json.loads(SPEC.read_text())
    a.work.mkdir(parents=True, exist_ok=True)
    subj = a.work / "subjects.txt"
    subj.write_text("\n".join(a.subjects.split(",")) + "\n")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    times = {}
    for tag, script in (("orig", REPO / "aau/data_scale/prepass_ops.py"), ("orig2", REPO / "aau/data_scale/prepass_ops.py"),
                        ("vec", REPO / "v3_work/trainer/prepass_v3.py")):
        t0 = time.time()
        subprocess.run([sys.executable, str(script), "--out-dir", str(a.work / tag), "--subjects", str(subj),
                        "--tars", *spec["tars"], "--tar-index", spec["tar_index"], "--n-proc", str(a.n_proc),
                        "--convention", "areanorm"], check=True, env=env)
        times[tag] = time.time() - t0
    names = sorted(p.name for p in (a.work / "orig").glob("*.npz"))

    def compare(t1, t2):
        worst, worst_grad, worst_evec_sign, rows = 0.0, 0.0, 0.0, []
        for n in names:
            with np.load(a.work / t1 / n) as A, np.load(a.work / t2 / n) as B:
                d = {}
                for k in A.files:
                    x, y = A[k].astype(np.float64), B[k].astype(np.float64)
                    d[k] = float(np.abs(x - y).max()) if x.shape == y.shape and x.size else (0.0 if x.shape == y.shape else float("inf"))
                ex, ey = A["evecs"].astype(np.float64), B["evecs"].astype(np.float64)
                d["evecs_up_to_sign"] = float(np.minimum(np.abs(ex - ey).max(0), np.abs(ex + ey).max(0)).max())
                rows.append({"file": n, "per_key": d})
                worst = max(worst, max(v for k, v in d.items() if k != "evecs_up_to_sign"))
                worst_grad = max(worst_grad, *(d[k] for k in d if k.startswith(("gradX", "gradY"))))
                worst_evec_sign = max(worst_evec_sign, d["evecs_up_to_sign"])
        return {"max_abs_all": worst, "max_abs_gradXY": worst_grad, "max_abs_evecs_up_to_sign": worst_evec_sign,
                "rows": rows}

    out = {"n_meshes": len(names), "seconds": times, "orig_vs_orig2": compare("orig", "orig2"),
           "orig_vs_vec": compare("orig", "vec")}
    a.out.write_text(json.dumps(out, indent=1))
    for k in ("orig_vs_orig2", "orig_vs_vec"):
        r = out[k]
        print(f"{k}: {len(names)} mesh, max|delta| tutte le chiavi {r['max_abs_all']:.3e}, gradX/gradY "
              f"{r['max_abs_gradXY']:.3e}, evecs a meno del segno {r['max_abs_evecs_up_to_sign']:.3e}")
    print(f"tempo: orig {times['orig']:.0f}s, orig2 {times['orig2']:.0f}s, vec {times['vec']:.0f}s")


if __name__ == "__main__":
    main()
