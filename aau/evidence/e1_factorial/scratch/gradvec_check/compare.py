#!/usr/bin/env python3
"""Confronta, file per file e tensore per tensore, gli operatori del pre-pass originale e di quello vettorizzato."""
import ast
import json
import sys
from pathlib import Path

import numpy as np

a, b = Path(sys.argv[1]), Path(sys.argv[2])
ta, tb = (ast.literal_eval(Path(p).read_text().strip()) for p in sys.argv[3:5])
fa = sorted(p.name for p in a.glob("*.npz"))
fb = sorted(p.name for p in b.glob("*.npz"))
assert fa == fb and fa, (len(fa), len(fb))
bad, n_arr, kinds, mags = [], 0, {}, []
for n in fa:
    with np.load(a / n) as za, np.load(b / n) as zb:
        if sorted(za.files) != sorted(zb.files):
            bad.append((n, "chiavi diverse"))
            continue
        for k in za.files:
            n_arr += 1
            x, y = za[k], zb[k]
            if x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y):
                bad.append((n, k))
                if x.shape == y.shape:
                    d = np.abs(x.astype(np.float64) - y.astype(np.float64))
                    mags.append({"file": n, "array": k, "n_diff": int((d > 0).sum()), "n": int(d.size),
                                 "max_abs": float(d.max()), "max_rel": float(d.max() / max(np.abs(x).max(), 1e-30)),
                                 "max_ulp32": float(np.max(np.abs(x.view(np.int32).astype(np.int64) - y.view(np.int32).astype(np.int64))))
                                 if x.dtype == np.float32 else None})
    lab = n[:-4].split("_GTready_")[1]
    kinds[lab] = kinds.get(lab, 0) + 1
rep = {"host": sys.argv[5], "n_files": len(fa), "n_arrays": n_arr, "topologies": kinds, "hook_lines_vec": int(sys.argv[6]),
       "identical": not bad, "n_files_different": len({f for f, _ in bad}), "n_arrays_different": len(bad),
       "max_abs_diff": max((m["max_abs"] for m in mags), default=0.0), "max_rel_diff": max((m["max_rel"] for m in mags), default=0.0),
       "max_ulp_fp32": max((m["max_ulp32"] or 0 for m in mags), default=0), "frac_elements_different":
       sum(m["n_diff"] for m in mags) / max(sum(m["n"] for m in mags), 1), "differences": mags[:10],
       "cpu_seconds_orig": ta["cpu_seconds"], "cpu_seconds_vec": tb["cpu_seconds"],
       "cpu_s_per_mesh_orig": ta["cpu_seconds"] / ta["n_computed"], "cpu_s_per_mesh_vec": tb["cpu_seconds"] / tb["n_computed"],
       "speedup": ta["cpu_seconds"] / tb["cpu_seconds"]}
Path("aau/runs/evidence/e1/gradvec_check.json").write_text(json.dumps(rep, indent=1) + "\n")
print(json.dumps(rep, indent=1))
