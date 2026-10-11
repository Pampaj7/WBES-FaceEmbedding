#!/usr/bin/env python3
"""D4 (``PROTOCOL_D4.md`` sez. 6): costo per mesh della ridiscretizzazione canonica contro la pipeline attuale.

    aau/run.sh aau/diagnostics/d4_cost.py                  (d4.sbatch, passo cost; dopo il passo prep)

Operatori areanorm k 128 (la procedura di ``areanorm_operators.py``: centro dei vertici, area totale 1,
``compute_operators``) sulla mesh originale della vista e su quella canonica (``datasets/DIAG_D4``), per i primi
``N_SUBJ`` soggetti valutati per id x 6 topologie x 3 viste, nello stesso processo, un thread, alternati (originale,
canonica). Remesh: i tempi per mesh di ``d4/remesh_<vista>.npz`` (tutte le 1.800, un thread per processo). Embedding:
i tempi a parete del log di ``d4_embed.sbatch`` si aggiungono nel riepilogo. Uscita ``d4/cost.json``.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import d4_remesh as R  # noqa: E402
import diag  # noqa: E402

sys.path.insert(0, str(diag.REPO / "diffusion-net/src"))
N_SUBJ = 3


def ops_seconds(V: np.ndarray, F: np.ndarray) -> float:
    """Tempo degli operatori areanorm k 128 (``areanorm_operators.main``, senza la scrittura)."""
    import torch
    from diffusion_net.geometry import compute_operators
    t0 = time.perf_counter()
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int32)
    V = (V - V.mean(0)) / np.sqrt(R.area(V, F))
    compute_operators(torch.tensor(V, dtype=torch.float32), torch.tensor(F, dtype=torch.int32), k_eig=128)
    return time.perf_counter() - t0


def main() -> None:
    import torch
    torch.set_num_threads(1)
    recs = []
    for view in R.VIEWS:
        sk = R.store_info(view)
        vdir = Path(sk["view_dir"])
        for s in sorted(sk["subjects"])[:N_SUBJ]:
            for t in R.TOPOLOGIES:
                name = f"{s}_GTready_{t}.npz"
                Vo, Fo = R.load_vf(vdir / name)
                Vc, Fc = R.load_vf(R.STAGE / view / "in" / name)
                to = ops_seconds(Vo, Fo)
                tc = ops_seconds(Vc, Fc)
                recs.append({"view": view, "name": name, "topology": t, "n_vertices_original": int(len(Vo)),
                             "n_vertices_canonical": int(len(Vc)), "ops_original_s": to, "ops_canonical_s": tc})
                print(f"[d4-cost] {view} {name}: originale {len(Vo)} v. {to:.2f} s, canonica {len(Vc)} v. {tc:.2f} s",
                      flush=True)
    rem = {}
    for view in R.VIEWS:
        with np.load(R.D4 / f"remesh_{view}.npz") as z:
            cols = [str(c) for c in z["columns"]]
            sec = np.asarray(z["stats"])[:, cols.index("seconds")]
            topo = np.asarray([str(x) for x in z["topologies"]])
            names = [f"{a}_GTready_{b}.npz" for a, b in zip(z["subjects"], z["topologies"])]
        rem[view] = dict(zip(names, sec))
        rem[view + "|topo"] = dict(zip(names, topo))
    for r in recs:
        r["remesh_s"] = float(rem[r["view"]][r["name"]])
        r["canonical_total_s"] = r["remesh_s"] + r["ops_canonical_s"]
    allrem = np.concatenate([np.asarray(list(rem[v].values()), np.float64) for v in R.VIEWS])
    alltopo = np.concatenate([np.asarray(list(rem[v + "|topo"].values())) for v in R.VIEWS])

    def med(key, sel=None):
        x = np.asarray([r[key] for r in recs if sel is None or sel(r)], np.float64)
        return {"median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "mean": float(x.mean()), "n": int(len(x))}

    out = {"definition": "PROTOCOL_D4.md sez. 6", "n_subjects_per_view": N_SUBJ, "threads": 1,
           "host": os.uname().nodename, "per_mesh": recs,
           "remesh_all_1800": {"median": float(np.nanmedian(allrem)), "p90": float(np.nanpercentile(allrem, 90)),
                               "mean": float(np.nanmean(allrem)), "n": int(np.isfinite(allrem).sum()),
                               "by_topology": {t: {"median": float(np.nanmedian(allrem[alltopo == t])),
                                                   "p90": float(np.nanpercentile(allrem[alltopo == t], 90))}
                                               for t in R.TOPOLOGIES}},
           "sample": {k: med(k) for k in ("ops_original_s", "ops_canonical_s", "remesh_s", "canonical_total_s")},
           "sample_by_view": {v: {k: med(k, lambda r, v=v: r["view"] == v)
                                  for k in ("ops_original_s", "ops_canonical_s", "remesh_s", "canonical_total_s")}
                              for v in R.VIEWS},
           "sample_by_topology": {t: {k: med(k, lambda r, t=t: r["topology"] == t)
                                      for k in ("ops_original_s", "ops_canonical_s", "remesh_s", "canonical_total_s")}
                                  for t in R.TOPOLOGIES}}
    ratio = np.asarray([r["canonical_total_s"] / r["ops_original_s"] for r in recs])
    out["sample"]["ratio_canonical_total_over_original_ops"] = {"median": float(np.median(ratio)),
                                                                "p10": float(np.percentile(ratio, 10)),
                                                                "p90": float(np.percentile(ratio, 90))}
    R.D4.mkdir(parents=True, exist_ok=True)
    (R.D4 / "cost.json").write_text(json.dumps(out, indent=1) + "\n")
    s = out["sample"]
    print(f"[d4-cost] mediane per mesh ({len(recs)}): operatori originali {s['ops_original_s']['median']:.2f} s, "
          f"remesh {s['remesh_s']['median']:.2f} s + operatori canonici {s['ops_canonical_s']['median']:.2f} s = "
          f"{s['canonical_total_s']['median']:.2f} s; rapporto mediano "
          f"{s['ratio_canonical_total_over_original_ops']['median']:.2f}", flush=True)


if __name__ == "__main__":
    main()
