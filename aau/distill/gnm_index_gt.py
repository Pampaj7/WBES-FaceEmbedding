#!/usr/bin/env python3
"""Indice dei tar e GT d'identita' delle identita' GNM generate da ``gen_gnm_shard.py``.

    aau/run.sh aau/distill/gnm_index_gt.py --shards-dir datasets/GNM_DISTILL/shards --gt-dir datasets/GNM_DISTILL/gt
    (gnm_index_gt.sbatch; formato in datasets/GNM_DISTILL/README.md)

1. ``<shards-dir>/index.npz``: lo stesso indice di ``aau/data_scale/cache_budget.py index`` (chiavi
   ``tars``, ``tar_id``, ``names``, ``offset``, ``size``, ``n``, ``m``, ``E``; ``_index_one`` importata),
   sui tar ``gnm_shard_*.tar`` (il glob di cache_budget e' ``shard_*.tar``). Lo leggono
   ``prepass_ops.py --tar-index`` e ``train_steps.py`` (``tar_index`` della data-spec).
2. ``<gt-dir>/gnm_matrix_distances_maxabs.npz``: la GT di ICT (``v2_work/genict/build_ict_gt_matrix.py``,
   ``aau/data_scale/build_gt.py``): vertex-mean-L2 fra le ``original`` NEUTRE normalizzate maxabs
   (centro sulla media dei vertici, divisione per il massimo valore assoluto), divisa per il suo
   massimo (``D_orig``, massimo 1) e ``names``; il massimo grezzo e le statistiche in ``manifest.json``
   (``normalization_scale.maxabs``, come ``datasets/ICT/gt/manifest.json``). Calcolo:
   ``build_gt.vml2_matrix_host`` (importata) su GPU. Le espressioni prendono la GT della loro identita'.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "data_scale"))

from build_gt import originals_of, vml2_matrix_host  # noqa: E402
from cache_budget import _index_one  # noqa: E402


def build_index(shards_dir: Path, n_proc: int) -> dict:
    """Come ``cache_budget.build_index``, sul glob ``gnm_shard_*.tar``."""
    tars = sorted(str(p) for p in shards_dir.glob("gnm_shard_*.tar"))
    with ProcessPoolExecutor(n_proc) as ex:
        parts = list(ex.map(_index_one, tars))
    tar_id = np.concatenate([np.full(len(p["names"]), i, dtype=np.int16) for i, p in enumerate(parts)])
    cat = lambda k, dt: np.concatenate([np.asarray(p[k], dtype=dt) for p in parts])  # noqa: E731
    out = shards_dir / "index.npz"
    tmp = out.with_name(".index.tmp.npz")
    names = np.array([n for p in parts for n in p["names"]])
    np.savez(tmp, tars=np.array([Path(p["tar"]).name for p in parts]), tar_id=tar_id, names=names,
             offset=cat("offs", np.int64), size=cat("sizes", np.int64),
             n=cat("n", np.int32), m=cat("m", np.int32), E=cat("E", np.int32))
    tmp.replace(out)
    labels = {}
    for n in names:
        lab = str(n).split("_GTready_")[1][:-4]
        labels[lab] = labels.get(lab, 0) + 1
    return {"n_tars": len(tars), "n_members": int(len(names)), "per_label": dict(sorted(labels.items()))}


def build_gt(shards_dir: Path, gt_dir: Path, device: str) -> dict:
    t0 = time.time()
    tars = sorted(shards_dir.glob("gnm_shard_*.tar"))
    names, V = [], []
    with ThreadPoolExecutor(8) as ex:
        for rows in ex.map(originals_of, tars):
            for name, v in rows:
                names.append(name)
                V.append(v.astype(np.float32))
    order = np.argsort([int(n[2:]) for n in names])
    names = [names[i] for i in order]
    if len(set(names)) != len(names) or len({len(V[i]) for i in order}) != 1:
        raise SystemExit("nomi duplicati o original con numeri di vertici diversi")
    Vs = np.stack([V[i] for i in order])
    del V
    t_read = time.time() - t0
    D = vml2_matrix_host(Vs, device=device)
    raw_max = float(D.max())
    off = D[np.triu_indices(len(D), 1)].astype(np.float64)
    gt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(gt_dir / "gnm_matrix_distances_maxabs.npz", D_orig=(D / raw_max).astype(np.float32),
             names=np.array(names))
    nn = np.where(np.eye(len(D), dtype=bool), np.inf, D).min(axis=1)
    man = {"n_identities": len(names), "variant": "original (neutra)", "metric": "vertex-mean-L2, maxabs per mesh",
           "normalization_scale": {"maxabs": raw_max},
           "offdiag_stats_unnormalized": {"maxabs": {"min": float(off.min()), "p1": float(np.percentile(off, 1)),
                                                     "median": float(np.median(off)), "max": raw_max}},
           "nearest_neighbour_unnormalized_median": float(np.median(nn)),
           "closest_pair_frac_of_median": float(off.min() / np.median(off)),
           "n_vertices": int(Vs.shape[1]), "seconds": {"read": t_read, "total": time.time() - t0}}
    (gt_dir / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    return man


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shards-dir", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--n-proc", type=int, default=16)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    t0 = time.time()
    idx = build_index(a.shards_dir, a.n_proc)
    print(f"[gnm-index] {json.dumps(idx)} in {time.time() - t0:.0f}s", flush=True)
    man = build_gt(a.shards_dir, a.gt_dir, a.device)
    print(f"[gnm-gt] {json.dumps(man)}", flush=True)


if __name__ == "__main__":
    main()
