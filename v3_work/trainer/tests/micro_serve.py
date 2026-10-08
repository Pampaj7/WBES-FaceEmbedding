#!/usr/bin/env python
"""Costo per mesh del servizio di un campione dalla cache (lato CPU del passo), per variante:

  v2        _serve come v2 (fast_data._rebuild_sparse): ricostruzione e .coalesce() di L, gradX, gradY
  nocoal    stessa ricostruzione con is_coalesced=True (gli indici vengono da un tensore gia' coalescente):
            stessi tensori, niente ordinamento
  compact   cache compatta (int32, niente L) come la serve oggi data_v3 (con .coalesce())
  e, con GPU, la copia sulla GPU: sample_to_device v1 (8 tensori) contro solo i 6 che il forward usa.
Controlla anche che 'nocoal' dia tensori identici a 'v2'.

    aau/run.sh v3_work/trainer/tests/micro_serve.py --out micro_serve.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import torch

THIS = Path(__file__).resolve().parent
TRAINER = THIS.parent
REPO = TRAINER.parents[1]
sys.path.insert(0, str(TRAINER))
import data_v3 as dv  # noqa: E402

FILES = ["datasets/REMESH/npz_data_topo_500_withops_areanorm/id0001_GTready_original.npz",
         "datasets/REMESH/npz_data_topo_500_withops_areanorm/id0001_GTready_up60k.npz",
         "datasets/ICT/train_ready/npz_withops/id10000_GTready_original.npz",
         "datasets/ICT/train_ready/npz_withops/id10000_GTready_down8k.npz"]


def serve_nocoal(s):
    out = dict(s)
    for k in dv.SPARSE_KEYS:
        t = s.get(k)
        if torch.is_tensor(t) and t.is_sparse:
            out[k] = torch.sparse_coo_tensor(t.indices(), t.values(), t.shape, is_coalesced=True)
    return out


def timeit(fn, reps=20):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp())
    for f in FILES:
        (tmp / Path(f).name).symlink_to((REPO / f).resolve())
    full = dv.CachedDataset(tmp, workers=2, compact=False, verbose=False)
    comp = dv.CachedDataset(tmp, workers=2, compact=True, verbose=False)
    rows = []
    dev = torch.device("cuda") if torch.cuda.is_available() else None
    for i, name in enumerate(full.files):
        raw = full._cache[i]
        r = {"file": name, "n": int(raw["verts"].shape[0]),
             "v2_ms": 1e3 * timeit(lambda: dv._serve(raw)),
             "nocoal_ms": 1e3 * timeit(lambda: serve_nocoal(raw)),
             "compact_ms": 1e3 * timeit(lambda: comp[i])}
        a_, b_ = dv._serve(raw), serve_nocoal(raw)
        r["nocoal_identical"] = all(torch.equal(a_[k].indices(), b_[k].indices()) and torch.equal(a_[k].values(), b_[k].values())
                                    and b_[k].is_coalesced() for k in dv.SPARSE_KEYS)
        if dev is not None:
            from robustness.data_utils import sample_to_device
            s = dv._serve(raw)
            r["to_gpu_v1_8keys_ms"] = 1e3 * timeit(lambda: sample_to_device(s, dev))
            keys = ("verts", "mass", "evals", "evecs", "gradX", "gradY")
            r["to_gpu_6keys_ms"] = 1e3 * timeit(lambda: {k: s[k].to(dev) for k in keys})
        rows.append(r)
        print(r, flush=True)
    a.out.write_text(json.dumps({"rows": rows, "cuda": dev is not None}, indent=1))


if __name__ == "__main__":
    main()
