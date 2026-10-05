#!/usr/bin/env python3
"""Tempo del loader congelato per campione, a parita' di supporto (/tmp del nodo).

    srun -p cpu -c 4 --mem=16G -t 00:15:00 env AAU_NV= aau/run.sh aau/data_scale/loader_timing.py

60 mesh ICT (ict4500-ict4509 x 6 topologie, operatori areanorm in uso) copiate su /tmp in
tre codifiche: ``ref`` (np.savez non compresso, come oggi), ``i32u`` (indici e facce int32,
NON compresso), ``i32z`` (int32, compresso). Due passate per codifica, si tiene la seconda
(cache di pagina calda: e' il regime di un'epoca dopo la prima).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "face_embedding/gt_encdec/remeshing/intrinsic"))
from robustness.data_utils import GTReadyDataset  # noqa: E402

SRC = REPO_ROOT / "datasets/ICT/topo_withops"
TOPOS = ("original", "remesh", "crop", "noisy", "down8k", "up60k")
names = [f"ict{i:04d}_GTready_{t}.npz" for i in range(4500, 4510) for t in TOPOS]
tmp = Path(tempfile.mkdtemp(prefix="wbes_loader_"))
out = {}
try:
    for enc in ("ref", "i32u", "i32z"):
        d = tmp / enc
        d.mkdir()
        for n in names:
            if enc == "ref":
                shutil.copy(SRC / n, d / n)
                continue
            with np.load(SRC / n) as z:
                a = {k: z[k] for k in z.files}
            a["faces"] = a["faces"].astype(np.int32)
            for b in ("L", "gradX", "gradY"):
                a[f"{b}_indices"] = a[f"{b}_indices"].astype(np.int32)
            (np.savez if enc == "i32u" else np.savez_compressed)(d / n, **a)
        ds = GTReadyDataset(str(d))
        for rep in range(2):
            t0 = time.time()
            for i in range(len(ds.files)):
                assert ds[i] is not None
            dt = (time.time() - t0) / len(ds.files)
        size = sum((d / n).stat().st_size for n in names) / len(names)
        out[enc] = {"loader_s_per_sample": dt, "bytes_per_mesh": size}
        print(enc, json.dumps(out[enc]), flush=True)
        shutil.rmtree(d)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
(REPO_ROOT / "aau/data_scale/loader_timing.json").write_text(json.dumps(out, indent=2) + "\n")
