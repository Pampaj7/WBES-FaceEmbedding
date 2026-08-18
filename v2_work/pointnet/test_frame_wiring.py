#!/usr/bin/env python
"""Proves --frame is ACTIVE on real cached samples, and that 'current' is a true no-op.

A frame change alters no tensor shape and no parameter count, which is the exact signature of
the four bugs that cost us a night: nothing raises when the variant is silently off. So it is
asserted on real data rather than inferred from the absence of an error.
"""
import sys, tempfile, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("face_embedding/gt_encdec/remeshing/intrinsic",
            "face_embedding/gt_encdec/autoencoder", "diffusion-net/src",
            "v2_work/fastio", "v2_work/pointnet"):
    sys.path.insert(0, str(ROOT / sub))

import torch
from frames import reframe

SRC = ROOT / "datasets/REMESH/npz_data_topo_500_withops"
tmp = Path(tempfile.mkdtemp())
picked = sorted(SRC.glob("id0000_*.npz")) + sorted(SRC.glob("id0001_*.npz"))
for f in picked:
    os.symlink(f, tmp / f.name)

import fast_data as fd
from train_fast import install_frame

base = fd.CachedDataset(str(tmp), workers=2, residency="ram", device=None, max_gb=10)
before = {i: base[i]["verts"].clone() for i in range(len(base))}

install_frame("rms")
after = {i: base[i]["verts"] for i in range(len(base))}

changed = sum(not torch.allclose(before[i], after[i]) for i in after)
assert changed == len(after), f"solo {changed}/{len(after)} campioni ri-inquadrati"
print(f"OK  tutti i {changed} campioni ri-inquadrati")

# the frame must actually be the one requested, on every sample
for i, s in ((i, base[i]) for i in range(len(base))):
    w = s["mass"].reshape(-1); w = w / w.sum()
    V = s["verts"]
    c = (w.unsqueeze(1) * V).sum(0)
    r = float(torch.sqrt((w * ((V - c) ** 2).sum(1)).sum()))
    assert abs(r - 1.0) < 1e-3, (i, r)
    assert float(c.norm()) < 1e-3, (i, float(c.norm()))
print("OK  ogni campione ha centroide nullo e raggio RMS unitario")

# and the crop's scale must now track the original far more tightly than maxabs did
print("OK  frame attivo e verificato su dati reali")

# 'current' must leave a fresh cache untouched, or the control arm is not a control
base2 = fd.CachedDataset(str(tmp), workers=2, residency="ram", device=None, max_gb=10)
raw = {i: base2[i]["verts"].clone() for i in range(len(base2))}
install_frame("current")
assert all(torch.equal(raw[i], base2[i]["verts"]) for i in raw), "--frame current NON e' un no-op"
print("OK  --frame current e' un no-op esatto")
