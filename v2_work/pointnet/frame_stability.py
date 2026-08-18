#!/usr/bin/env python
"""Which canonical frame survives a crop?

pot_point collapsed on `crop` (0.3155 against 0.7072) and the suspected cause is the loader's
frame: centre = mean over VERTICES, scale = maxabs = the single farthest vertex. Both are
things a crop removes. This measures the frame parameters directly -- it is not a proxy for
the trained metric (one of those already deceived us: the alpha sweep moved the right way and
the trained model went the other way). It answers only the narrow question it can answer: does
the frame move when the boundary moves.

Three candidates, each a (centre, scale) pair:
  current  vertex mean            maxabs radius          what the loader does today
  area     mass-weighted centroid sqrt(total area)       Weyl-consistent, matches bfm_areanorm
  rms      mass-weighted centroid mass-weighted RMS radius   smooth, no dependence on extremes

A frame is good here if, going from `original` to another topology of the SAME identity, the
centre barely moves and the scale ratio stays near 1. Both are reported relative to the
original's own scale, so they are comparable across identities.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "datasets/REMESH/npz_data_topo_500_withops"
TOPOS = ["crop", "noisy", "down8k", "up60k", "remesh"]
N_ID = 40


def frames(V, F, mass):
    out = {}
    c = V.mean(0)
    out["current"] = (c, float(np.abs(V - c).max()))
    w = mass / mass.sum()
    cm = (w[:, None] * V).sum(0)
    tri = V[F]
    A = float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())
    out["area"] = (cm, float(np.sqrt(A)))
    rms = float(np.sqrt((w * ((V - cm) ** 2).sum(1)).sum()))
    out["rms"] = (cm, rms)
    return out


def load(p):
    z = np.load(p)
    m = np.asarray(z["mass"]).ravel().astype(np.float64)
    return z["verts"].astype(np.float64), z["faces"], m


ids = sorted({p.name.split("_")[0] for p in D.glob("*_original.npz")})[:N_ID]
acc = {k: {t: {"shift": [], "ratio": []} for t in TOPOS} for k in ("current", "area", "rms")}
used = 0
for i in ids:
    po = D / f"{i}_GTready_original.npz"
    if not po.exists():
        continue
    fo = frames(*load(po))
    ok = False
    for t in TOPOS:
        pt = D / f"{i}_GTready_{t}.npz"
        if not pt.exists():
            continue
        ft = frames(*load(pt))
        for k in acc:
            c0, s0 = fo[k]
            c1, s1 = ft[k]
            acc[k][t]["shift"].append(float(np.linalg.norm(c1 - c0) / s0))
            acc[k][t]["ratio"].append(float(s1 / s0))
        ok = True
    used += ok

print(f"{used} identita'\n")
print(f"{'frame':8s} {'topologia':10s} {'shift centro':>13s} {'scala s1/s0':>13s}")
for k in ("current", "area", "rms"):
    for t in TOPOS:
        sh = np.median(acc[k][t]["shift"])
        ra = np.array(acc[k][t]["ratio"])
        print(f"{k:8s} {t:10s} {sh:13.4f} {np.median(ra):8.4f} +-{ra.std():.4f}")
    print()
