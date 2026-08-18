#!/usr/bin/env python
"""The number missing from the runtime comparison: DiffusionNet's own forward.

Yesterday's measurement put the operators at 6.10 s per mesh against 0.096 s for the point
encoder's forward, and called it 63x. That factor compares PRECOMPUTE against FORWARD, which is
not a comparison between two pipelines -- DiffusionNet also pays a forward, and it had never
been measured. Nothing goes into the paper's runtime table until it is.

Both encoders at the shipped configuration (latent 256, width 128), same six topologies of the
same identity, same machine, median of three.
"""
import os, sys, time, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
for sub in ("diffusion-net/src", "face_embedding/gt_encdec/autoencoder", "v2_work/pointnet"):
    sys.path.insert(0, str(ROOT / sub))

import numpy as np, torch
from diffusion_autoencoder import DiffusionEncoderOnly
from model_point import PointEncoder

REPS = 3
D = ROOT / "datasets/REMESH/npz_data_topo_500_withops"

dn = DiffusionEncoderOnly(256, 128, 4, 0.1, "meanmax").eval()
pt = PointEncoder(latent_dim=256, width=128, n_samples=2048, k=20).eval()
print(f"\ncores: {len(os.sched_getaffinity(0))}, threads torch: {torch.get_num_threads()}")
print(f"{'topologia':10s} {'vert':>7s} {'DiffusionNet':>13s} {'PointEncoder':>13s}")


def sparse(z, key, n):
    i = torch.tensor(z[f"{key}_indices"], dtype=torch.long)
    v = torch.tensor(z[f"{key}_values"], dtype=torch.float32)
    shp = tuple(int(x) for x in z[f"{key}_shape"])
    return torch.sparse_coo_tensor(i, v, shp).coalesce()


rows = []
for f in sorted(D.glob("id0000_*.npz")):
    topo = f.stem.split("_")[-1]
    z = np.load(f)
    V = torch.tensor(z["verts"], dtype=torch.float32); V = (V - V.mean(0)) / V.abs().max()
    F = torch.tensor(z["faces"], dtype=torch.long)
    mass = torch.tensor(z["mass"], dtype=torch.float32).ravel()
    evals = torch.tensor(z["evals"], dtype=torch.float32)
    evecs = torch.tensor(z["evecs"], dtype=torch.float32)
    L, gX, gY = (sparse(z, k, V.shape[0]) for k in ("L", "gradX", "gradY"))
    with torch.no_grad():
        ts = []
        for _ in range(REPS):
            t = time.perf_counter()
            dn(V, mass, L, evals, evecs, F, gX, gY, add_noise=False)
            ts.append(time.perf_counter() - t)
        a = statistics.median(ts)
        ts = []
        for _ in range(REPS):
            t = time.perf_counter(); pt(V, mass, add_noise=False); ts.append(time.perf_counter() - t)
        b = statistics.median(ts)
    rows.append((topo, V.shape[0], a, b))
    print(f"{topo:10s} {V.shape[0]:7d} {a:12.3f}s {b:12.3f}s", flush=True)

mdn = sum(r[2] for r in rows) / len(rows)
mpt = sum(r[3] for r in rows) / len(rows)
print(f"\nmedia forward: DiffusionNet {mdn:.3f}s, PointEncoder {mpt:.3f}s")
print(f"pipeline per mesh (operatori 6.10s + forward):")
print(f"  DiffusionNet  {6.10 + mdn:.2f}s")
print(f"  PointEncoder  {mpt:.3f}s")
print(f"  rapporto      {(6.10 + mdn) / mpt:.0f}x")
