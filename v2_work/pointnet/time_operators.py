#!/usr/bin/env python
"""Cost of the DiffusionNet operators, measured where the claim will be defended.

The paper's runtime section states 1.2-2.15 s for 16.5k-23.5k vertex meshes and 6.3 s for the
60.4k-vertex ones. A first attempt at reproducing it read 15.3 s at k_eig=128 -- but that was
taken on the login node, whose cgroup grants a single core and which we were contending with.
Timings taken on a machine we are ourselves saturating already cost us one wrong scope
decision (a 100x overestimate of the Poisson cost). So this runs on an allocated node, pinned
to its own cores, and reports the median of repeated trials rather than one draw.

It also times the point encoder's forward on the same meshes, because the comparison that
matters for the runtime table is not "how slow are operators" but "what does the alternative
pay instead".
"""
import os, sys, time, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "diffusion-net/src"))
sys.path.insert(0, str(ROOT / "v2_work/pointnet"))

import numpy as np, torch
from diffusion_net.geometry import compute_operators
from model_point import PointEncoder

REPS = 3
SRC = ROOT / "datasets/REMESH/npz_data_topo_500_withops"
picks = {}
for f in sorted(SRC.glob("id0000_*.npz")):
    picks[f.stem.split("_")[-1]] = f

print(f"cores visibili: {len(os.sched_getaffinity(0))}, threads torch: {torch.get_num_threads()}")
print(f"{'topologia':10s} {'vert':>7s} {'k=128':>9s} {'k=300':>9s} {'point fwd':>10s}")

enc = PointEncoder(latent_dim=256, width=128, n_samples=2048, k=20).eval()
rows = []
for topo, f in picks.items():
    z = np.load(f)
    V = torch.tensor(z["verts"], dtype=torch.float32)
    F = torch.tensor(z["faces"], dtype=torch.int32)
    mass = torch.tensor(z["mass"], dtype=torch.float32)
    out = []
    for k in (128, 300):
        ts = []
        for _ in range(REPS):
            t = time.perf_counter(); compute_operators(V, F, k_eig=k); ts.append(time.perf_counter() - t)
        out.append(statistics.median(ts))
    ts = []
    with torch.no_grad():
        for _ in range(REPS):
            t = time.perf_counter(); enc(V, mass, add_noise=False); ts.append(time.perf_counter() - t)
    pf = statistics.median(ts)
    rows.append((topo, V.shape[0], out[0], out[1], pf))
    print(f"{topo:10s} {V.shape[0]:7d} {out[0]:8.2f}s {out[1]:8.2f}s {pf:9.3f}s", flush=True)

tot128 = sum(r[2] for r in rows) / len(rows)
totpf = sum(r[4] for r in rows) / len(rows)
print(f"\nmedia per mesh: operatori k=128 {tot128:.2f}s, point forward {totpf:.3f}s "
      f"({tot128/totpf:.0f}x)")
print(f"su 3000 mesh: operatori {tot128*3000/3600:.1f} h-core, point {totpf*3000/60:.1f} min-core")
