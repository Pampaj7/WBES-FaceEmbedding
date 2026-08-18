#!/usr/bin/env python
"""Does the operator cost fall with more threads? Decides whether the paper's printed
1.2-2.15 s can be reconciled with the 3.7-5.7 s measured at 4 threads, before anyone claims
the paper is wrong. Thread count is set from argv before torch/scipy are imported."""
import os, sys
n = sys.argv[1]
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = n
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "diffusion-net/src"))
import time, statistics, numpy as np, torch
torch.set_num_threads(int(n))
from diffusion_net.geometry import compute_operators
z = np.load(ROOT / "datasets/REMESH/npz_data_topo_500_withops/id0000_GTready_original.npz")
V = torch.tensor(z["verts"], dtype=torch.float32); F = torch.tensor(z["faces"], dtype=torch.int32)
ts = []
for _ in range(3):
    t = time.perf_counter(); compute_operators(V, F, k_eig=128); ts.append(time.perf_counter() - t)
print(f"threads={n:>3s}  {V.shape[0]} vert  k=128  mediana {statistics.median(ts):.2f}s  "
      f"(min {min(ts):.2f}, max {max(ts):.2f})", flush=True)
