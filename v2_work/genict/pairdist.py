"""All-pairs vertex-mean-L2 between meshes in dense correspondence, on torch.

The GT protocol of REMESH-2 is `mean_v ||V_i[v] - V_j[v]||` (BFM:
`autoencoder/latent_analysis/compute_gt_distance_matrix_normalized.py`; FLAME:
`v2_work/genflame/build_flame_gt_matrix.py`).  It does not factor through a
Gram matrix -- it is a mean of norms, not a norm of the mean -- so the cost is
genuinely O(n^2 * nv).  On ICT-5000 that is 25e6 pairs x 9409 vertices, which
the numpy double loop of the FLAME script would take hours to chew through;
here it is one batched torch expression, on GPU when there is one.

Precision: the meshes differ from each other by ~1e-2 on coordinates of ~1e1,
so a raw float32 subtraction throws away four digits of the difference.  The
mean identity is subtracted from every mesh first (exact: it cancels in
`V_i - V_j`), which brings the operands down to the scale of the differences
and puts float32 back at ~1e-8 absolute.
"""

from __future__ import annotations

import time

import numpy as np
import torch


def pick_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        spec = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(spec)


def vertex_mean_l2_matrix(V: np.ndarray, device: str = "auto", block: int = 0,
                          verbose: bool = True) -> np.ndarray:
    """(n, n) float64 matrix of mean per-vertex L2 distances. `V` is (n, nv, 3)."""
    dev = pick_device(device)
    if block <= 0:
        block = 4 if dev.type == "cuda" else 1

    V = np.asarray(V, dtype=np.float64)
    n = V.shape[0]
    t = torch.from_numpy((V - V.mean(axis=0, keepdims=True)).astype(np.float32)).to(dev)

    D = torch.zeros((n, n), dtype=torch.float32, device=dev)
    t0 = time.time()
    for i0 in range(0, n, block):
        a = t[i0:i0 + block]                                  # (b, nv, 3)
        diff = a.unsqueeze(1) - t.unsqueeze(0)                # (b, n, nv, 3)
        D[i0:i0 + block] = diff.pow(2).sum(-1).sqrt_().mean(-1)
        if verbose and (i0 // max(block, 1)) % 100 == 0:
            done = min(i0 + block, n)
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  rows {done}/{n} ({rate:.1f}/s, eta {(n - done) / max(rate, 1e-9) / 60:.1f} min)",
                  flush=True)

    D = D.double().cpu().numpy()
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    if verbose:
        print(f"  {n}x{n} in {(time.time() - t0) / 60:.2f} min on {dev}", flush=True)
    return D


def offdiag_stats(D: np.ndarray) -> dict:
    """min / 1st percentile / median / max over the strict upper triangle."""
    v = D[np.triu_indices(len(D), 1)]
    return {
        "min": float(v.min()),
        "p1": float(np.percentile(v, 1)),
        "median": float(np.median(v)),
        "max": float(v.max()),
    }
