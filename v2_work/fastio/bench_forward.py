#!/usr/bin/env python
"""ms/mesh for the sequential v1 path vs the batched path, at several group sizes.

    .conda_env/bin/python v2_work/fastio/bench_forward.py --device cpu
    .conda_env/bin/python v2_work/fastio/bench_forward.py --device cuda

Also holds the v1-model / v1-sample loaders shared with test_batched.py.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (
    REPO_ROOT / "face_embedding/gt_encdec/remeshing/intrinsic",
    REPO_ROOT / "face_embedding/gt_encdec/autoencoder",
    REPO_ROOT / "diffusion-net/src",
    Path(__file__).resolve().parent,
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from batched import embed_samples, size_groups  # noqa: E402

CKPT = REPO_ROOT / (
    "face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1/"
    "mixed_xtopo_rank0p5_id0p25_bs5_best/checkpoints/best_by_xtopo_mesh_clean.pth"
)
FLAME_DIR = REPO_ROOT / "v2_work/genflame/flame_train_ready/npz_withops"
TOPOLOGIES = ("original", "noisy", "remesh", "crop", "down8k", "up60k")


def build_v1_model(device, ckpt: Path = CKPT):
    """The released xyz_dn top model, eval mode, config taken from the checkpoint itself."""
    from robustness.model_helpers import build_model

    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    model = build_model(SimpleNamespace(**state["args"]), torch.device(device))
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def load_samples(n_per_topology: int = 3, topologies: Sequence[str] = TOPOLOGIES) -> List[Dict]:
    """Prepared FLAME samples, n_per_topology subjects for each topology."""
    from dataset_gtready import GTReadyDatasetNPZ

    ds = GTReadyDatasetNPZ(str(FLAME_DIR))
    by_name = {f: i for i, f in enumerate(ds.files)}
    subjects = sorted({f.split("_")[0] for f in ds.files})[:n_per_topology]
    out = []
    for topo in topologies:
        for sid in subjects:
            out.append(ds[by_name[f"{sid}_GTready_{topo}.npz"]])
    return out


def sequential(model, samples: Sequence[Dict], device, add_noise: bool = False) -> torch.Tensor:
    """The v1 one-mesh-per-call path, via the frozen forward_model()."""
    from robustness.data_utils import sample_to_device
    from robustness.model_helpers import forward_model

    zs = []
    for s in samples:
        sd = sample_to_device(s, device=torch.device(device))
        z, _ = forward_model(model=model, sample_dict=sd, V_in=sd["verts"],
                            return_gate_info=False, add_noise=add_noise)
        zs.append(z.squeeze(0))
    return torch.stack(zs)


def _n_ops(fn) -> int:
    """Dispatched aten ops: on a GPU each is one or more kernel launches, and launch
    overhead is what the batched path removes."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU]) as p:
        fn()
    return sum(e.count for e in p.key_averages() if e.self_cpu_time_total > 0)


def _timed(fn, device) -> float:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 5, 15, 40])
    ap.add_argument("--pad-slack", type=float, default=0.05)
    ap.add_argument("--ops", action="store_true",
                    help="also count dispatched ops (proxy for GPU kernel launches)")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = build_v1_model(device)
    # 8 subjects x 6 topologies = 48 samples, enough for the largest group size
    pool = load_samples(n_per_topology=8)

    print(f"device={device}  pad_slack={args.pad_slack}  latent={model.latent_dim}")
    print(f"{'group':>6} {'seq ms/mesh':>12} {'batch ms/mesh':>14} {'speedup':>8} {'calls':>6}"
          + ("  ops seq/batched" if args.ops else ""))
    with torch.no_grad():
        for n in args.sizes:
            group = [pool[i % len(pool)] for i in range(n)]
            embed_samples(model, group[:1], device)  # warm up kernels/allocator
            sequential(model, group[:1], device)
            t_seq = _timed(lambda: sequential(model, group, device), device)
            t_bat = _timed(lambda: embed_samples(model, group, device, pad_slack=args.pad_slack), device)
            calls = len(size_groups(group, args.pad_slack))
            tail = ""
            if args.ops:
                a = _n_ops(lambda: sequential(model, group, device))
                b = _n_ops(lambda: embed_samples(model, group, device, pad_slack=args.pad_slack))
                tail = f"  {a}/{b} = {a/b:.2f}x"
            print(f"{n:>6} {t_seq/n*1e3:>12.1f} {t_bat/n*1e3:>14.1f} {t_seq/t_bat:>7.2f}x {calls:>6}{tail}")


if __name__ == "__main__":
    main()
