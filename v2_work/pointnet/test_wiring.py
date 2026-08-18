#!/usr/bin/env python
"""Proves the point backbone is ACTIVE, on real data, at every call site.

Written because the night's four bugs all shared one shape: a variant that changes neither
tensor shapes nor parameter count raises nothing when it is silently inactive. A checkpoint
from an unmasked model loaded into a masked one without error; a training ran on 2792 of 3000
meshes without error. Absence of an error proves nothing here, so the wiring is asserted.
"""
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("face_embedding/gt_encdec/remeshing/intrinsic",
            "face_embedding/gt_encdec/autoencoder", "diffusion-net/src",
            "v2_work/fastio", "v2_work/pointnet"):
    sys.path.insert(0, str(ROOT / sub))

import numpy as np, torch
import robustness.train_runner, robustness.eval_utils, robustness.model_helpers as mh
from train_fast import install_point_backbone
from model_point import PointEncoder

install_point_backbone(2048, 20)

# 1. every module that copied the names at import time must now hold the patched ones
for mod in (mh, robustness.train_runner, robustness.eval_utils):
    for name in ("build_model", "forward_model"):
        if hasattr(mod, name):
            fn = getattr(mod, name)
            assert fn.__module__ == "train_fast", f"{mod.__name__}.{name} NOT patched ({fn.__module__})"
print("OK  build_model/forward_model patched at every call site")


class A:  # the trainer's args, only the fields the builder reads
    latent_dim, width, n_blocks, dropout, pool_mode, model = 256, 128, 4, 0.1, "meanmax", "xyz_dn"


dev = torch.device("cpu")
m = mh.build_model(A(), dev)
assert isinstance(m, PointEncoder), type(m)
n_point = sum(p.numel() for p in m.parameters())

# 2. capacity comparison against the arm we will compare against, printed not guessed
from diffusion_autoencoder import DiffusionEncoderOnly
n_dn = sum(p.numel() for p in DiffusionEncoderOnly(256, 128, 4, 0.1, "meanmax").parameters())
print(f"OK  parametri: PointEncoder {n_point} vs xyz_dn {n_dn}  (rapporto {n_point/n_dn:.2f})")

# 3. a real mesh, through the real forward path, on every topology present
d = ROOT / "v2_work/potential/bfm_areanorm"
files = {p.name.split("_")[-1][:-4]: p for p in sorted(d.glob("id0000_*.npz"))}
assert files, f"no operators under {d}"
m.eval()
zs = {}
for topo, f in files.items():
    npz = np.load(f)
    V = torch.tensor(npz["verts"], dtype=torch.float32)
    V = V - V.mean(0)
    sample = {"mass": torch.tensor(npz["mass"], dtype=torch.float32)}
    t = time.time()
    with torch.no_grad():
        z, gate = mh.forward_model(m, sample, V, False, False)
    zs[topo] = z
    assert z.shape == (1, 256) and torch.isfinite(z).all()
    print(f"    {topo:9s} {npz['verts'].shape[0]:6d} vert -> z{tuple(z.shape)}  {time.time()-t:.2f}s")

# 4. the embedding must actually depend on the geometry, not be a constant the pooling emits
import itertools
spread = max(float((zs[a] - zs[b]).norm()) for a, b in itertools.combinations(zs, 2))
assert spread > 1e-3, f"embeddings collapsed across topologies (spread {spread})"

# 5. the operators must be genuinely unused: same mesh, deliberately corrupted spectrum
npz = np.load(next(iter(files.values())))
V = torch.tensor(npz["verts"], dtype=torch.float32); V = V - V.mean(0)
good = {"mass": torch.tensor(npz["mass"], dtype=torch.float32),
        "evals": torch.rand(128), "evecs": torch.rand(V.shape[0], 128)}
bad = dict(good, evals=torch.full((128,), float("nan")), evecs=torch.zeros(V.shape[0], 128))
torch.manual_seed(7); za, _ = mh.forward_model(m, good, V, False, False)
torch.manual_seed(7); zb, _ = mh.forward_model(m, bad, V, False, False)
assert torch.allclose(za, zb), "backbone reacted to the eigensystem: it is NOT operator-free"
print("OK  NaN eigensystem changes nothing: zero dipendenza dagli operatori")
print("TUTTI I CONTROLLI PASSATI")
