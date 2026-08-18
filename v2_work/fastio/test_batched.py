#!/usr/bin/env python
"""Equivalence check: batched embeddings == the v1 one-mesh-per-call embeddings.

    .conda_env/bin/python v2_work/fastio/test_batched.py [--device cpu]

Asserts on the released xyz_dn checkpoint with add_noise=False (deterministic), over
18 FLAME samples spanning all 6 topologies, for the default mixed grouping, for
exact-size-only groups, and for one single padded group holding all 6 vertex counts.
"""
from __future__ import annotations

import argparse
import time

import torch

from batched import embed_samples, size_groups
from bench_forward import TOPOLOGIES, build_v1_model, load_samples, sequential

TOL = 1e-4  # max abs difference on latents whose scale is ~1


def _report(tag, ref, got):
    d = (got - ref).abs()
    # relative to the scale of the latent, not to individual near-zero components
    rel = (d.max() / ref.abs().max()).item()
    print(f"  {tag:<28} max_abs={d.max().item():.2e}  rel_to_scale={rel:.2e}")
    assert d.max().item() < TOL, f"{tag}: batched != sequential (max abs {d.max().item():.3e})"
    return d.max().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)

    model = build_v1_model(device)
    samples = load_samples(n_per_topology=3)  # 3 subjects x 6 topologies = 18
    counts = sorted({int(s["verts"].shape[0]) for s in samples})
    assert len(samples) == 18 and len(counts) >= 6, (len(samples), counts)
    print(f"device={device}  {len(samples)} samples, vertex counts {counts}")

    with torch.no_grad():
        t0 = time.perf_counter()
        ref = sequential(model, samples, device)
        t_seq = time.perf_counter() - t0

        t0 = time.perf_counter()
        z_default = embed_samples(model, samples, device)
        t_bat = time.perf_counter() - t0

        z_exact = embed_samples(model, samples, device, pad_slack=0.0)
        z_one = embed_samples(model, samples, device, pad_slack=float("inf"))

    assert z_default.shape == (18, model.latent_dim), z_default.shape

    # exact-size groups: no padding at all
    g_exact = size_groups(samples, 0.0)
    assert all(len({int(samples[i]["verts"].shape[0]) for i in g}) == 1 for g in g_exact)
    # one group holding every distinct vertex count -> the padded path is exercised
    g_one = size_groups(samples, float("inf"))
    assert len(g_one) == 1 and len({int(samples[i]["verts"].shape[0]) for i in g_one[0]}) == len(counts)

    print(f"groups: default={len(size_groups(samples, 0.05))}  exact={len(g_exact)}  single={len(g_one)}")
    _report("default grouping", ref, z_default)
    _report("exact-size groups (no pad)", ref, z_exact)
    _report("single padded group", ref, z_one)

    # padding must not shift a mesh's own embedding depending on who it shares a batch with
    _report("padded vs unpadded batched", z_exact, z_one)

    # guard against passing for the wrong reason: without the pooling mask, the same
    # padded batch must be visibly wrong, i.e. this test can actually see a mask bug
    from batched import _collate, _pool
    with torch.no_grad():
        b = _collate([samples[i] for i in g_one[0]], device)
        x = model.vertex_bottleneck(model.encoder(
            b["verts"], b["mass"], None, b["evals"], b["evecs"], gradX=b["gradX"], gradY=b["gradY"]))
        z_unmasked = _pool(model, x, torch.ones_like(b["mask"]))
    err = (z_unmasked - z_one[torch.tensor(g_one[0], device=device)]).abs().max().item()
    print(f"  {'unmasked pooling error':<28} max_abs={err:.2e}  (must exceed tol)")
    assert err > TOL, "padded rows are not affecting unmasked pooling: the test is vacuous"

    print(f"wall clock ({device}, {len(samples)} meshes): sequential {t_seq*1e3:.0f} ms "
          f"({t_seq/len(samples)*1e3:.1f} ms/mesh) | batched {t_bat*1e3:.0f} ms "
          f"({t_bat/len(samples)*1e3:.1f} ms/mesh) | {t_seq/t_bat:.2f}x")
    assert len(TOPOLOGIES) == 6
    print("OK")


if __name__ == "__main__":
    main()
