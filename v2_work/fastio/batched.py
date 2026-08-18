"""Batched embedding for the v1 DiffusionNet encoders: one forward call per size group.

The v1 trainer embeds one mesh per forward call (30 calls per optimizer step at
batch_subjects=5). With 1930-vertex FLAME meshes the per-call Python/kernel-launch
overhead dominates: the GPU is idle. This module runs a whole group of meshes as a
single [B,N,C] DiffusionNet call.

Shape contract established for `DiffusionNet.forward` with a leading batch dim B
(diffusion-net/src/diffusion_net/layers.py, docstring ~line 322):

    x_in    [B, N, C_in]        dense
    mass    [B, N]              dense
    L       [B, N, N] sparse    UNUSED by diffusion_method='spectral' -> we pass None
    evals   [B, K]              dense
    evecs   [B, N, K]           dense
    gradX   [B, N, N] sparse    only ever indexed as gradX[b,...] -> see _SparseBatch
    gradY   [B, N, N] sparse    idem
    edges   [B, ...]            unused for outputs_at='vertices' -> None
    faces   [B, ...]            unused for outputs_at='vertices' -> None
    returns [B, N, C_out]

Padding correctness (why a padded vertex cannot perturb a real one):
  * spectral diffusion is `evecs @ diag(coef) @ evecs^T @ diag(mass) @ x`. Padded
    rows of `evecs` and `mass` are zero, so padded values of `x` contribute nothing
    to the spectral coefficients, and padded rows of the reconstruction are exactly 0.
  * gradX/gradY are declared [N_pad,N_pad] but keep only the original mesh's nnz,
    so no padded row or column ever touches a real index.
  * so real-vertex features are bit-comparable to the unpadded run; only the pooling
    has to exclude the padded rows, which `_pool` does (mask for the mean, -inf for
    the max).
  * K (128 here) is identical across samples, so no eigen-padding is involved.

Consequence for speed: `DiffusionNetBlock.forward` loops over B for the two sparse
mm's (layers.py ~line 216, torch.mm does not batch), so the gradient term stays
serial. Only the spectral transforms and the MLPs parallelize.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch

Sample = Dict[str, object]


def size_groups(samples: Sequence[Sample], pad_slack: float = 0.05) -> List[List[int]]:
    """Group sample indices so that within a group max(N) <= (1+pad_slack)*min(N).

    pad_slack=0 gives exact-size groups (zero padding); float('inf') gives a single
    padded group. The default 5% keeps each FLAME topology in one group (crop
    1873-1886, remesh 1363-1364, down8k 678-679) while never mixing down8k with up60k.
    """
    order = sorted(range(len(samples)), key=lambda i: int(samples[i]["verts"].shape[0]))
    groups: List[List[int]] = []
    cur: List[int] = []
    for i in order:
        n = int(samples[i]["verts"].shape[0])
        if cur and n > (1.0 + pad_slack) * int(samples[cur[0]]["verts"].shape[0]):
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    return groups


class _SparseBatch:
    """The [B,N,N] sparse operand, as the only thing DiffusionNet actually asks of it.

    `DiffusionNetBlock.forward` uses gradX/gradY exclusively as `gradX[b,...]`
    (layers.py ~line 216) because torch.mm does not batch. A genuine [B,N,N] sparse
    COO tensor works, but `select` on it scans all B*nnz entries, so the whole block
    costs O(B^2 * nnz) -- measured 1.4x SLOWER than the sequential path at B=15.
    This serves the same per-sample matrices in O(1) instead. Padding is expressed by
    re-declaring each matrix as [n_pad,n_pad]: no entry is added, so no padded row or
    column is ever touched.
    """

    __slots__ = ("mats",)

    def __init__(self, mats: Sequence[torch.Tensor], n_pad: int, device: torch.device) -> None:
        self.mats = []
        for m in mats:
            m = m.to(device).coalesce()
            if m.shape[-1] != n_pad:
                m = torch.sparse_coo_tensor(m.indices(), m.values(), (n_pad, n_pad)).coalesce()
            self.mats.append(m)

    def __getitem__(self, key) -> torch.Tensor:
        return self.mats[key[0] if isinstance(key, tuple) else key]


def _collate(samples: Sequence[Sample], device: torch.device) -> Dict[str, object]:
    ns = [int(s["verts"].shape[0]) for s in samples]
    B, N = len(samples), max(ns)
    K = int(samples[0]["evals"].numel())

    out = {
        "verts": torch.zeros(B, N, 3, device=device),
        "mass": torch.zeros(B, N, device=device),
        "evecs": torch.zeros(B, N, K, device=device),
        "evals": torch.zeros(B, K, device=device),
        "mask": torch.zeros(B, N, dtype=torch.bool, device=device),
    }
    for b, (s, n) in enumerate(zip(samples, ns)):
        evecs = s["evecs"].to(device)
        evals = s["evals"].to(device).flatten()
        if evecs.shape[-1] != K or evals.numel() != K:
            raise ValueError(f"sample {s.get('name', b)} has K={evals.numel()}/{evecs.shape[-1]}, expected {K}")
        out["verts"][b, :n] = s["verts"].to(device)
        out["mass"][b, :n] = s["mass"].to(device)
        out["evecs"][b, :n] = evecs
        out["evals"][b] = evals
        out["mask"][b, :n] = True
    out["gradX"] = _SparseBatch([s["gradX"] for s in samples], N, device)
    out["gradY"] = _SparseBatch([s["gradY"] for s in samples], N, device)
    return out


def _pool(model: torch.nn.Module, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean / mean+max pooling over vertices, matching DiffusionEncoderOnly."""
    m = mask.unsqueeze(-1)
    z_mean = (x * m).sum(dim=-2) / mask.sum(dim=-1, keepdim=True)
    if model.pool_mode == "meanmax":
        z_max = x.masked_fill(~m, float("-inf")).max(dim=-2).values
        return model.pool_proj(torch.cat([z_mean, z_max], dim=-1))
    return model.pool_proj(z_mean)


def _embed_group(model, samples: Sequence[Sample], device: torch.device, add_noise: bool) -> torch.Tensor:
    b = _collate(samples, device)
    x = model.encoder(
        b["verts"], b["mass"], None, b["evals"], b["evecs"],
        gradX=b["gradX"], gradY=b["gradY"],
    )
    x = model.vertex_bottleneck(x)
    if add_noise:
        x = x + 0.01 * torch.randn_like(x)
    return _pool(model, x, b["mask"])


def embed_samples(model, samples: Sequence[Sample], device, add_noise: bool = False,
                  pad_slack: float = 0.05) -> torch.Tensor:
    """Embed prepared samples in as few forward calls as size grouping allows.

    `samples` are the dicts GTReadyDatasetNPZ/CachedDataset return (verts, mass,
    evals, evecs, faces, gradX, gradY, L, name). To embed perturbed vertices, pass
    `{**sample, "verts": V_in}` -- nothing else in the sample depends on V.

    Returns [len(samples), latent_dim] in the input order.
    """
    for attr in ("encoder", "vertex_bottleneck", "pool_mode", "pool_proj"):
        if not hasattr(model, attr):
            raise TypeError(f"{type(model).__name__} is not a DiffusionEncoderOnly-style model (no .{attr})")
    if model.encoder.diffusion_method != "spectral":
        raise ValueError("batched path passes L=None, which is only valid for diffusion_method='spectral'")
    if model.encoder.outputs_at != "vertices":
        raise ValueError(f"batched path expects outputs_at='vertices', got {model.encoder.outputs_at!r}")

    device = torch.device(device)
    out = torch.empty(len(samples), int(model.latent_dim), device=device)
    for g in size_groups(samples, pad_slack):
        out[torch.tensor(g, device=device)] = _embed_group(model, [samples[i] for i in g], device, add_noise)
    return out
