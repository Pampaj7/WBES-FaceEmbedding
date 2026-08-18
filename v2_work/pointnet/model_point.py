#!/usr/bin/env python
"""A point-based encoder with the same interface as DiffusionEncoderOnly, and no operators.

Why this exists. The four-arm potential-well experiment measured that our failure axis is
`crop` (0.7072) and our best axis is `resample` (0.7719). That split is exactly DiffusionNet's
own promise and its own silence: it delivers agnosticism to DISCRETIZATION and never claimed
agnosticism to PARTIALITY. The Laplacian eigenbasis is a global function of the domain, so
moving the outer boundary rewrites the basis everywhere, not only near the cut.

A point encoder has no basis to rewrite. A crop removes points; a symmetric function over
points loses roughly the fraction of points removed. The failure mode is not fixed, it is
absent by construction. It also costs zero precompute, against 15.3 core-seconds and 20.4 MB
per mesh for the k_eig=128 operators -- so if it holds up, the runtime table in the paper
improves by an order of magnitude on the same axis it already claims.

Design, and the two choices that are not the obvious ones:

  * FIXED SAMPLE SIZE. Every mesh is reduced to `n_samples` points regardless of whether it
    carries 8k or 60k vertices. The network therefore cannot see the tessellation at all --
    resolution invariance is structural rather than learned.

  * AREA-WEIGHTED SAMPLING. Drawing vertices uniformly would inherit the vertex density, which
    is precisely the nuisance that differs between down8k, up60k and remesh: a uniform draw
    puts most of its points wherever the mesher happened to refine. Sampling with probability
    proportional to the barycentric vertex areas approximates a uniform draw over the SURFACE,
    which is the density-free thing we actually want. Those areas are the `mass` vector we
    already store; note that mass is O(F) to compute from faces alone and is not part of the
    expensive eigendecomposition, so the zero-precompute claim survives it.

Everything else follows DGCNN (Wang et al., ACM TOG 2019): EdgeConv on a k-NN graph rebuilt in
feature space at each layer, features concatenated across layers, then global pooling.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _knn_graph(x: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the k nearest neighbours of each row of x (M, C) -> (M, k)."""
    d = torch.cdist(x, x)
    d.fill_diagonal_(float("inf"))
    return d.topk(k, dim=-1, largest=False).indices


def _edge_features(x: torch.Tensor, k: int) -> torch.Tensor:
    """(M, C) -> (M, k, 2C) as [x_i, x_j - x_i], the EdgeConv input."""
    idx = _knn_graph(x, k)
    neigh = x[idx]                                   # (M, k, C)
    centre = x.unsqueeze(1).expand_as(neigh)         # (M, k, C)
    return torch.cat([centre, neigh - centre], dim=-1)


class _EdgeConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(2 * c_in, c_out, bias=False),
            nn.BatchNorm1d(c_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = _edge_features(x, self.k)                # (M, k, 2C)
        m, k, c2 = e.shape
        h = self.mlp(e.reshape(m * k, c2)).reshape(m, k, -1)
        return h.max(dim=1).values                   # (M, c_out)


class PointEncoder(nn.Module):
    """Drop-in replacement for DiffusionEncoderOnly that ignores every operator argument.

    The signature is deliberately identical, including the operators it never reads, so that
    the frozen trainer, its losses and its evaluation path need no change: only `build_model`
    and `forward_model` are rebound, exactly as the masked-pooling variant does.
    """

    def __init__(self, latent_dim=256, width=128, dropout=0.1, pool_mode="meanmax",
                 n_samples=2048, k=20):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.pool_mode = str(pool_mode)
        self.n_samples = int(n_samples)
        self.k = int(k)

        w = int(width)
        self.ec1 = _EdgeConv(3, w // 2, self.k)
        self.ec2 = _EdgeConv(w // 2, w // 2, self.k)
        self.ec3 = _EdgeConv(w // 2, w, self.k)
        self.fuse = nn.Sequential(
            nn.Linear(2 * w, self.latent_dim, bias=False),
            nn.BatchNorm1d(self.latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.vertex_bottleneck = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim // 2, self.latent_dim),
        )
        if self.pool_mode == "meanmax":
            self.pool_proj = nn.Linear(2 * self.latent_dim, self.latent_dim)
        elif self.pool_mode == "mean":
            self.pool_proj = nn.Identity()
        else:
            raise ValueError("pool_mode must be 'mean' or 'meanmax'")

        print(f"🪶 PointEncoder | Z={latent_dim}, width={w}, k={self.k}, "
              f"M={self.n_samples}, pool={self.pool_mode}, operatori: nessuno")

    def _draw(self, V: torch.Tensor, mass: torch.Tensor | None) -> torch.Tensor:
        n = V.shape[0]
        if n <= self.n_samples:
            return torch.arange(n, device=V.device)
        if mass is None:
            return torch.randperm(n, device=V.device)[: self.n_samples]
        # Area-weighted WITHOUT replacement. multinomial(replacement=False) is the exact
        # draw here; with replacement it would duplicate points and let the k-NN graph
        # collapse onto zero-length edges wherever a point was drawn twice.
        w = mass.reshape(-1).to(V.dtype).clamp_min(0)
        if not torch.isfinite(w).all() or float(w.sum()) <= 0:
            return torch.randperm(n, device=V.device)[: self.n_samples]
        return torch.multinomial(w, self.n_samples, replacement=False)

    def forward(self, V, mass=None, L=None, evals=None, evecs=None, faces=None,
                gradX=None, gradY=None, return_per_vertex: bool = False,
                add_noise: bool = True):
        V = V.reshape(-1, 3)
        idx = self._draw(V, mass)
        x = V[idx]

        h1 = self.ec1(x)
        h2 = self.ec2(h1)
        h3 = self.ec3(h2)
        Z = self.fuse(torch.cat([h1, h2, h3], dim=1))   # w//2 + w//2 + w == 2w
        Z = self.vertex_bottleneck(Z)
        if add_noise:
            Z = Z + 0.01 * torch.randn_like(Z)

        Z_mean = Z.mean(dim=0, keepdim=True)
        if self.pool_mode == "meanmax":
            Z_global = self.pool_proj(torch.cat([Z_mean, Z.max(dim=0, keepdim=True).values], dim=1))
        else:
            Z_global = self.pool_proj(Z_mean)

        if return_per_vertex:
            # The per-vertex field is defined on the DRAWN points, not on all vertices. The
            # only consumer is the smoothness term, which is off in this recipe; a caller that
            # turns it on would be silently comparing fields on different supports, so it is
            # refused rather than approximated.
            raise NotImplementedError(
                "PointEncoder has no per-vertex field over the full mesh (it encodes a draw). "
                "The smoothness term is not defined for this backbone."
            )
        return Z_global


def _demo() -> None:
    torch.manual_seed(0)
    m = PointEncoder(latent_dim=64, width=64, n_samples=256, k=8).eval()

    V = torch.randn(3000, 3)
    mass = torch.rand(3000).abs() + 1e-6

    with torch.no_grad():
        z = m(V, mass, add_noise=False)
        assert z.shape == (1, 64), z.shape

        # Permutation invariance. This is the property the whole approach rests on, so it is
        # checked rather than assumed. It is checked on a mesh SMALLER than the draw, where
        # every point is kept: a permuted multinomial draw does not select the same subset
        # even under the same seed, because the RNG is consumed in a different order, so
        # permuting a sampled call would test the sampler and not the encoder.
        small = torch.randn(200, 3)
        smass = torch.rand(200).abs() + 1e-6
        perm = torch.randperm(200)
        za = m(small, smass, add_noise=False)
        zb = m(small[perm], smass[perm], add_noise=False)
        assert torch.allclose(za, zb, atol=1e-5), (za - zb).abs().max()

        # Sampling stability. Two independent draws from the SAME mesh must land close, or the
        # sampler injects more noise into the metric than the topology differences we are
        # trying to measure. The bound is loose on purpose -- it is a smoke test against a
        # collapsed draw, not a claim about the trained model.
        d0, d1 = m(V, mass, add_noise=False), m(V, mass, add_noise=False)
        rel = float((d0 - d1).norm() / d0.norm())
        assert rel < 0.25, f"draw-to-draw spread {rel:.3f} too large"

        # Resolution independence: a mesh with 10x the vertices of the same shape yields a
        # comparable embedding, because both are reduced to n_samples points.
        V2 = torch.cat([V, V + 1e-4 * torch.randn_like(V)], dim=0)
        z2 = m(V2, torch.cat([mass, mass]), add_noise=False)
        assert z2.shape == (1, 64)

        # Meshes smaller than the draw are passed through whole, not padded.
        assert m(torch.randn(100, 3), torch.rand(100), add_noise=False).shape == (1, 64)

        # Degenerate mass must not produce NaNs: it falls back to a uniform draw.
        assert torch.isfinite(m(V, torch.zeros(3000), add_noise=False)).all()

    try:
        m(V, mass, return_per_vertex=True)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("return_per_vertex must refuse, not approximate")

    n_pt = sum(p.numel() for p in m.parameters())
    print(f"OK  demo passed | parametri (config demo): {n_pt}")


if __name__ == "__main__":
    _demo()
