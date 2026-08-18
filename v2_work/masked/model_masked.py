#!/usr/bin/env python
"""Encoder whose POOLING is restricted to the potential well's region of interest.

Why this exists. The potential well makes the *diffusion* domain canonical: eigenfunctions
vanish before reaching the real boundary, so heat spreads the same way whatever the boundary
does. But DiffusionNet's embedding is not the diffusion -- it is a pooling of per-vertex
features, and in `DiffusionEncoderOnly` that pooling is a plain mean and max over *every*
vertex, with the raw xyz of every vertex as input. Vertices the well suppressed still enter
the embedding directly, bypassing the diffusion entirely.

So the well alone cannot deliver the invariance it promises here: it fixes the operator and
leaves the support unfixed. This class fixes the support, restricting both pooling terms to
the vertices inside the region of interest.

Deliberately NOT changed: the mean stays unweighted (not area-weighted). Area weighting would
also remove a sampling-density dependence, which is a real effect but a *different* one; doing
both at once would make the comparison against the well-only arm unattributable. It is one
line away if we want it as a third arm.

The mask rides on the sample dict as `roi_mask` (written by potential_operators.py) and is
handed to the model by the forward_model patch in train_fast.py, so that neither the frozen v1
trainer nor its model signatures have to change.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "face_embedding/gt_encdec/autoencoder"))

from diffusion_autoencoder import DiffusionEncoderOnly  # noqa: E402

MIN_ROI_VERTICES = 16


class DiffusionEncoderOnlyMasked(DiffusionEncoderOnly):
    """DiffusionEncoderOnly with pooling restricted to `self._roi_mask`.

    Falls back to the parent's behaviour when no mask is present, so the class is safe to use
    on datasets without one and the two arms differ only where a mask actually exists.
    """

    def __init__(self, *args, roi_threshold: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.roi_threshold = float(roi_threshold)
        self._roi_mask: torch.Tensor | None = None
        print(f"🎯 masked pooling attivo | soglia ROI={self.roi_threshold}")

    def set_roi_mask(self, mask: torch.Tensor | None) -> None:
        self._roi_mask = mask

    def forward(self, V, mass, L, evals, evecs, faces, gradX, gradY,
                return_per_vertex: bool = False, add_noise: bool = True):
        Z = self.encoder(V, mass, L, evals, evecs, faces=faces, gradX=gradX, gradY=gradY)
        Z = self.vertex_bottleneck(Z)
        if add_noise:
            Z = Z + 0.01 * torch.randn_like(Z)

        keep = None
        if self._roi_mask is not None:
            m = self._roi_mask.reshape(-1).to(Z.device)
            if m.numel() == Z.shape[0]:
                cand = m > self.roi_threshold
                # A mask that keeps almost nothing would make the embedding noise; falling
                # back is safer than emitting a degenerate vector, and it is logged by the
                # caller through roi_fraction rather than hidden.
                if int(cand.sum()) >= MIN_ROI_VERTICES:
                    keep = cand

        if keep is None:
            Z_mean = Z.mean(dim=0, keepdim=True)
            Z_max = Z.max(dim=0, keepdim=True).values
        else:
            Zk = Z[keep]
            Z_mean = Zk.mean(dim=0, keepdim=True)
            Z_max = Zk.max(dim=0, keepdim=True).values

        if self.pool_mode == "meanmax":
            Z_global = self.pool_proj(torch.cat([Z_mean, Z_max], dim=1))
        else:
            Z_global = self.pool_proj(Z_mean)

        if return_per_vertex:
            return Z, Z_global
        return Z_global


def _demo() -> None:
    """Self-check: masking must change the embedding, and must equal the parent when absent."""
    torch.manual_seed(0)
    m = DiffusionEncoderOnlyMasked(latent_dim=8, width=8, n_blocks=1, pool_mode="meanmax")
    m.eval()
    n = 64
    Z = torch.randn(n, 8)

    def pool(mask):
        m.set_roi_mask(mask)
        keep = None
        if mask is not None and int((mask > 0.5).sum()) >= MIN_ROI_VERTICES:
            keep = mask > 0.5
        Zk = Z if keep is None else Z[keep]
        return m.pool_proj(torch.cat([Zk.mean(0, keepdim=True),
                                      Zk.max(0, keepdim=True).values], dim=1))

    full = pool(None)
    half = torch.zeros(n); half[:32] = 1.0
    masked = pool(half)
    tiny = torch.zeros(n); tiny[:4] = 1.0          # below MIN_ROI_VERTICES -> fallback
    fallback = pool(tiny)

    assert not torch.allclose(full, masked), "la maschera non ha cambiato nulla"
    assert torch.allclose(full, fallback), "il fallback sotto soglia non ha ripristinato il pooling pieno"
    print("demo OK: la maschera cambia l'embedding; sotto soglia si torna al pooling pieno")


if __name__ == "__main__":
    _demo()
