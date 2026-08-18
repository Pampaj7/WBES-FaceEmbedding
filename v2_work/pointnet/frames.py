#!/usr/bin/env python
"""Alternative canonical frames for the vertex coordinates, applied to already-cached samples.

The frozen loader always does `V -= V.mean(0); V /= maxabs`. Both quantities are set by the
boundary -- the mean is over VERTICES rather than the surface, and maxabs is the single
farthest vertex, exactly what a crop removes. Measured over 40 identities, going original ->
crop moves the scale by a factor 0.878 with a spread of +-0.034 ACROSS IDENTITIES; that spread,
not the systematic shrink, is what corrupts a ranking between identities.

    frame      centre                  scale
    current    vertex mean             maxabs radius          (what the loader does)
    rms        mass-weighted centroid  mass-weighted RMS radius
    area       mass-weighted centroid  sqrt(total area)

WHY THIS CAN RUN ON ALREADY-NORMALISED COORDINATES. Both replacements are, like the loader's
own, a translation followed by a uniform scale. Writing the loader's map as
V_n = (V - c0)/s0, the mass-weighted centroid of V_n is (cm - c0)/s0 and its RMS radius is
rms/s0, so (V_n - cm_n)/rms_n = (V - cm)/rms exactly: the result does not depend on the frame
it was applied to. The same cancellation holds for sqrt(area), since area scales as s0^2.
So re-framing the cached tensors is identical to re-framing the raw mesh, and the frozen v1
loader stays untouched.
"""
from __future__ import annotations
import torch

FRAMES = ("current", "rms", "area")


def _total_area(V: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    tri = V[F.long()]
    return 0.5 * torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1).norm(dim=-1).sum()


def reframe(V: torch.Tensor, mass: torch.Tensor, faces: torch.Tensor, frame: str) -> torch.Tensor:
    if frame == "current":
        return V
    w = mass.reshape(-1).to(V.dtype).clamp_min(0)
    tot = w.sum()
    if not torch.isfinite(tot) or float(tot) <= 0:
        # Degenerate mass would silently produce NaNs downstream; fall back to the frame the
        # loader already produced rather than emit a broken sample.
        return V
    w = w / tot
    c = (w.unsqueeze(1) * V).sum(0, keepdim=True)
    X = V - c
    if frame == "rms":
        s = torch.sqrt((w * (X * X).sum(1)).sum())
    elif frame == "area":
        s = torch.sqrt(_total_area(V, faces))
    else:
        raise ValueError(f"unknown frame {frame!r}")
    if not torch.isfinite(s) or float(s) <= 1e-12:
        return V
    return X / s


def _demo() -> None:
    torch.manual_seed(0)
    V = torch.randn(500, 3)
    F = torch.randint(0, 500, (900, 3))
    mass = torch.rand(500) + 1e-3

    for f in ("rms", "area"):
        A = reframe(V, mass, F, f)
        # The whole justification is that the output does not depend on the frame the input
        # arrived in. Feed the same cloud under an arbitrary translation and uniform rescale.
        B = reframe(V * 7.3 - 4.1, mass, F, f)
        assert torch.allclose(A, B, atol=1e-4), (f, (A - B).abs().max())

    # rms frame: unit mass-weighted RMS radius by construction
    A = reframe(V, mass, F, "rms")
    w = mass / mass.sum()
    assert abs(float(torch.sqrt((w * (A * A).sum(1)).sum())) - 1.0) < 1e-4

    # area frame: unit total surface area by construction
    B = reframe(V, mass, F, "area")
    assert abs(float(_total_area(B, F)) - 1.0) < 1e-3, float(_total_area(B, F))

    # 'current' must be a genuine no-op, so the flag's default cannot silently change a run
    assert reframe(V, mass, F, "current") is V

    # degenerate mass falls back instead of emitting NaNs
    assert torch.isfinite(reframe(V, torch.zeros(500), F, "rms")).all()
    print("OK  frames demo passed")


if __name__ == "__main__":
    _demo()
