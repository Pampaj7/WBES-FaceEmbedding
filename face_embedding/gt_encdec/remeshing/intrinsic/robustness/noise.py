from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class PerturbationParams:
    outlier_frac: float
    outlier_scale: float
    rigid_rot_deg: float
    rigid_trans_scale: float
    rigid_rot_deg_min: float = 0.0
    rigid_trans_scale_min: float = 0.0

    @classmethod
    def from_namespace(cls, args: object) -> "PerturbationParams":
        return cls(
            outlier_frac=float(getattr(args, "outlier_frac")),
            outlier_scale=float(getattr(args, "outlier_scale")),
            rigid_rot_deg=float(getattr(args, "rigid_rot_deg")),
            rigid_trans_scale=float(getattr(args, "rigid_trans_scale")),
            rigid_rot_deg_min=float(getattr(args, "rigid_rot_deg_min", 0.0)),
            rigid_trans_scale_min=float(getattr(args, "rigid_trans_scale_min", 0.0)),
        )


def parse_noise_modes(text: str) -> List[str]:
    allowed = {"jitter", "rigid", "rotation", "translation", "outliers"}
    modes = [tok.strip().lower() for tok in text.split(",") if tok.strip()]
    if not modes:
        raise ValueError("noise_modes must contain at least one mode")
    bad = [mode for mode in modes if mode not in allowed]
    if bad:
        raise ValueError(f"Unsupported noise modes: {bad}. Allowed={sorted(allowed)}")
    return modes


def parse_noise_mode_weights(text: str, noise_modes: Sequence[str]) -> List[float]:
    modes = [str(mode) for mode in noise_modes]
    if not modes:
        raise ValueError("noise_modes must contain at least one mode")

    if not str(text).strip():
        weights = np.ones(len(modes), dtype=np.float64)
        return (weights / weights.sum()).tolist()

    allowed = set(modes)
    weight_map = {mode: 1.0 for mode in modes}
    seen = set()

    for raw_token in str(text).split(","):
        token = raw_token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                "noise_mode_weights must use mode=weight entries, e.g. 'jitter=1,translation=6,rotation=2,outliers=1'"
            )
        mode, raw_weight = token.split("=", 1)
        mode = mode.strip().lower()
        if mode not in allowed:
            raise ValueError(f"Unsupported noise mode in noise_mode_weights: {mode}. Allowed={sorted(allowed)}")
        if mode in seen:
            raise ValueError(f"Duplicate weight for noise mode: {mode}")
        try:
            weight = float(raw_weight.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid weight for noise mode {mode}: {raw_weight!r}") from exc
        if weight < 0.0:
            raise ValueError(f"Noise mode weight must be >= 0 for {mode}, got {weight}")
        weight_map[mode] = weight
        seen.add(mode)

    weights = np.asarray([weight_map[mode] for mode in modes], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("noise_mode_weights must have a positive total weight")
    return (weights / total).tolist()


def sample_log_uniform_sigma(sigma_min: float, sigma_max: float, rng: np.random.Generator) -> float:
    lo = max(float(sigma_min), 1e-12)
    hi = max(float(sigma_max), lo)
    if hi <= lo:
        return lo
    u = rng.uniform(math.log(lo), math.log(hi))
    return float(math.exp(u))


def rigid_angle_max_deg_from_sigma(
    sigma: float,
    rigid_rot_deg: float,
    rigid_rot_deg_min: float = 0.0,
) -> float:
    sigma = float(max(sigma, 0.0))
    if sigma <= 0.0:
        return 0.0
    return float(max(float(rigid_rot_deg_min) + float(rigid_rot_deg) * sigma, 0.0))


def rigid_trans_axis_std_from_sigma(
    sigma: float,
    rigid_trans_scale: float,
    rigid_trans_scale_min: float = 0.0,
) -> float:
    sigma = float(max(sigma, 0.0))
    if sigma <= 0.0:
        return 0.0
    return float(max(float(rigid_trans_scale_min) + float(rigid_trans_scale) * sigma, 0.0))


def apply_xyz_perturbation(
    V: torch.Tensor,
    mode: str,
    sigma: float,
    outlier_frac: float,
    outlier_scale: float,
    rigid_rot_deg: float,
    rigid_trans_scale: float,
    rigid_rot_deg_min: float = 0.0,
    rigid_trans_scale_min: float = 0.0,
) -> torch.Tensor:
    """Apply GPU-safe XYZ perturbation with vectorized operations."""
    sigma = float(max(sigma, 0.0))
    if sigma <= 0.0:
        return V

    mode = mode.lower()
    device = V.device
    dtype = V.dtype

    if mode == "jitter":
        noise = sigma * torch.randn(V.shape, device=device, dtype=dtype)
        return V + noise

    if mode in {"rigid", "rotation", "translation"}:
        center = V.mean(dim=0, keepdim=True)

        if mode in {"rigid", "rotation"}:
            axis = torch.randn(3, device=device, dtype=dtype)
            axis = axis / (axis.norm() + 1e-12)

            angle_max = math.radians(
                rigid_angle_max_deg_from_sigma(
                    sigma=sigma,
                    rigid_rot_deg=rigid_rot_deg,
                    rigid_rot_deg_min=rigid_rot_deg_min,
                )
            )
            angle = (2.0 * torch.rand((), device=device, dtype=dtype) - 1.0) * angle_max

            ax, ay, az = axis[0], axis[1], axis[2]
            z0 = torch.zeros((), device=device, dtype=dtype)
            K = torch.stack(
                [
                    torch.stack([z0, -az, ay]),
                    torch.stack([az, z0, -ax]),
                    torch.stack([-ay, ax, z0]),
                ],
                dim=0,
            )
            I = torch.eye(3, device=device, dtype=dtype)
            sin_a = torch.sin(angle)
            cos_a = torch.cos(angle)
            R = I + sin_a * K + (1.0 - cos_a) * (K @ K)
            V_out = (V - center) @ R.transpose(0, 1) + center
        else:
            V_out = V

        if mode in {"rigid", "translation"}:
            trans_axis_std = rigid_trans_axis_std_from_sigma(
                sigma=sigma,
                rigid_trans_scale=rigid_trans_scale,
                rigid_trans_scale_min=rigid_trans_scale_min,
            )
            trans = trans_axis_std * torch.randn((1, 3), device=device, dtype=dtype)
            V_out = V_out + trans

        return V_out

    if mode == "outliers":
        n_verts = int(V.shape[0])
        if n_verts == 0:
            return V

        p = float(np.clip(outlier_frac, 0.0, 1.0))
        mask = torch.rand(n_verts, device=device) < p
        if p > 0.0 and not bool(mask.any()):
            idx = torch.randint(0, n_verts, (1,), device=device)
            mask[idx] = True

        displacement = outlier_scale * sigma * torch.randn(V.shape, device=device, dtype=dtype)
        return V + displacement * mask[:, None].to(dtype)

    raise ValueError(f"Unknown perturbation mode: {mode}")


def apply_xyz_perturbation_with_params(
    V: torch.Tensor,
    mode: str,
    sigma: float,
    params: PerturbationParams,
) -> torch.Tensor:
    return apply_xyz_perturbation(
        V=V,
        mode=mode,
        sigma=sigma,
        outlier_frac=params.outlier_frac,
        outlier_scale=params.outlier_scale,
        rigid_rot_deg=params.rigid_rot_deg,
        rigid_trans_scale=params.rigid_trans_scale,
        rigid_rot_deg_min=params.rigid_rot_deg_min,
        rigid_trans_scale_min=params.rigid_trans_scale_min,
    )
