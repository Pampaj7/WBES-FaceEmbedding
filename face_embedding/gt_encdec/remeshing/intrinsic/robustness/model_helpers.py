from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .paths import ensure_autoencoder_dir_on_syspath


ensure_autoencoder_dir_on_syspath()

from diffusion_autoencoder import (  # noqa: E402
    DiffusionAutoencoder,
    DiffusionEncoderOnly,
    DiffusionEncoderOnlyIntrinsec,
    DiffusionEncoderXYZSpectrum,
    TwoTowerConcat,
    TwoTowerGatedRobust,
)
from latent_loss import smooth_loss  # noqa: E402


class SpectrumMLPBaseline(nn.Module):
    """Small spectrum-only baseline: lambda_1..lambda_k -> latent embedding."""

    def __init__(
        self,
        k_spec: int,
        latent_dim: int,
        hidden_dim: int,
        dropout: float,
        log_spec: bool,
        eps: float,
    ) -> None:
        super().__init__()
        self.k_spec = int(k_spec)
        self.log_spec = bool(log_spec)
        self.eps = float(eps)

        self.net = nn.Sequential(
            nn.Linear(self.k_spec, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def _spectrum_vector(self, evals: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        ev = evals.flatten()
        if ev.numel() <= 1:
            return torch.zeros(self.k_spec, device=ev.device, dtype=dtype)

        ev = ev[1:].to(dtype)
        if ev.numel() >= self.k_spec:
            spec = ev[: self.k_spec]
        else:
            pad = torch.zeros(self.k_spec - ev.numel(), device=ev.device, dtype=dtype)
            spec = torch.cat([ev, pad], dim=0)

        if self.log_spec:
            spec = torch.log(spec.clamp_min(self.eps))
        return spec

    def forward(self, evals: torch.Tensor) -> torch.Tensor:
        spec = self._spectrum_vector(evals, dtype=evals.dtype).unsqueeze(0)
        return self.net(spec)


def _default_gate_info(z: torch.Tensor) -> Dict[str, torch.Tensor]:
    nan = torch.tensor(float("nan"), device=z.device, dtype=z.dtype)
    return {"g_mean": nan, "g_p10": nan, "g_p90": nan}


def _normalize_gate_info(gate_info: Dict[str, torch.Tensor], z: torch.Tensor) -> Dict[str, torch.Tensor]:
    out = _default_gate_info(z)
    for key in ("g_mean", "g_p10", "g_p90"):
        if key not in gate_info:
            continue
        value = gate_info[key]
        if torch.is_tensor(value):
            out[key] = value.to(device=z.device, dtype=z.dtype)
        else:
            out[key] = torch.tensor(float(value), device=z.device, dtype=z.dtype)
    return out


def build_model(args: object, device: torch.device) -> nn.Module:
    model_name = str(getattr(args, "model"))

    if model_name == "gated_twotower":
        if int(getattr(args, "k_spec")) <= 0:
            raise ValueError("gated_twotower requires --k_spec > 0")
        model: nn.Module = TwoTowerGatedRobust(
            latent_dim=getattr(args, "latent_dim"),
            width=getattr(args, "width"),
            n_blocks=getattr(args, "n_blocks"),
            dropout=getattr(args, "dropout"),
            k_spec=getattr(args, "k_spec"),
            log_spec=getattr(args, "log_spec"),
            eps=getattr(args, "eps"),
            pool_mode=getattr(args, "pool_mode"),
            xyz_feature_dropout=getattr(args, "xyz_feature_dropout"),
        )
    elif model_name == "concat_twotower":
        if int(getattr(args, "k_spec")) <= 0:
            raise ValueError("concat_twotower requires --k_spec > 0")
        model = TwoTowerConcat(
            latent_dim=getattr(args, "latent_dim"),
            width=getattr(args, "width"),
            n_blocks=getattr(args, "n_blocks"),
            dropout=getattr(args, "dropout"),
            k_spec=getattr(args, "k_spec"),
            log_spec=getattr(args, "log_spec"),
            eps=getattr(args, "eps"),
            pool_mode=getattr(args, "pool_mode"),
            xyz_feature_dropout=getattr(args, "xyz_feature_dropout"),
        )
    elif model_name == "xyz_dn":
        model = DiffusionEncoderOnly(
            latent_dim=getattr(args, "latent_dim"),
            width=getattr(args, "width"),
            n_blocks=getattr(args, "n_blocks"),
            dropout=getattr(args, "dropout"),
            pool_mode=getattr(args, "pool_mode"),
        )
    elif model_name == "intrinsic_dn":
        if int(getattr(args, "eig_k")) <= 0:
            raise ValueError("intrinsic_dn requires --eig_k > 0")
        if int(getattr(args, "n_hks")) < 0 or int(getattr(args, "n_wks")) < 0:
            raise ValueError("--n_hks and --n_wks must be >= 0")
        if (not bool(getattr(args, "use_xyz"))) and int(getattr(args, "n_hks")) <= 0 and int(getattr(args, "n_wks")) <= 0:
            raise ValueError("intrinsic_dn requires at least one active input among --use_xyz/--n_hks/--n_wks")
        model = DiffusionEncoderOnlyIntrinsec(
            latent_dim=getattr(args, "latent_dim"),
            width=getattr(args, "width"),
            n_blocks=getattr(args, "n_blocks"),
            dropout=getattr(args, "dropout"),
            use_xyz=getattr(args, "use_xyz"),
            n_hks=getattr(args, "n_hks"),
            n_wks=getattr(args, "n_wks"),
            eig_k=getattr(args, "eig_k"),
            eps=getattr(args, "eps"),
            pool_mode=getattr(args, "pool_mode"),
        )
    elif model_name == "xyz_spec_dn":
        if not (bool(getattr(args, "use_xyz")) or bool(getattr(args, "use_spectrum"))):
            raise ValueError("xyz_spec_dn requires at least one of --use_xyz/--use_spectrum")
        if bool(getattr(args, "use_spectrum")) and int(getattr(args, "k_spec")) <= 0:
            raise ValueError("xyz_spec_dn with --use_spectrum requires --k_spec > 0")
        model = DiffusionEncoderXYZSpectrum(
            latent_dim=getattr(args, "latent_dim"),
            width=getattr(args, "width"),
            n_blocks=getattr(args, "n_blocks"),
            dropout=getattr(args, "dropout"),
            use_xyz=getattr(args, "use_xyz"),
            use_spectrum=getattr(args, "use_spectrum"),
            k_spec=getattr(args, "k_spec"),
            log_input=getattr(args, "log_spec"),
            eps=getattr(args, "eps"),
        )
    elif model_name == "spec_mlp":
        if int(getattr(args, "k_spec")) <= 0:
            raise ValueError("spec_mlp requires --k_spec > 0")
        model = SpectrumMLPBaseline(
            k_spec=getattr(args, "k_spec"),
            latent_dim=getattr(args, "latent_dim"),
            hidden_dim=getattr(args, "width"),
            dropout=getattr(args, "dropout"),
            log_spec=getattr(args, "log_spec"),
            eps=getattr(args, "eps"),
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model.to(device)


def forward_model(
    model: nn.Module,
    sample_dict: Dict[str, torch.Tensor],
    V_in: torch.Tensor,
    return_gate_info: bool,
    add_noise: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    mass = sample_dict["mass"]
    L = sample_dict["L"]
    evals = sample_dict["evals"]
    evecs = sample_dict["evecs"]
    faces = sample_dict["faces"]
    gradX = sample_dict["gradX"]
    gradY = sample_dict["gradY"]

    if isinstance(model, (TwoTowerGatedRobust, TwoTowerConcat)):
        if return_gate_info:
            z, gate_info = model(
                V_in,
                mass,
                L,
                evals,
                evecs,
                faces,
                gradX,
                gradY,
                return_gate_info=True,
                add_noise=add_noise,
            )
            return z, _normalize_gate_info(gate_info, z)

        z = model(
            V_in,
            mass,
            L,
            evals,
            evecs,
            faces,
            gradX,
            gradY,
            return_gate_info=False,
            add_noise=add_noise,
        )
        return z, _default_gate_info(z)

    if isinstance(model, (DiffusionEncoderOnly, DiffusionEncoderOnlyIntrinsec, DiffusionEncoderXYZSpectrum)):
        z = model(
            V_in,
            mass,
            L,
            evals,
            evecs,
            faces,
            gradX,
            gradY,
            return_per_vertex=False,
            add_noise=add_noise,
        )
        return z, _default_gate_info(z)

    if isinstance(model, DiffusionAutoencoder):
        _, z = model(
            V_in,
            mass,
            L,
            evals,
            evecs,
            faces,
            gradX,
            gradY,
        )
        return z, _default_gate_info(z)

    if isinstance(model, SpectrumMLPBaseline):
        z = model(evals=evals)
        return z, _default_gate_info(z)

    raise TypeError(f"Unsupported model type in forward_model(): {type(model).__name__}")


def smooth_term_from_model(
    model: nn.Module,
    sample_dict: Dict[str, torch.Tensor],
    V_in: torch.Tensor,
) -> torch.Tensor | None:
    mass = sample_dict["mass"]
    L = sample_dict["L"]
    evals = sample_dict["evals"]
    evecs = sample_dict["evecs"]
    faces = sample_dict["faces"]
    gradX = sample_dict["gradX"]
    gradY = sample_dict["gradY"]

    if isinstance(model, (TwoTowerGatedRobust, TwoTowerConcat)):
        z_field = model.xyz_vertex_bottleneck(
            model.xyz_encoder(
                V_in,
                mass,
                L,
                evals,
                evecs,
                faces=faces,
                gradX=gradX,
                gradY=gradY,
            )
        )
        return smooth_loss(z_field, L)

    if isinstance(model, (DiffusionEncoderOnly, DiffusionEncoderOnlyIntrinsec, DiffusionEncoderXYZSpectrum)):
        z_field, _ = model(
            V_in,
            mass,
            L,
            evals,
            evecs,
            faces,
            gradX,
            gradY,
            return_per_vertex=True,
            add_noise=False,
        )
        return smooth_loss(z_field, L)

    if isinstance(model, DiffusionAutoencoder):
        z_field = model.vertex_bottleneck(
            model.encoder(
                V_in,
                mass,
                L,
                evals,
                evecs,
                faces=faces,
                gradX=gradX,
                gradY=gradY,
            )
        )
        return smooth_loss(z_field, L)

    return None
