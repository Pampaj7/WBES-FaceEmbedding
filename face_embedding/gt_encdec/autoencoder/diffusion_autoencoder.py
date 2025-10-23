import torch
import torch.nn as nn
from diffusion_net import DiffusionNet


class DiffusionAutoencoder(nn.Module):
    def __init__(self, latent_dim=128, width=128, n_blocks=4):
        super().__init__()

        self.encoder = DiffusionNet(
            C_in=3,
            C_out=latent_dim,
            C_width=width,
            with_gradient_features=False,
            dropout=0.0,
        )

        self.decoder = DiffusionNet(
            C_in=latent_dim,
            C_out=3,
            C_width=width,
            with_gradient_features=False,
            dropout=0.0,
        )

        # 🔹 nuovi layer di stabilizzazione
        self.norm_latent = nn.LayerNorm(latent_dim)
        self.tanh_out = nn.Tanh()

    def forward(self, V, mass, L, evals, evecs, gradX, gradY):
        # V shape attesa: (N, 3)
        Z = self.encoder(V, mass, L, evals, evecs, gradX, gradY)

        # 🔹 normalizza il codice latente per evitare esplosioni
        if Z.ndim == 2:  # (N, latent_dim)
            Z = self.norm_latent(Z)
        elif Z.ndim == 3:  # (B, N, latent_dim)
            Z = self.norm_latent(Z.transpose(1, 2)).transpose(1, 2)

        V_rec = self.decoder(Z, mass, L, evals, evecs, gradX, gradY)

        # 🔹 limita i vertici ricostruiti in [-1, 1]
        V_rec = self.tanh_out(V_rec)

        return V_rec, Z
