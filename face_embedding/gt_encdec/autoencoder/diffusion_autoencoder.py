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

    def forward(self, V, mass, L, evals, evecs, gradX, gradY):
        # V shape attesa: (N, 3)
        Z = self.encoder(V, mass, L, evals, evecs, gradX, gradY)
        V_rec = self.decoder(Z, mass, L, evals, evecs, gradX, gradY)
        return V_rec, Z
