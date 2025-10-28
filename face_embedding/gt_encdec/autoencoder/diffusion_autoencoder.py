import torch
import torch.nn as nn
import sys

try:
    import diffusion_net
    DiffusionNet = diffusion_net.layers.DiffusionNet
except Exception as e1:
    try:
        from diffusion_net import DiffusionNet
    except Exception as e2:
        print("Errore: diffusion_net non trovato.")
        print(e1, e2)
        sys.exit(1)


class DiffusionAutoencoder(nn.Module):
    """
    Versione per-vertex (nessun pooling globale)
    """
    def __init__(self, latent_dim=128, width=128, n_blocks=4):
        super().__init__()
        self.latent_dim = latent_dim
        print(f"🧬 DiffusionAutoencoder (per-vertex) | Latent={latent_dim} Width={width} Blocks={n_blocks}")

        # Encoder: crea feature per vertice
        self.encoder = DiffusionNet(
            C_in=3, C_out=latent_dim, C_width=width, N_block=n_blocks,
            with_gradient_features=True, dropout=0.0,
        )

        # piccolo bottleneck MLP per regolarizzare le feature
        self.vertex_bottleneck = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, latent_dim)
        )

        self.k_spec = 16
        # Decoder riceve: [Z_per_vertex_bottleneck, V]
        self.decoder = DiffusionNet(
            C_in=latent_dim + 3 + self.k_spec, C_out=3, C_width=width, N_block=n_blocks,
            with_gradient_features=True, dropout=0.0,
        )

        self.tanh_out = nn.Identity()  # rimetti Tanh dopo il debug

    def forward(self, V, mass, L, evals, evecs, faces, gradX, gradY):
        # === ENCODER ===
        Z_per_vertex = self.encoder(V, mass, L, evals, evecs,
                                    faces=faces, gradX=gradX, gradY=gradY)
        Z_per_vertex = self.vertex_bottleneck(Z_per_vertex)
        Z_per_vertex = Z_per_vertex + 0.01 * torch.randn_like(Z_per_vertex)

        S = evecs[:, :self.k_spec]  # [n_verts, k_spec]
        # === DECODER ===
        Z_in = torch.cat([Z_per_vertex, V, S], dim=1)
        V_rec = self.decoder(Z_in, mass, L, evals, evecs,
                             faces=faces, gradX=gradX, gradY=gradY)
        V_rec = self.tanh_out(V_rec)

        # Z globale solo per logging
        Z_global = (Z_per_vertex * mass.unsqueeze(1)).sum(dim=0, keepdim=True) / (mass.sum() + 1e-9)
        return V_rec, Z_global