import torch
import torch.nn as nn
import sys

try:
    import diffusion_net
    DiffusionNet = diffusion_net.layers.DiffusionNet
except Exception:
    from diffusion_net import DiffusionNet


class DiffusionAutoencoder(nn.Module):
    """
    DiffusionNet Autoencoder unificato con vettore globale Z.
    Input encoder: V, operatori spettrali.
    Input decoder: Z_broadcast + S (nessuna dipendenza diretta da V).
    """
    def __init__(self, latent_dim=256, width=128, n_blocks=4, k_spec=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.k_spec = k_spec

        print(f"🧬 DiffusionAutoencoder | Z={latent_dim}, width={width}, blocks={n_blocks}, k_spec={k_spec}")

        # === ENCODER ===
        self.encoder = DiffusionNet(
            C_in=3,
            C_out=latent_dim,
            C_width=width,
            N_block=n_blocks,
            with_gradient_features=True,
            dropout=0.0,
        )

        # === BOTTLENECK ===
        self.vertex_bottleneck = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, latent_dim),
        )

        # === DECODER ===
        # Input = Z_broadcast + S
        cin_decoder = latent_dim + k_spec
        self.decoder = DiffusionNet(
            C_in=cin_decoder,
            C_out=3,
            C_width=width,
            N_block=n_blocks,
            with_gradient_features=True,
            dropout=0.0,
        )

        self.tanh_out = nn.Identity()  # Niente saturazione iniziale

    @staticmethod
    def _take_or_pad_evecs(evecs: torch.Tensor, k: int) -> torch.Tensor:
        """Assicura che evecs abbia esattamente k colonne (pad se necessario)."""
        n, k_avail = evecs.shape
        if k_avail >= k:
            return evecs[:, :k]
        pad = torch.zeros(n, k - k_avail, device=evecs.device, dtype=evecs.dtype)
        return torch.cat([evecs, pad], dim=1)

        """    def forward(self, V, mass, L, evals, evecs, faces, gradX, gradY):
        # === ENCODER ===
        Z_per_vertex = self.encoder(V, mass, L, evals, evecs,
                                    faces=faces, gradX=gradX, gradY=gradY)
        Z_per_vertex = self.vertex_bottleneck(Z_per_vertex)
        Z_per_vertex = Z_per_vertex + 0.01 * torch.randn_like(Z_per_vertex)

        # === GLOBAL LATENT VECTOR ===
        Z_global = (Z_per_vertex * mass.unsqueeze(1)).sum(dim=0, keepdim=True) / (mass.sum() + 1e-9)
        Z_broadcast = Z_global.expand(V.shape[0], -1)

        # === DECODER ===
        S = self._take_or_pad_evecs(evecs, self.k_spec)
        Z_in = torch.cat([Z_broadcast, S], dim=1)

        V_rec = self.decoder(Z_in, mass, L, evals, evecs,
                             faces=faces, gradX=gradX, gradY=gradY)
        V_rec = self.tanh_out(V_rec)
        return V_rec, Z_global"""

    def forward(self, V, mass, L, evals, evecs, faces, gradX, gradY):
        # === ENCODER ===
        Z_per_vertex = self.encoder(V, mass, L, evals, evecs,
                                    faces=faces, gradX=gradX, gradY=gradY)
        Z_per_vertex = self.vertex_bottleneck(Z_per_vertex)
        Z_per_vertex = Z_per_vertex + 0.01 * torch.randn_like(Z_per_vertex)

        # === DECODER ===
        S = self._take_or_pad_evecs(evecs, self.k_spec)
        Z_in = torch.cat([Z_per_vertex, S], dim=1)  # no V, no broadcast
        V_rec = self.decoder(Z_in, mass, L, evals, evecs,
                            faces=faces, gradX=gradX, gradY=gradY)
        V_rec = self.tanh_out(V_rec)
        return V_rec, Z_per_vertex.mean(dim=0, keepdim=True)  # just for logging


'''Z_broadcast → feature globali

Z_broadcast è la versione per-vertex del vettore globale Z, ottenuto dall encoder.

Ogni vertice riceve la stessa informazione globale — cioè il contesto del volto intero.

Dim: [n_verts, latent_dim].

Serve per dire al decoder in quale punto dello “spazio delle forme” ci troviamo 
(espressione, identità, ecc.).

V → coordinate originali

Le coordinate 3D dei vertici in input.

Dim: [n_verts, 3].

Serve al decoder per mantenere allineamento spaziale e non “perdere” la struttura 
globale della mesh.
In pratica, gli dai una base geometrica per capire dove ricostruire.

S → basi spettrali (autovettori Laplaciano)

S = evecs[:, :k_spec] sono i primi k_spec autovettori del Laplaciano.

Dim: [n_verts, k_spec].

Servono come coordinate intrinseche della superficie, cioè un sistema di riferimento 
indipendente dalla topologia esplicita, che permette al decoder di ricostruire forme 
anche se ordinamento dei vertici cambia.
'''       
