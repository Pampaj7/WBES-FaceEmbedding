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
    DiffusionNet-based autoencoder producing:
    (1) a per-vertex latent field (rich local structure, used for reconstruction)
    (2) a global latent vector (compressed identity descriptor, used for metric learning)

    Key idea:
    - Reconstruction uses the full Z_per_vertex (local, detailed representation)
    - Identity analysis uses Z_global = pooled(Z_per_vertex) (compact descriptor)
    """

    def __init__(self, latent_dim=256, width=128, n_blocks=4, k_spec=16, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.k_spec = k_spec

        print(f"🧬 DiffusionAutoencoder | Z={latent_dim}, width={width}, blocks={n_blocks}, k_spec={k_spec}")

        # ------------------------------------------------------------
        # ENCODER
        # Produces a per-vertex latent field Z_per_vertex ∈ R^{N_verts × latent_dim}.
        # This representation contains rich local geometric information and
        # is used directly by the decoder to reconstruct the surface.
        # ------------------------------------------------------------
        self.encoder = DiffusionNet(
            C_in=3,
            C_out=latent_dim,
            C_width=width,
            N_block=n_blocks,
            with_gradient_features=True,
            dropout=0.0,
        )

        # ------------------------------------------------------------
        # BOTTLENECK
        # A small MLP applied independently at each vertex.
        # Helps compress/regularize the vertex-wise latent field.
        # Dropout added to prevent collapse and force diversity.
        # ------------------------------------------------------------
        self.vertex_bottleneck = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, latent_dim),
        )

        # ------------------------------------------------------------
        # DECODER
        # Input: [Z_per_vertex, spectral_basis S]
        #
        # IMPORTANT:
        # - The decoder does NOT receive the global latent vector.
        # - It reconstructs purely from local latent structure + intrinsic coords.
        # ------------------------------------------------------------
        cin_decoder = latent_dim + k_spec
        self.decoder = DiffusionNet(
            C_in=cin_decoder,
            C_out=3,
            C_width=width,
            N_block=n_blocks,
            with_gradient_features=True,
            dropout=0.0,
        )

        self.tanh_out = nn.Identity()  # no output saturation


    @staticmethod
    def _take_or_pad_evecs(evecs: torch.Tensor, k: int) -> torch.Tensor:
        """
        Ensures that the intrinsic spectral basis S has exactly k components.
        Pads with zeros if not enough eigenvectors are available.
        """
        n, k_avail = evecs.shape
        if k_avail >= k:
            return evecs[:, :k]
        pad = torch.zeros(n, k - k_avail, device=evecs.device, dtype=evecs.dtype)
        return torch.cat([evecs, pad], dim=1)


    def forward(self, V, mass, L, evals, evecs, faces, gradX, gradY):
        # ================================================================
        # 1. ENCODER → produces vertex-wise latent representation
        # ================================================================
        Z_per_vertex = self.encoder(
            V, mass, L, evals, evecs, faces=faces, gradX=gradX, gradY=gradY
        )

        # Local bottleneck
        Z_per_vertex = self.vertex_bottleneck(Z_per_vertex)

        # Small noise improves stability and avoids degenerate solutions
        Z_per_vertex = Z_per_vertex + 0.01 * torch.randn_like(Z_per_vertex)

        # ================================================================
        # 2. DECODER INPUT → concatenate per-vertex latent + spectral coords
        #
        # NOTE:
        #   We DO NOT use the global latent vector in the decoder.
        #   Reconstruction relies entirely on Z_per_vertex (local) + S (intrinsic).
        # ================================================================
        S = self._take_or_pad_evecs(evecs, self.k_spec)
        Z_in = torch.cat([Z_per_vertex, S], dim=1)

        V_rec = self.decoder(
            Z_in, mass, L, evals, evecs, faces=faces, gradX=gradX, gradY=gradY
        )
        V_rec = self.tanh_out(V_rec)

        # ================================================================
        # 3. GLOBAL LATENT VECTOR
        #
        # Z_global = mean_pooling(Z_per_vertex)
        #
        # Purpose:
        # - This is NOT used for reconstruction.
        # - This is a compact identity embedding used for:
        #       - comparing identities
        #       - computing latent distances
        #       - Pearson/Spearman/WBES-style metrics
        #       - verifying latent space structure
        #
        # We return it only for analysis, logging, and metric learning.
        # ================================================================
        Z_global = Z_per_vertex.mean(dim=0, keepdim=True)

        return V_rec, Z_global


"""
--------------------------------------------------------------------------------
NOTE ON LATENT REPRESENTATIONS (Z_per_vertex vs Z_global)

This autoencoder produces two different kinds of latent features:

1) Z_per_vertex  — shape [N_vertices, latent_dim]
   A high-resolution latent field. Each vertex has its own 256-dimensional
   descriptor. This representation is extremely expressive and allows the
   decoder to reconstruct mesh geometry with high fidelity. However:

     • It is topology-dependent (same vertex order needed).
     • It contains millions of values (e.g., 53k × 256).
     • It is extremely sensitive to local noise and mesh artifacts.
     • It does not define a stable global identity descriptor.
     • It cannot be used directly for metric-learning or cross-topology tasks.

   Z_per_vertex is therefore used *only* to drive reconstruction quality,
   not as an identity descriptor.

2) Z_global — shape [1, latent_dim]
   A compact global embedding obtained by pooling Z_per_vertex. This is the
   identity-level representation that can be compared across subjects.
   It is:

     • topology-agnostic
     • robust to local noise
     • compact (256-D)
     • suitable for Pearson/Spearman correlation, clustering, PCA/UMAP
     • usable for metric-learning
     • stable across different mesh discretizations

   Even though Z_global is derived from Z_per_vertex, it encodes only the
   *global facial structure*, not the per-vertex geometry. This makes it the
   correct latent space for identity comparison and correlation analysis.

Summary:
   - Z_per_vertex → used for reconstruction only.
   - Z_global     → used for identity, distances, embedding space analysis.

--------------------------------------------------------------------------------
"""
