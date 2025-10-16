import torch
import torch.nn as nn
import diffusion_net


class DiffusionBackbone(nn.Module):
    """
    DiffusionNet-based backbone for extracting global embeddings from 3D meshes.

    This module represents the *shared geometric encoder* used in cross-topology
    training. It takes vertex coordinates and the corresponding geometric operators
    (mass matrix, Laplacian, eigenvectors, etc.), runs them through a DiffusionNet,
    and outputs a single L2-normalized embedding vector representing the whole mesh.

    Args:
        c_in (int): Number of input vertex features (usually 3 for (x, y, z)).
        c_width (int): Internal channel width of DiffusionNet (typical range: 32–512).
        c_out (int): Output feature dimension per vertex from DiffusionNet.
        emb_dim (int): Final global embedding dimension.
    """

    def __init__(self, c_in=3, c_width=128, c_out=64, emb_dim=64):
        super().__init__()

        # -------------------------------------------------------
        # (1) Geometric feature extractor (DiffusionNet backbone)
        # -------------------------------------------------------
        # DiffusionNet computes per-vertex features using diffusion operators
        # derived from the Laplace–Beltrami spectrum. Each block propagates
        # information across the mesh surface via spectral diffusion.
        #
        # outputs_at='vertices' → output features live on mesh vertices.
        # last_activation=None  → raw output, later processed by a projection head.
        self.encoder = diffusion_net.layers.DiffusionNet(
            C_in=c_in,
            C_out=c_out,
            C_width=c_width,
            N_block=4,
            outputs_at="vertices",
            last_activation=None,
        )

        # -------------------------------------------------------
        # (2) Global projection head (MLP)
        # -------------------------------------------------------
        # After computing per-vertex features (N × c_out),
        # we aggregate them globally (via mean pooling)
        # and project the pooled vector into a fixed-size
        # embedding space of dimension `emb_dim`.
        #
        # The small MLP introduces non-linearity and improves
        # the expressive power of the final latent representation.
        self.proj = nn.Sequential(
            nn.Linear(c_out, emb_dim),  # first projection
            nn.ReLU(inplace=True),      # non-linearity
            nn.Linear(emb_dim, emb_dim)  # second projection (bottleneck)
        )

    def forward(self, verts, mass, L, evals, evecs, gradX, gradY, faces):
        """
        Forward pass for a single mesh instance.

        Args:
            verts (torch.Tensor): Normalized vertex positions [N, 3].
            mass (torch.Tensor): Vertex mass vector [N].
            L (torch.Tensor): Discrete Laplacian matrix [N, N].
            evals (torch.Tensor): Laplacian eigenvalues [K].
            evecs (torch.Tensor): Laplacian eigenvectors [N, K].
            gradX (torch.Tensor): Gradient operator in X direction [N, K].
            gradY (torch.Tensor): Gradient operator in Y direction [N, K].
            faces (torch.Tensor): Face connectivity [F, 3].

        Returns:
            torch.Tensor: Global L2-normalized embedding vector [emb_dim].
        """
        # -------------------------------------------------------
        # (1) Add batch dimension (DiffusionNet expects batched input)
        # -------------------------------------------------------
        feats = verts.unsqueeze(0)       # [1, N, 3]
        mass = mass.unsqueeze(0)
        L = L.unsqueeze(0)
        evals = evals.unsqueeze(0)
        evecs = evecs.unsqueeze(0)
        gradX = gradX.unsqueeze(0)
        gradY = gradY.unsqueeze(0)
        faces = faces.unsqueeze(0)

        # -------------------------------------------------------
        # (2) DiffusionNet forward pass
        # -------------------------------------------------------
        # Produces per-vertex latent features describing local geometry.
        per_vertex = self.encoder(
            feats,
            mass,
            L=L,
            evals=evals,
            evecs=evecs,
            gradX=gradX,
            gradY=gradY,
            faces=faces,
        )  # → [1, N, c_out]

        # -------------------------------------------------------
        # (3) Global average pooling
        # -------------------------------------------------------
        # Collapse the per-vertex representation into a single vector.
        # (Optional) You can use mass-weighted pooling for better geometric fidelity:
        #   global_feat = (per_vertex.squeeze(0) * mass.T).sum(0) / mass.sum()
        per_vertex = per_vertex.squeeze(0)   # [N, c_out]
        global_feat = per_vertex.mean(dim=0)  # [c_out]

        # -------------------------------------------------------
        # (4) MLP projection + L2 normalization
        # -------------------------------------------------------
        emb = self.proj(global_feat)               # project to emb_dim
        emb = nn.functional.normalize(emb, p=2, dim=0)  # enforce unit norm

        return emb  # [emb_dim]
