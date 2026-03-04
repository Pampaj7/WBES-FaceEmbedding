import torch
import torch.nn as nn
import sys
import math

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


class DiffusionEncoderOnly(nn.Module):
    """
    Encoder-only:
    - NO decoder
    - NO reconstruction
    - Z_per_vertex → Z_global via mean pooling
    - Ideal for metric-learning experiments and latent space analysis.

    Goal:
    Learn a latent space that preserves subject-wise structure
    without the pressure of a geometric reconstruction loss.
    """

    def __init__(self, latent_dim=256, width=128, n_blocks=4, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim

        print(f"🧬 DiffusionEncoderOnly | Z={latent_dim}, width={width}, blocks={n_blocks}")

        # ------------------------------------------------------------
        # ENCODER → vertex-wise latent field
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
        # BOTTLENECK → regularization of per-vertex latents
        # ------------------------------------------------------------
        self.vertex_bottleneck = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, latent_dim),
        )

    def forward(
        self,
        V,
        mass,
        L,
        evals,
        evecs,
        faces,
        gradX,
        gradY,
        return_per_vertex: bool = False,
        add_noise: bool = True,
    ):
        """
        If return_per_vertex == False (default):
            returns Z_global  ∈ R^{1 × latent_dim}
        If return_per_vertex == True:
            returns (Z_per_vertex, Z_global)
        """
        # 1. Encoder → per-vertex latents
        Z_per_vertex = self.encoder(
            V, mass, L, evals, evecs,
            faces=faces, gradX=gradX, gradY=gradY
        )  # (N_verts, latent_dim)

        # 2. Bottleneck
        Z_per_vertex = self.vertex_bottleneck(Z_per_vertex)

        # 3. Optional noise (for training only)
        if add_noise:
            Z_per_vertex = Z_per_vertex + 0.01 * torch.randn_like(Z_per_vertex)

        # 4. Global pooling
        Z_global = Z_per_vertex.mean(dim=0, keepdim=True)  # (1, latent_dim)

        if return_per_vertex:
            return Z_per_vertex, Z_global

        return Z_global


class DiffusionEncoderOnlyIntrinsec(nn.Module):
    """
    Encoder-only variant of DiffusionEncoderOnly with optional intrinsic descriptors (HKS/WKS) + XYZ.

    Notes:
      - Handles possible batch dims in (evals, evecs).
      - Uses a limited number of eigenpairs for stability/speed.
      - Optional mean+max pooling (recommended).
    """

    def __init__(
        self,
        latent_dim=256,
        width=128,
        n_blocks=4,
        dropout=0.1,
        use_xyz=True,
        n_hks=0,
        n_wks=0,
        hks_k_high=50,
        eig_k=300,              # NEW: cap eigenpairs used for descriptors
        eps=1e-6,
        pool_mode="mean",       # NEW: "mean" or "meanmax"
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.use_xyz = use_xyz
        self.n_hks = int(n_hks)
        self.n_wks = int(n_wks)
        self.hks_k_high = int(hks_k_high)
        self.eig_k = int(eig_k)
        self.eps = float(eps)
        self.pool_mode = str(pool_mode)

        C_in = (3 if use_xyz else 0) + self.n_hks + self.n_wks
        if C_in <= 0:
            raise ValueError("At least one input feature must be enabled.")

        print(
            f"🧬 DiffusionEncoderOnlyIntrinsec | "
            f"Z={latent_dim}, C_in={C_in} "
            f"[XYZ={use_xyz} + HKS={self.n_hks} + WKS={self.n_wks}], "
            f"eig_k={self.eig_k}, pool={self.pool_mode}, "
            f"width={width}, blocks={n_blocks}"
        )

        self.encoder = DiffusionNet(
            C_in=C_in,
            C_out=latent_dim,
            C_width=width,
            N_block=n_blocks,
            with_gradient_features=True,
            dropout=0.0,
        )

        self.vertex_bottleneck = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, latent_dim),
        )

        # If mean+max pooling, project back to latent_dim
        if self.pool_mode == "meanmax":
            self.pool_proj = nn.Linear(2 * latent_dim, latent_dim)
        elif self.pool_mode == "mean":
            self.pool_proj = nn.Identity()
        else:
            raise ValueError("pool_mode must be 'mean' or 'meanmax'")

    @staticmethod
    def _squeeze_eigs(evals: torch.Tensor, evecs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if evals.dim() == 2:
            evals = evals.squeeze(0)
        if evecs.dim() == 3:
            evecs = evecs.squeeze(0)
        return evals, evecs

    def _truncate_eigs(self, evals: torch.Tensor, evecs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Expects evals [K], evecs [N,K]
        K = min(int(evals.numel()), int(evecs.shape[1]), int(self.eig_k))
        if K < 3:
            return evals[:K], evecs[:, :K]
        return evals[:K], evecs[:, :K]

    def compute_hks(self, evals: torch.Tensor, evecs: torch.Tensor, n_scales: int) -> torch.Tensor:
        if n_scales <= 0:
            return torch.zeros(evecs.size(0), 0, device=evecs.device, dtype=evecs.dtype)

        evals, evecs = self._squeeze_eigs(evals, evecs)
        evals, evecs = self._truncate_eigs(evals, evecs)

        N = int(evecs.shape[0])
        K = int(evals.numel())
        if K < 3:
            return torch.zeros(N, n_scales, device=evecs.device, dtype=evecs.dtype)

        evals = evals.clamp_min(self.eps)

        # Use low/high eigenvalues robustly
        lam_low = evals[1]
        hi_idx = min(K - 1, max(2, self.hks_k_high))
        lam_high = evals[hi_idx]

        t_min = 4.0 / lam_high
        t_max = 4.0 / lam_low

        # logspace expects python floats
        t = torch.logspace(
            math.log10(float(t_min)),
            math.log10(float(t_max)),
            int(n_scales),
            device=evals.device,
            dtype=evals.dtype,
        )

        exp_kt = torch.exp(-evals[:, None] * t[None, :])  # [K, S]
        hks = (evecs ** 2) @ exp_kt                       # [N, S]
        hks = torch.log(hks + self.eps)
        return hks

    def compute_wks(self, evals: torch.Tensor, evecs: torch.Tensor, n_energies: int) -> torch.Tensor:
        if n_energies <= 0:
            return torch.zeros(evecs.size(0), 0, device=evecs.device, dtype=evecs.dtype)

        evals, evecs = self._squeeze_eigs(evals, evecs)
        evals, evecs = self._truncate_eigs(evals, evecs)

        N = int(evecs.shape[0])
        K = int(evals.numel())
        if K < 3:
            return torch.zeros(N, n_energies, device=evecs.device, dtype=evecs.dtype)

        evals = evals.clamp_min(self.eps)
        log_ev = torch.log(evals)

        e1 = log_ev[1]
        eN = log_ev[-1]
        energies = torch.linspace(float(e1), float(eN), int(n_energies), device=evals.device, dtype=evals.dtype)

        # sigma heuristic; protect degenerate case
        if energies.numel() > 1:
            sigma = (energies[1] - energies[0]) * 7.0
        else:
            sigma = torch.tensor(1.0, device=evals.device, dtype=evals.dtype)

        diff = log_ev[:, None] - energies[None, :]  # [K,E]
        kernel = torch.exp(-(diff ** 2) / (2 * sigma ** 2 + 1e-12))
        kernel = kernel / (kernel.sum(dim=0, keepdim=True) + 1e-8)

        wks = (evecs ** 2) @ kernel
        wks = torch.log(wks + self.eps)
        return wks

    def forward(
        self,
        V,
        mass,
        L,
        evals,
        evecs,
        faces,
        gradX,
        gradY,
        return_per_vertex: bool = False,
        add_noise: bool = True,
    ):
        feats = []

        if self.use_xyz:
            feats.append(V)

        if self.n_hks > 0:
            feats.append(self.compute_hks(evals, evecs, self.n_hks))

        if self.n_wks > 0:
            feats.append(self.compute_wks(evals, evecs, self.n_wks))

        x = torch.cat(feats, dim=1)

        Z_per_vertex = self.encoder(
            x, mass, L, evals, evecs, faces=faces, gradX=gradX, gradY=gradY
        )

        Z_per_vertex = self.vertex_bottleneck(Z_per_vertex)

        if add_noise:
            Z_per_vertex = Z_per_vertex + 0.01 * torch.randn_like(Z_per_vertex)

        # Pooling
        Z_mean = Z_per_vertex.mean(dim=0, keepdim=True)
        if self.pool_mode == "meanmax":
            Z_max = Z_per_vertex.max(dim=0, keepdim=True).values
            Z_global = self.pool_proj(torch.cat([Z_mean, Z_max], dim=1))
        else:
            Z_global = self.pool_proj(Z_mean)

        if return_per_vertex:
            return Z_per_vertex, Z_global
        return Z_global
    
class DiffusionEncoderXYZSpectrum(nn.Module):
    """
    DiffusionNet encoder that uses:
      - XYZ per-vertex (optional)
      - Laplacian eigenvalues spectrum lambda_1..lambda_K as a GLOBAL descriptor,
        replicated per-vertex and concatenated to XYZ.

    This is the closest DiffusionNet analogue to the Hybrid MLP (XYZ + spectrum),
    while keeping the SAME training regime as DiffusionEncoderOnly:
      - no input normalization
      - mean pooling
      - same bottleneck + noise
    """

    def __init__(
        self,
        latent_dim: int = 256,
        width: int = 128,
        n_blocks: int = 4,
        dropout: float = 0.1,
        use_xyz: bool = True,
        use_spectrum: bool = True,
        k_spec: int = 100,
        log_input: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()

        if not (use_xyz or use_spectrum):
            raise ValueError("At least one of use_xyz or use_spectrum must be True.")
        if use_spectrum and k_spec <= 0:
            raise ValueError("k_spec must be > 0 when use_spectrum=True.")

        self.latent_dim = latent_dim
        self.use_xyz = use_xyz
        self.use_spectrum = use_spectrum
        self.k_spec = int(k_spec)
        self.log_input = bool(log_input)
        self.eps = float(eps)

        C_in = (3 if use_xyz else 0) + (self.k_spec if use_spectrum else 0)

        print(
            f"🧬 DiffusionEncoderXYZSpectrum | "
            f"Z={latent_dim}, C_in={C_in} "
            f"[XYZ={use_xyz} + Spectrum={use_spectrum} (k={self.k_spec})], "
            f"width={width}, blocks={n_blocks}"
        )

        self.encoder = DiffusionNet(
            C_in=C_in,
            C_out=latent_dim,
            C_width=width,
            N_block=n_blocks,
            with_gradient_features=True,
            dropout=0.0,
        )

        self.vertex_bottleneck = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, latent_dim),
        )

    def _spectrum_vector(self, evals: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """
        Returns spec ∈ R^{k_spec} using the SAME logic as the MLP baseline:
          ev = evals[1:]
          spec = ev[:k_spec] (pad with zeros if needed)
          if log_input: log(clamp_min(eps))
        """
        ev = evals.flatten()
        if ev.numel() <= 1:
            # pathological case: return zeros
            spec = torch.zeros(self.k_spec, device=ev.device, dtype=dtype)
            return spec

        ev = ev[1:].to(dtype)  # exclude lambda_0

        if ev.numel() >= self.k_spec:
            spec = ev[: self.k_spec]
        else:
            pad = torch.zeros(self.k_spec - ev.numel(), device=ev.device, dtype=dtype)
            spec = torch.cat([ev, pad], dim=0)

        if self.log_input:
            spec = torch.log(spec.clamp_min(self.eps))

        return spec

    def _spectrum_feature(self, evals: torch.Tensor, n_vertices: int, dtype: torch.dtype) -> torch.Tensor:
        """
        Returns spec_feat ∈ R^{N x k_spec} by replicating the global spectrum vector.
        """
        spec = self._spectrum_vector(evals, dtype=dtype)               # (k_spec,)
        spec_feat = spec.unsqueeze(0).expand(n_vertices, -1)           # (N, k_spec) view (no alloc)
        return spec_feat

    def forward(
        self,
        V: torch.Tensor,
        mass: torch.Tensor,
        L: torch.Tensor,
        evals: torch.Tensor,
        evecs: torch.Tensor,
        faces: torch.Tensor,
        gradX: torch.Tensor,
        gradY: torch.Tensor,
        return_per_vertex: bool = False,
        add_noise: bool = True,
    ):
        feats = []

        if self.use_xyz:
            feats.append(V)

        if self.use_spectrum:
            spec_feat = self._spectrum_feature(evals, n_vertices=V.shape[0], dtype=V.dtype)
            feats.append(spec_feat)

        x = torch.cat(feats, dim=1)

        Z_per_vertex = self.encoder(
            x,
            mass,
            L,
            evals,
            evecs,
            faces=faces,
            gradX=gradX,
            gradY=gradY,
        )

        Z_per_vertex = self.vertex_bottleneck(Z_per_vertex)

        if add_noise:
            Z_per_vertex = Z_per_vertex + 0.01 * torch.randn_like(Z_per_vertex)

        # SAME regime as old encoder
        Z_global = Z_per_vertex.mean(dim=0, keepdim=True)

        if return_per_vertex:
            return Z_per_vertex, Z_global
        return Z_global