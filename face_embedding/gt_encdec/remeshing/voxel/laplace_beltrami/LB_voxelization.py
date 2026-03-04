# lb_voxelizer.py
import torch


class LBVoxelizer:
    """
    Defines intrinsic regions (voxels) in Laplace–Beltrami spectral space.

    Each vertex is mapped to a coordinate in R^k using the first k eigenfunctions.
    The space is then discretized into bins, defining intrinsic regions.
    """

    def __init__(self, n_evecs=3, bins=4):
        self.n_evecs = n_evecs
        self.bins = bins

    def __call__(self, evecs: torch.Tensor, Z: torch.Tensor):
        """
        Args:
            evecs : (N, K) Laplace–Beltrami eigenvectors
            Z     : (N, D) per-vertex latent field

        Returns:
            pooled_latents : (B, D) pooled latent per intrinsic region
            mask           : (B,) whether region has vertices
        """
        assert evecs.shape[0] == Z.shape[0]

        # --- intrinsic coordinates ---
        Phi = evecs[:, :self.n_evecs]  # (N, k)

        # normalize per-eigenfunction (important!)
        Phi = (Phi - Phi.mean(dim=0)) / (Phi.std(dim=0) + 1e-8)

        # quantize
        coords = []
        for i in range(self.n_evecs):
            coords.append(
                torch.bucketize(
                    Phi[:, i],
                    torch.linspace(
                        Phi[:, i].min(),
                        Phi[:, i].max(),
                        self.bins + 1,
                        device=Phi.device,
                    )[1:-1],
                )
            )

        coords = torch.stack(coords, dim=1)  # (N, k)

        # flatten multi-dim voxel index
        voxel_id = torch.zeros(coords.shape[0], dtype=torch.long, device=Z.device)
        for i in range(self.n_evecs):
            voxel_id += coords[:, i] * (self.bins ** i)

        n_voxels = self.bins ** self.n_evecs

        # --- pooling ---
        D = Z.shape[1]
        pooled = torch.zeros(n_voxels, D, device=Z.device)
        counts = torch.zeros(n_voxels, device=Z.device)

        pooled.index_add_(0, voxel_id, Z)
        counts.index_add_(0, voxel_id, torch.ones_like(voxel_id, dtype=torch.float))

        mask = counts > 0
        pooled[mask] /= counts[mask][:, None]

        return pooled, mask