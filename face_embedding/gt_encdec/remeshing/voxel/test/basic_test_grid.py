import torch

class SpatialGrid:
    def __init__(
        self,
        grid_size=8,
        bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
    ):
        self.grid_size = grid_size
        self.bounds_min = torch.tensor(bounds[0])
        self.bounds_max = torch.tensor(bounds[1])

    def _vertex_to_cell(self, V):
        """
        V: [N, 3] in canonical space
        returns: linear cell index [N]
        """
        # Porta bounds sullo stesso device di V
        bounds_min = self.bounds_min.to(V.device)
        bounds_max = self.bounds_max.to(V.device)

        # Normalize to [0, 1]
        Vn = (V - bounds_min) / (bounds_max - bounds_min)
        Vn = torch.clamp(Vn, 0.0, 0.999999)

        idx = (Vn * self.grid_size).long()  # [N, 3]
        ix, iy, iz = idx[:, 0], idx[:, 1], idx[:, 2]

        lin_idx = ix * self.grid_size**2 + iy * self.grid_size + iz
        return lin_idx


    def __call__(self, V, feats):
        """
        V:     [N, 3]
        feats: [N, D]

        returns:
            grid_feats: [K, D]
            mask:       [K]  (True if cell has at least one vertex)
        """
        assert V.ndim == 2 and feats.ndim == 2
        assert V.shape[0] == feats.shape[0]

        device = V.device
        D = feats.shape[1]
        K = self.grid_size ** 3

        lin_idx = self._vertex_to_cell(V)

        grid_feats = torch.zeros((K, D), device=device)
        counts = torch.zeros((K,), device=device)

        # Accumulate
        grid_feats.index_add_(0, lin_idx, feats)
        counts.index_add_(0, lin_idx, torch.ones_like(lin_idx, dtype=torch.float))

        # Mean pooling
        mask = counts > 0
        grid_feats[mask] = grid_feats[mask] / counts[mask].unsqueeze(1)

        return grid_feats, mask
