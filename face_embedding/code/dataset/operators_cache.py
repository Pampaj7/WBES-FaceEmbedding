import numpy as np
import torch
import trimesh
from pathlib import Path
from diffusion_net.geometry import normalize_positions, compute_operators


class OperatorCache:
    """
    Computes and caches DiffusionNet geometric operators (Laplacian, eigenvectors, etc.)
    for a given mesh and topology.

    This class ensures that expensive geometric computations are performed only once.
    When the requested operators already exist in the cache, they are loaded from disk;
    otherwise, they are computed using DiffusionNet’s geometry utilities and saved as .npz.

    Args:
        cache_dir (str or Path): Directory path where operator files will be stored.

    Example:
        >>> cache = OperatorCache("./cache_ops")
        >>> (mass, L, evals, evecs, gradX, gradY), verts, faces = cache.get_with_geo(
        ...     topo_name="BFM",
        ...     mesh_path="/path/to/mesh.obj",
        ...     k_eig=128
        ... )
    """

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Internal helper: map mesh path + topology → cache filename
    # --------------------------------------------------------------
    def _key2path(self, topo_name, mesh_path):
        """Return the cache file path for a given topology and mesh name."""
        base = f"{topo_name}__{Path(mesh_path).stem}.npz"
        return self.cache_dir / base

    # --------------------------------------------------------------
    # Main method: compute or load precomputed operators
    # --------------------------------------------------------------
    def get_with_geo(self, topo_name, mesh_path, k_eig):
        """
        Return precomputed operators along with normalized vertices and faces.

        Args:
            topo_name (str): Name of the topology (e.g., "BFM", "FLAME").
            mesh_path (str or Path): Path to the input mesh (.obj).
            k_eig (int): Number of Laplacian eigenfunctions to compute.

        Returns:
            tuple:
                - (mass, L, evals, evecs, gradX, gradY): torch.Tensors
                - verts (torch.Tensor): normalized vertices [N, 3]
                - faces (torch.Tensor): mesh faces [F, 3]
        """
        # Define cache path
        out = self._key2path(topo_name, mesh_path)

        # ----------------------------------------------------------
        # Step 1: Load mesh geometry and normalize vertex positions
        # ----------------------------------------------------------
        mesh = trimesh.load_mesh(mesh_path, process=False)
        verts = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.faces, dtype=torch.long)
        verts = normalize_positions(verts)

        # ----------------------------------------------------------
        # Step 2: Load from cache if available, otherwise compute
        # ----------------------------------------------------------
        if out.exists():
            # Load precomputed operators from disk
            data = np.load(out, allow_pickle=True)
            mass, L, evals, evecs, gradX, gradY = [
                torch.from_numpy(data[k]).float()
                for k in ["mass", "L", "evals", "evecs", "gradX", "gradY"]
            ]
        else:
            # Compute operators (slow, done only once)
            print(
                f"[OperatorCache] Computing operators for {Path(mesh_path).name} ...")
            frames, mass, L, evals, evecs, gradX, gradY = compute_operators(
                verts, faces, k_eig=k_eig
            )

            # Save to disk for future use
            np.savez(
                out,
                mass=mass.numpy(),
                L=L.numpy(),
                evals=evals.numpy(),
                evecs=evecs.numpy(),
                gradX=gradX.numpy(),
                gradY=gradY.numpy(),
            )
            print(f"[OperatorCache] Saved operators → {out.name}")

        # ----------------------------------------------------------
        # Step 3: Return everything (operators + geometry)
        # ----------------------------------------------------------
        return (mass, L, evals, evecs, gradX, gradY), verts, faces
