import numpy as np
from scipy.spatial import cKDTree


def chamfer_correspondence(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Computes the nearest neighbor correspondence from each point in X to points in Y using a KD-tree.

    Parameters
    ----------
    X : np.ndarray, shape (N, 3)
        Source point cloud.
    Y : np.ndarray, shape (M, 3)
        Target point cloud.

    Returns
    -------
    np.ndarray, shape (N,)
        Indices of the nearest neighbors in Y for each point in X.
    """
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"Expected X shape (N, 3), got {X.shape}")
    if Y.ndim != 2 or Y.shape[1] != 3:
        raise ValueError(f"Expected Y shape (M, 3), got {Y.shape}")

    tree = cKDTree(Y)
    _, indices = tree.query(X, k=1)

    return indices.astype(int)
