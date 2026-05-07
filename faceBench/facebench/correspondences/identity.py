import numpy as np


def identity_correspondence(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Returns a one-to-one correspondence assuming X and Y are already aligned and have the same number of points.

    Parameters
    ----------
    X : np.ndarray, shape (N, 3)
        Source mesh or point cloud.
    Y : np.ndarray, shape (N, 3)
        Target mesh or point cloud, assumed to be in correspondence with X.

    Returns
    -------
    np.ndarray, shape (N,)
        An array of indices [0, 1, ..., N-1] representing the identity correspondence.

    Raises
    ------
    ValueError
        If the number of points in X and Y does not match.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have the same number of points, got {X.shape[0]} and {Y.shape[0]}.")

    return np.arange(X.shape[0], dtype=int)
