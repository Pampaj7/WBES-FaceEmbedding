import numpy as np


def p2p_distance(
        X: np.ndarray,
        Y: np.ndarray,
        pidx: np.ndarray
) -> np.ndarray:
    """
    Compute point-to-point Euclidean distance between corresponding vertices.

    Parameters
    ----------
    X : np.ndarray, shape (N, 3)
        Source mesh.
    Y : np.ndarray, shape (M, 3)
        Target mesh.
    pidx : np.ndarray, shape (N,)
        Indices mapping each point in X to a point in Y.

    Returns
    -------
    np.ndarray, shape (N,)
        Per-vertex Euclidean distances.

    Raises
    ------
    ValueError
        If dimensions do not match.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    pidx = np.asarray(pidx, dtype=int)

    if X.shape[0] != pidx.shape[0]:
        raise ValueError(f"Mismatch: X has {X.shape[0]} points, but pidx has {pidx.shape[0]} entries.")

    if Y.shape[0] <= np.max(pidx):
        raise ValueError("Index out of bounds: some entries in pidx exceed the number of vertices in Y.")

    return np.linalg.norm(X - Y[pidx], axis=1)
