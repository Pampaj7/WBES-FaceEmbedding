import numpy as np


def landmark_distance(Xlmks: np.ndarray, Ylmks: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distances between corresponding landmark points.

    Parameters
    ----------
    Xlmks : np.ndarray, shape (L, 3)
        Landmark points from the source mesh.
    Ylmks : np.ndarray, shape (L, 3)
        Landmark points from the target mesh.

    Returns
    -------
    np.ndarray, shape (L,)
        Per-landmark Euclidean distances.

    Raises
    ------
    ValueError
        If the input arrays do not have the same shape.
    """
    Xlmks = np.asarray(Xlmks)
    Ylmks = np.asarray(Ylmks)

    if Xlmks.shape != Ylmks.shape:
        raise ValueError(f"Shape mismatch: Xlmks {Xlmks.shape}, Ylmks {Ylmks.shape}")

    return np.linalg.norm(Xlmks - Ylmks, axis=1)
