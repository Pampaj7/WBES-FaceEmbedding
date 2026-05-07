import numpy as np
from typing import List, Tuple, Literal


def landmark_based_align(
        X: np.ndarray,
        Y: np.ndarray,
        Xlmks: np.ndarray,
        Ylmks: np.ndarray,
        ref_lmk_indices: List[int] = [13, 19, 28, 31, 37]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aligns source mesh X to target mesh Y using Procrustes alignment on selected landmarks.

    Parameters
    ----------
    X : ndarray (N, 3)
        Source mesh vertices.
    Y : ndarray (M, 3)
        Target mesh vertices.
    Xlmks : ndarray (L, 3)
        Landmarks from the source mesh.
    Ylmks : ndarray (L, 3)
        Landmarks from the target mesh.
    ref_lmk_indices : list of int
        Indices of reference landmarks used for alignment.

    Returns
    -------
    X_aligned : ndarray (N, 3)
        Aligned source mesh.
    Xlmks_aligned : ndarray (L, 3)
        Aligned source landmarks.
    """
    b, R, t = _procrustes(Ylmks[ref_lmk_indices], Xlmks[ref_lmk_indices])
    X_aligned = b * (X @ R) + t
    Xlmks_aligned = b * (Xlmks @ R) + t
    return X_aligned, Xlmks_aligned


def _procrustes(
        X: np.ndarray,
        Y: np.ndarray,
        scaling: bool = True,
        reflection: Literal["best", True, False] = "best",
        tol: float = 1e-8
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes the optimal Procrustes transformation to align Y to X.

    Parameters
    ----------
    X : ndarray (N, 3)
        Target shape.
    Y : ndarray (N, 3)
        Source shape to be aligned.
    scaling : bool
        Whether to apply scaling.
    reflection : {'best', True, False}
        Whether to allow reflection.
    tol : float
        Numerical stability tolerance.

    Returns
    -------
    scale : float
        Optimal scale.
    rotation : ndarray (3, 3)
        Optimal rotation matrix.
    translation : ndarray (3,)
        Translation vector.
    """
    if X.shape != Y.shape:
        raise ValueError(f"Shape mismatch: X {X.shape} and Y {Y.shape} must be the same.")

    muX, muY = X.mean(axis=0), Y.mean(axis=0)
    X0, Y0 = X - muX, Y - muY

    normX = np.linalg.norm(X0)
    normY = max(np.linalg.norm(Y0), tol)

    X0 /= normX
    Y0 /= normY

    U, s, Vt = np.linalg.svd(X0.T @ Y0)
    V = Vt.T
    R = V @ U.T

    if reflection != "best" and (np.linalg.det(R) < 0) != reflection:
        V[:, -1] *= -1
        R = V @ U.T

    traceTA = s.sum()
    scale = traceTA * normX / normY if scaling else 1.0
    translation = muX - scale * (muY @ R)

    return scale, R, translation
