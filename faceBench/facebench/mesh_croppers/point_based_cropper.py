import numpy as np
from typing import Optional


def point_based_crop(
        X: np.ndarray,
        Xlmks: np.ndarray,
        dist_threshold_ratio: float = 1.0,
        ref_lmk_index: int = 28,
        leyec_index: int = 36,
        reyec_index: int = 45,
) -> np.ndarray:
    """
    Crops a 3D face mesh by retaining only the vertices within a certain distance from a reference point.
    The distance threshold is based on a ratio of the interpupillary distance.

    Parameters
    ----------
    X : (N, 3) ndarray
        Source mesh vertices.
    Xlmks : (L, 3) ndarray
        Landmark coordinates on the source mesh.
    dist_threshold_ratio : float
        Ratio of maximum distance allowed from the reference landmark, relative to interocular distance.
    ref_lmk_index : int
        Index of the landmark used as reference for cropping.
    leyec_index : int
        Index of the left eye corner landmark.
    reyec_index : int
        Index of the right eye corner landmark.

    Returns
    -------
    (K, 3) ndarray
        Cropped subset of source mesh vertices.
    """
    # Compute interocular distance
    iod = np.linalg.norm(Xlmks[reyec_index] - Xlmks[leyec_index])

    # Compute distance of each vertex from the reference landmark
    dists = np.linalg.norm(X - Xlmks[ref_lmk_index], axis=1)

    # Create mask to keep points within threshold
    threshold = dist_threshold_ratio * iod
    mask = dists < threshold

    return X[mask]
