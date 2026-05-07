import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple


def p2tri_distance(X: np.ndarray, Y: np.ndarray, pidx: np.ndarray) -> np.ndarray:
    """
    Computes point-to-triangle distances from points in X to corresponding triangles in Y,
    constructed using their three nearest neighbors.

    Parameters
    ----------
    X : (N, 3) ndarray
        Source mesh points.
    Y : (M, 3) ndarray
        Target mesh points.
    pidx : (N,) ndarray
        Indices mapping each source point to a target region.

    Returns
    -------
    distances : (N,) ndarray
        Euclidean distances from each source point to the corresponding triangle.
    """

    def _point_triangle_distance(P: np.ndarray, TRI: np.ndarray) -> float:
        """
        Computes the point-to-triangle distance from point P to triangle TRI.

        Parameters
        ----------
        P : (3,) ndarray
            Query point.
        TRI : (3, 3) ndarray
            Triangle vertices.

        Returns
        -------
        distance : float
            Minimum distance from point to triangle or its edges.
        """
        A, B, C = TRI[0], TRI[1], TRI[2]
        AB, AC, AP = B - A, C - A, P - A
        N = np.cross(AB, AC)
        N_norm = np.linalg.norm(N)

        def segment_distance(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
            AB = B - A
            t = np.dot(P - A, AB) / np.dot(AB, AB)
            t = np.clip(t, 0.0, 1.0)
            return np.linalg.norm(P - (A + t * AB))

        # Degenerate triangle
        if N_norm == 0:
            if np.linalg.norm(AB) > 0:
                return segment_distance(P, A, B)
            if np.linalg.norm(AC) > 0:
                return segment_distance(P, A, C)
            return np.linalg.norm(P - A)

        # Project point onto triangle plane
        N_unit = N / N_norm
        dist_to_plane = np.dot(AP, N_unit)
        P_proj = P - dist_to_plane * N_unit

        # Check if projection lies inside triangle using barycentric coordinates
        v0, v1, v2 = C - A, B - A, P_proj - A
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        denom = dot00 * dot11 - dot01 * dot01
        if denom == 0:
            return np.linalg.norm(P - A)  # Triangle collapsed

        inv_denom = 1.0 / denom
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        if (u >= 0) and (v >= 0) and (u + v <= 1):
            return abs(dist_to_plane)

        # Outside triangle: return distance to nearest segment
        if u < 0:
            return segment_distance(P, A, B)
        if v < 0:
            return segment_distance(P, A, C)
        if u + v > 1:
            return segment_distance(P, B, C)

        return np.linalg.norm(P - A)

    # Find 3 nearest neighbors on the target mesh
    Y_selected = Y[pidx]
    tree = cKDTree(Y_selected)
    _, neighbor_indices = tree.query(X, k=3)

    return np.array([
        _point_triangle_distance(X[i], Y_selected[neighbor_indices[i]])
        for i in range(X.shape[0])
    ])
