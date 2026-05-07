import numpy as np


def suggest_radius(points: np.ndarray, factor: float = 0.01) -> float:
    """
    Suggest a radius for neighborhood queries based on the point cloud's bounding box diagonal.

    Parameters:
        points (np.ndarray): (N, 3) point cloud
        factor (float): fraction of the diagonal length to use as radius (default 1%)

    Returns:
        float: suggested radius
    """
    diag = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    return diag * factor


def suggest_density_based_radius(points: np.ndarray, k: int = 8) -> float:
    """
    Suggest a radius based on average k-nearest neighbor distances.

    Parameters:
        points (np.ndarray): (N, 3) point cloud
        k (int): number of neighbors to consider

    Returns:
        float: average distance to k-th nearest neighbor
    """
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
    distances, _ = nbrs.kneighbors(points)
    # Exclude the point itself
    avg_dist = np.mean(distances[:, 1:])
    return avg_dist
