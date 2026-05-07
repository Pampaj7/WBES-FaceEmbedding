# facebench/cloud_quality/local_pointcloud_stats.py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def local_planarity_stats(points: np.ndarray, k: int = 20) -> dict:
    """
    Calcola la planarity locale stimata tramite PCA dei vicini.
    Planarity = lambda2 / (lambda1 + lambda2 + lambda3)

    Parameters:
        points (np.ndarray): (N, 3)
        k (int): numero di vicini da considerare

    Returns:
        dict: mean, std, min, max planarity
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
    _, indices = nbrs.kneighbors(points)

    planarities = []
    for i in range(points.shape[0]):
        neighbors = points[indices[i][1:]]  # exclude self
        pca = PCA(n_components=3).fit(neighbors)
        lambdas = pca.singular_values_ ** 2
        planarity = lambdas[1] / np.sum(lambdas)
        planarities.append(planarity)

    planarities = np.array(planarities)
    return {
        "mean_planarity": np.mean(planarities),
        "std_planarity": np.std(planarities),
        "min_planarity": np.min(planarities),
        "max_planarity": np.max(planarities),
    }


def local_density_stats(points: np.ndarray, radius: float = 0.01) -> dict:
    """
    Calcola la densità locale come numero di punti nel raggio.

    Parameters:
        points (np.ndarray): (N, 3)
        radius (float): raggio per il conteggio dei vicini

    Returns:
        dict: densità media, std, min, max
    """
    from scipy.spatial import KDTree
    tree = KDTree(points)
    densities = np.array([len(tree.query_ball_point(p, r=radius)) for p in points])
    return {
        "mean_density": np.mean(densities),
        "std_density": np.std(densities),
        "min_density": np.min(densities),
        "max_density": np.max(densities),
    }


def knn_distance_stats(points: np.ndarray, k: int = 10) -> dict:
    """
    Statistiche sulle distanze ai k-nearest neighbors.

    Returns:
        dict: mean, std, min, max distanza ai K vicini
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
    distances, _ = nbrs.kneighbors(points)
    distances = distances[:, 1:]  # exclude self
    return {
        "mean_knn_distance": np.mean(distances),
        "std_knn_distance": np.std(distances),
        "min_knn_distance": np.min(distances),
        "max_knn_distance": np.max(distances),
    }
