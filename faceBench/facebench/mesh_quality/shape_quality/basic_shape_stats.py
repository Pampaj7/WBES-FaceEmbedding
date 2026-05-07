# facebench/mesh_quality/shape_quality/basic_shape_stats.py
import numpy as np
from sklearn.decomposition import PCA


def bounding_box_ratios(points: np.ndarray) -> dict:
    """
    Calcola le dimensioni dell'asse bounding box e i rapporti tra gli assi.

    Returns:
        dict: lunghezze x/y/z, rapporti tra assi
    """
    extent = points.max(axis=0) - points.min(axis=0)
    return {
        "bbox_x": extent[0],
        "bbox_y": extent[1],
        "bbox_z": extent[2],
        "bbox_xy_ratio": extent[0] / extent[1] if extent[1] != 0 else np.inf,
        "bbox_xz_ratio": extent[0] / extent[2] if extent[2] != 0 else np.inf,
        "bbox_yz_ratio": extent[1] / extent[2] if extent[2] != 0 else np.inf,
    }


def pca_shape_energy(points: np.ndarray) -> dict:
    """
    Calcola la distribuzione della varianza PCA sugli assi principali.

    Returns:
        dict: proporzioni di varianza spiegata dai primi 3 componenti
    """
    pca = PCA(n_components=3)
    pca.fit(points)
    var = pca.explained_variance_ratio_
    return {
        "pca_var1": var[0],
        "pca_var2": var[1],
        "pca_var3": var[2],
        "pca_drop12": var[0] - var[1],
        "pca_drop23": var[1] - var[2],
    }


def shape_spread_stats(points: np.ndarray) -> dict:
    """
    Calcola la deviazione standard su ciascun asse.

    Returns:
        dict: std x, y, z
    """
    return {
        "spread_x": np.std(points[:, 0]),
        "spread_y": np.std(points[:, 1]),
        "spread_z": np.std(points[:, 2])
    }
