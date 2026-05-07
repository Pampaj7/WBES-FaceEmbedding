import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import griddata
from typing import Optional, Tuple
from scipy.spatial import ConvexHull
from matplotlib.path import Path


def plot_face_heatmap(
        vertices: np.ndarray,
        errors: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        landmark_indices: Optional[Tuple[int, int]] = None,
        coef: int = 300,
        vmax: Optional[float] = None,
        cmap: str = "jet",
        title: Optional[str] = None,
        save_path: Optional[str] = None
):
    """
    Proietta i vertici su un piano 2D XY e visualizza una heatmap dell'errore.

    Parameters
    ----------
    vertices : (N, 3)
        Vertici 3D della mesh.
    errors : (N,)
        Valore di errore per ogni vertice.
    landmarks : (L, 3), optional
        Landmark della mesh per normalizzazione.
    landmark_indices : (int, int), optional
        Indici dei landmark occhi (sx, dx) per normalizzazione IOD.
    coef : int
        Risoluzione della griglia.
    vmax : float, optional
        Massimo valore per la colorbar.
    cmap : str
        Colormap da usare.
    title : str, optional
        Titolo della figura.
    save_path : str, optional
        Se fornito, salva l'immagine a questo path.
    """
    V = vertices.copy()
    err = errors.copy()

    if landmarks is not None and landmark_indices is not None:
        le_idx, re_idx = landmark_indices
        iod = np.linalg.norm(landmarks[le_idx] - landmarks[re_idx])
        V /= iod

    # Normalizza in [0, 1] e scala
    x = (V[:, 0] - V[:, 0].min()) / (V[:, 0].ptp() + 1e-8)
    y = (V[:, 1] - V[:, 1].min()) / (V[:, 1].ptp() + 1e-8)

    x = (x * coef).round().astype(int)
    y = (y * coef).round().astype(int)

    h, w = y.max() + 1, x.max() + 1
    H = np.full((h, w), np.nan)

    for xi, yi, ei in zip(x, y, err):
        H[yi, xi] = ei
    mask = np.isnan(H)
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    # Interpolazione con fallback
    try:
        H_interp = griddata(
            (xx[~mask], yy[~mask]), H[~mask], (xx[mask], yy[mask]),
            method="linear", fill_value=np.nan
        )
    except Exception:
        H_interp = griddata(
            (xx[~mask], yy[~mask]), H[~mask], (xx[mask], yy[mask]),
            method="nearest", fill_value=np.nan
        )
    H[mask] = H_interp

    points_2d = np.stack([x, y], axis=1)
    hull = ConvexHull(points_2d)
    hull_path = Path(points_2d[hull.vertices])
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    inside = hull_path.contains_points(grid_points).reshape(H.shape)
    H[~inside] = np.nan  # Blocca l’esterno

    # === Visualizzazione ===
    if vmax is None:
        vmax = np.nanpercentile(H, 95)  # Usa solo valori reali

    norm = colors.Normalize(vmin=0, vmax=vmax)
    plt.figure(figsize=(6, 6))
    im = plt.imshow(H, cmap=cmap, norm=norm)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0)
    else:
        plt.show()
