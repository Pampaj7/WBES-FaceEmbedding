import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from typing import Literal, Tuple, List


def compute_landmark_base_vertex_weights(
        mm: dict,
        weight_power: Literal["square", "sqrt"],
        weight_strategy: Literal["mixed", "min", "mean"]
) -> np.ndarray:
    """
    Computes per-vertex weights for a face model based on landmark proximity and spatial distribution.

    Parameters
    ----------
    mm : dict
        Morphable model dictionary with 'mean_face_shape', 'leye_oc_rel_index', 'reye_oc_rel_index', and 'lmk_indices'.
    weight_power : {'square', 'sqrt'}
        Transformation to apply to final weights.
    weight_strategy : {'mixed', 'min', 'mean'}
        Strategy to combine distance-based weight components.

    Returns
    -------
    weights : ndarray
        Per-vertex weight array of shape (N,).
    """
    Xmean = np.asarray(mm['mean_face_shape'])
    lix = mm['leye_oc_rel_index']
    rix = mm['reye_oc_rel_index']
    lmk_indices = mm['lmk_indices']

    # Normalize mean shape by interocular distance
    iod = np.linalg.norm(Xmean[lmk_indices[lix]] - Xmean[lmk_indices[rix]])
    Xmean /= iod
    Xmean_lmks = Xmean[lmk_indices]

    # Compute minimum distances to landmarks
    adists = np.linalg.norm(Xmean_lmks[:, None, :] - Xmean[None, :, :], axis=2)
    adists = np.maximum(adists, 0.01)
    weights_min = 1.0 / adists.min(axis=0)

    # Compute deviation from reference landmark distances
    ref_lis = np.array([0, 2, 4, 5, 7, 9, 20, 21, 23, 24, 26, 27, 29, 30, 19, 22,
                        25, 28, 13, 14, 18, 31, 33, 34, 35, 37, 44, 45, 46, 39,
                        40, 41, 49, 48, 50])
    indices = np.array(lmk_indices)[ref_lis] if len(lmk_indices) == 51 else np.array(lmk_indices)
    adists2 = np.linalg.norm(Xmean[indices, None, :] - Xmean[None, :, :], axis=2)
    mean_dists = np.mean(adists2, axis=0)
    weights_mean = 1.0 / np.abs(mean_dists - 0.48)

    # Combine weights
    if weight_strategy == 'mixed':
        weights = (weights_min + weights_mean) / 2.0
    elif weight_strategy == 'min':
        weights = weights_min
    elif weight_strategy == 'mean':
        weights = weights_mean
    else:
        raise ValueError(f"Invalid weight_strategy: {weight_strategy}")

    weights = np.maximum(weights, 1.0)

    # Apply power transformation
    if weight_power == 'square':
        weights = (weights ** 2) / (np.mean(weights ** 2) / np.mean(weights))
    elif weight_power == 'sqrt':
        weights = np.sqrt(weights)
    else:
        raise ValueError(f"Invalid weight_power: {weight_power}")

    return weights


def sparse_cholesky(A: sparse.csc_matrix) -> sparse.csc_matrix:
    """Computes a sparse Cholesky-like factor using LU decomposition."""
    LU = splu(A, diag_pivot_thresh=0)
    return LU.L @ sparse.diags(np.sqrt(LU.U.diagonal()))


def build_cholseky_factor(ads: np.ndarray, b: float, N: int) -> Tuple[List[float], List[float]]:
    """Constructs a Cholesky factorization from diagonals and off-diagonals for a tridiagonal matrix."""
    diags = []
    off_diags = []
    sum2_off_diags = 0.0
    for i in range(N):
        diag_val = np.sqrt(ads[i] - sum2_off_diags)
        diags.append(diag_val)
        if i == N - 1:
            break
        off_val = (b - sum2_off_diags) / diag_val
        off_diags.append(off_val)
        sum2_off_diags += off_val * off_val
    return diags, off_diags


def solve_lower_tri(alphas: List[float], betas: List[float], b: np.ndarray) -> np.ndarray:
    """Solves L y = b where L is a lower-triangular matrix defined by alphas and betas."""
    N = len(alphas)
    y = np.zeros(N)
    prev_sum = 0.0
    for i in range(N):
        y[i] = (b[i] - prev_sum) / alphas[i]
        if i != N - 1:
            prev_sum += betas[i] * y[i]
    return y


def solve_upper_tri(alphas: List[float], betas: List[float], y: np.ndarray) -> np.ndarray:
    """Solves L^T x = y for upper-triangular matrix defined by alphas and betas."""
    N = len(alphas)
    x = np.zeros(N)
    x_sum = 0.0
    betas_extended = betas + [0.0]  # Ensure indexing is valid
    for i in reversed(range(N)):
        x[i] = (y[i] - x_sum * betas_extended[i]) / alphas[i]
        x_sum += x[i]
    return x
