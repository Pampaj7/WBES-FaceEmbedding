import numpy as np
from scipy.sparse import spdiags, csc_matrix
from scipy.sparse.linalg import spsolve
from typing import Dict, Literal

from .topology_utils import (
    build_cholseky_factor,
    solve_lower_tri,
    solve_upper_tri,
    sparse_cholesky,
    compute_landmark_base_vertex_weights
)


def topology_consistency_corrector(
    X: np.ndarray,
    G: np.ndarray,
    corr: np.ndarray,
    mm: Dict,
    correction_strategy: Literal["pair", "trace"] = "pair",
    weight_power: Literal["sqrt", "square"] = "sqrt",
    weight_strategy: Literal["mixed", "min", "mean"] = "mixed"
) -> np.ndarray:
    """
    Applies a topology-aware correction to the mesh G, modifying it based on reference X.

    Parameters
    ----------
    X : (N, 3) np.ndarray
        Aligned reconstruction mesh.
    G : (M, 3) np.ndarray
        Ground-truth mesh to be corrected.
    corr : (N,) np.ndarray
        Correspondence indices mapping each point in X to G.
    mm : dict
        Morphable model metadata.
    correction_strategy : str
        Correction method: 'pair' or 'trace'.
    weight_power : str
        Weighting power: 'sqrt' or 'square'.
    weight_strategy : str
        Landmark weight aggregation strategy: 'mixed', 'min', or 'mean'.

    Returns
    -------
    np.ndarray
        Corrected version of G (same shape as G).
    """
    weights = compute_landmark_base_vertex_weights(mm, weight_power, weight_strategy)
    N = X.shape[0]
    corrected_dims = []

    Y = G[corr]  # Subset of G to correct

    for dim in range(3):
        r = X[:, dim]
        g = Y[:, dim]

        if correction_strategy == "trace":
            r = r.reshape(-1, 1)
            g = g.reshape(-1, 1)
            error = g - r
            r_new = np.ones_like(r) * np.sum(error) - N * error
            a = 2 * (N - 1)
            b_val = -2
            diag = weights * N * 2 + a
            diags_, off_diags = build_cholseky_factor(diag, b_val, N)
            rhs = -2 * r_new.T
            y = solve_lower_tri(diags_, off_diags, -rhs)
            correction = solve_upper_tri(diags_, off_diags, y)
            corrected_dims.append(correction)

        elif correction_strategy == "pair":
            rix = np.argsort(r)
            r_sorted = r[rix]
            g_sorted = g[rix]

            c = np.empty(N)
            c[1:-1] = -(2 * g_sorted[1:-1] - g_sorted[:-2] - g_sorted[2:]
                        - 2 * r_sorted[1:-1] + r_sorted[:-2] + r_sorted[2:])
            c[0] = -(g_sorted[0] - g_sorted[1] - (r_sorted[0] - r_sorted[1]))
            c[-1] = -(g_sorted[-1] - g_sorted[-2] - (r_sorted[-1] - r_sorted[-2]))

            d0 = 2 * np.ones(N) + 2 * weights[rix]
            d1 = -np.ones(N)
            A: csc_matrix = 0.5 * spdiags([d1, d0, d1], [-1, 0, 1], N, N, format='csc')

            L = sparse_cholesky(A).T
            y = spsolve(L.T, c)
            correction = spsolve(L, y)

            irix = np.empty_like(rix)
            irix[rix] = np.arange(N)
            corrected_dims.append(correction[irix])

        else:
            raise ValueError(f"Invalid correction strategy: {correction_strategy}")

    dG = np.stack(corrected_dims, axis=1)

    # Apply the correction directly to a copy of G
    G_corrected = G.copy()
    G_corrected[corr] -= dG
    return G_corrected
