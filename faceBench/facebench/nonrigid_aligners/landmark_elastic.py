import numpy as np
import cvxpy as cp
from scipy.spatial import cKDTree
from typing import List


def landmark_elastic_align(
        R: np.ndarray,
        G: np.ndarray,
        Glmks: np.ndarray,
        lmk_indices: List[int],
        gamma: float = 1.0,
        sel_lmk_ids: List[int] = list(range(51))
) -> np.ndarray:
    """
    Landmark-based non-rigid elastic alignment.

    Parameters
    ----------
    R : np.ndarray (N, 3)
        Source mesh.
    G : np.ndarray (M, 3)
        Target mesh.
    Glmks : np.ndarray (L, 3)
        Target landmarks.
    lmk_indices : List[int]
        Indices of landmarks in the mesh.
    gamma : float
        Weighting power for spatial influence.
    sel_lmk_ids : List[int]
        Subset of landmark indices to use for constraint.

    Returns
    -------
    np.ndarray
        Aligned source mesh (non-rigidly deformed).
    """

    def _solve_elastic_opt(D: np.ndarray, Dl: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solves the constrained least-squares optimization problem for deformation weights.

        Parameters
        ----------
        D : np.ndarray
            Full distance matrix.
        Dl : np.ndarray
            Landmark-subset of distance matrix.
        b : np.ndarray
            Displacement constraints from source to target landmarks.

        Returns
        -------
        np.ndarray
            Optimal displacement weights.
        """
        np.random.seed(1907)
        bmax = np.max(np.abs(b))
        subsample_factor = max(2, min(10, D.shape[0] // 500))
        Dsub = D[::subsample_factor, :]

        x = cp.Variable(D.shape[1])
        objective = cp.Minimize(cp.sum_squares(Dl @ x - b))
        constraints = [cp.norm_inf(Dsub @ x) <= bmax]
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.SCS)
        except cp.error.SolverError:
            print("Clarabel failed. Falling back to SCS.")
            prob.solve(solver=cp.CLARABEL)
        return x.value

    lmk_indices = np.array(lmk_indices)

    # === Compute distance matrix
    tree = cKDTree(R, leafsize=10)
    Dx_new, indices = tree.query(R[lmk_indices], k=R.shape[0], eps=0, p=2)

    Dx_new_fixed = np.full((R.shape[0], len(lmk_indices)), np.inf)
    for i, lmk in enumerate(lmk_indices):
        Dx_new_fixed[indices[i], i] = Dx_new[i]
    Dx = Dx_new_fixed

    # === Normalize and apply gamma
    for j in range(Dx.shape[1]):
        Dx_max = Dx[:, j].max()
        if Dx_max == 0:
            Dx[:, j] = 0
        else:
            Dx[:, j] = 1 - Dx[:, j] / Dx_max
    Dx = Dx ** gamma
    Dy = Dz = Dx

    # === Landmark matrices
    Dxl = Dx[lmk_indices[sel_lmk_ids], :]
    Dyl = Dy[lmk_indices[sel_lmk_ids], :]
    Dzl = Dz[lmk_indices[sel_lmk_ids], :]

    bxl = Glmks[sel_lmk_ids, 0] - R[lmk_indices[sel_lmk_ids], 0]
    byl = Glmks[sel_lmk_ids, 1] - R[lmk_indices[sel_lmk_ids], 1]
    bzl = Glmks[sel_lmk_ids, 2] - R[lmk_indices[sel_lmk_ids], 2]

    dxu = _solve_elastic_opt(Dx, Dxl, bxl)
    dyu = _solve_elastic_opt(Dy, Dyl, byl)
    dzu = _solve_elastic_opt(Dz, Dzl, bzl)

    R2 = np.zeros_like(R)
    R2[:, 0] = R[:, 0] + Dx @ dxu
    R2[:, 1] = R[:, 1] + Dy @ dyu
    R2[:, 2] = R[:, 2] + Dz @ dzu

    return R2
