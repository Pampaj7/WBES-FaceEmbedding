import numpy as np
import scipy
from scipy.spatial import Delaunay, KDTree
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu, spsolve
from typing import Optional, Tuple

from .landmark_elastic import landmark_elastic_align


def nonrigid_icp_align(
        source_points: np.ndarray,
        target_points: np.ndarray,
        gamma: float = 1.0,
        alpha: float = 50.0,
        epsilon: float = 1.0,
        source_point_lmks: Optional[np.ndarray] = None,
        lmk_indices: Optional[np.ndarray] = None,
        prealign: bool = False,
) -> np.ndarray:
    """
    Performs non-rigid ICP alignment between source and target 3D meshes.

    Parameters
    ----------
    source_points : ndarray (N, 3)
        Source mesh vertices.
    target_points : ndarray (M, 3)
        Target mesh vertices.
    gamma : float
        Elasticity factor in the regularization term.
    alpha : float
        Regularization weight.
    epsilon : float
        Convergence threshold.
    source_point_lmks : Optional[ndarray (L, 3)]
        Landmarks of the source mesh (used if prealign=True).
    lmk_indices : Optional[ndarray]
        Indices of landmarks in the mesh (used if prealign=True).
    prealign : bool
        Whether to apply landmark-based elastic alignment before nonrigid ICP.

    Returns
    -------
    aligned_source : ndarray (N, 3)
        Deformed source points after non-rigid alignment.
    """
    if prealign:
        source_points = landmark_elastic_align(
            source_points, target_points, source_point_lmks, lmk_indices, gamma=gamma
        )

    source_tri = Delaunay(source_points[:, :2]).simplices.T
    source = source_points.T
    target = target_points.T

    G = np.diag([1.0, 1.0, 1.0, gamma]).astype(np.float32)
    M = _triangles_to_edge_vertex_adjacent_matrix(source_tri)
    A1 = scipy.sparse.kron(M, G)
    B1 = csc_matrix((A1.shape[0], 3), dtype=np.float32)

    cur_src = source
    cur_X = np.zeros((source.shape[1] * 4, 3))
    X = np.ones_like(cur_X)

    for decay in range(3):
        cur_alpha = alpha * np.exp(-0.5 * decay)
        eps = epsilon * (0.5 ** decay) * 6

        while np.linalg.norm(X - cur_X) > eps:
            cur_X = X.copy()

            nn_indices, nn_distances = _find_nearest_neighbors(cur_src, target)
            cur_nn_dst = target[:, nn_indices]

            D = _sparse_matrix_from_vertices(cur_src)
            weights = (nn_distances < max(np.mean(nn_distances) * 2, 1)).astype(np.float32)

            A2 = D.multiply(weights[:, np.newaxis])
            B2 = csc_matrix(np.multiply(cur_nn_dst.T, weights[:, np.newaxis]))

            A = scipy.sparse.vstack([A1.multiply(cur_alpha), A2])
            B = scipy.sparse.vstack([B1, B2])

            X = _spsolve_system(A, B)
            cur_src = (D @ X).T

    return cur_src.T


def _find_nearest_neighbors(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tree = KDTree(dst.T)
    dists, indices = tree.query(src.T, k=1, p=2)
    return indices.astype(np.int32), dists.astype(np.float32)


def _triangles_to_edge_vertex_adjacent_matrix(triangles: np.ndarray) -> csc_matrix:
    pairs = []
    for v1, v2, v3 in triangles.T:
        pairs += [tuple(sorted((v1, v2))), tuple(sorted((v1, v3))), tuple(sorted((v2, v3)))]

    pair_array = np.array(list(set(pairs)))
    edge_indices = np.arange(len(pair_array))

    i = np.concatenate([edge_indices, edge_indices])
    j = np.concatenate([pair_array[:, 0], pair_array[:, 1]])
    data = np.concatenate([np.ones(len(edge_indices)), -np.ones(len(edge_indices))])

    return csc_matrix((data, (i, j)), shape=(len(pair_array), np.max(triangles) + 1))


def _sparse_matrix_from_vertices(src: np.ndarray) -> csc_matrix:
    ones = np.ones((1, src.shape[1]), dtype=np.float32)
    src_hom = np.vstack([src, ones])

    Di = np.arange(src.shape[1])
    Dj = np.stack([Di * 4 + i for i in range(4)], axis=0).reshape(-1)
    Di = np.tile(Di, 4)
    values = src_hom.reshape(-1)

    return csc_matrix((values, (Di, Dj)), shape=(src.shape[1], src.shape[1] * 4), dtype=np.float32)


def _spsolve_system(A: csc_matrix, B: csc_matrix) -> np.ndarray:
    ATA = (A.T @ A).tocsc()
    LU = splu(ATA, diag_pivot_thresh=0)

    P1 = csc_matrix((np.ones(LU.L.shape[0]), (LU.perm_r, np.arange(LU.L.shape[0])))).T
    P2 = csc_matrix((np.ones(LU.L.shape[0]), (np.arange(LU.L.shape[0]), LU.perm_c))).T

    b_tilde = P1.T @ (A.T @ B)
    z = spsolve(LU.L, b_tilde)
    y = spsolve(LU.U, z)

    return (P2.T @ y).toarray()
