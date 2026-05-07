import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import linalg as splinalg
from scipy import sparse
from facebenchmark import compute_landmark_base_vertex_weights


class BaseCorrector():
    def __init__(self, opts):
        self.opts = opts

    def align(self, R, G):
        raise NotImplementedError("This class is not implemented (a virtual method has been called)")


class TopologyConsistencyCorrector(BaseCorrector):
    def __init__(self, opts, mm):
        super().__init__(opts)
        default_opts = {
            "correction_strategy": "pair",
            "weight_power": "sqrt",
            "weight_strategy": "mixed"
        }
        self.opts = {**default_opts, **opts}
        self.weights = compute_landmark_base_vertex_weights(mm, self.opts['weight_power'], self.opts['weight_strategy'])
        self.lmk_indices = mm['lmk_indices']

    def correct(self, X, Y):
        """
        Corrects the solution X based on the target Y for each dimension using a chosen correction strategy.
        Returns an array dG of updates for each vertex.
        """
        N = Y.shape[0]
        updates = []

        for dim in range(3):
            if self.opts['correction_strategy'] == 'trace':
                # "Trace" strategy: fully vectorized operations.
                r = X[:, dim].reshape(-1, 1)
                g = Y[:, dim].reshape(-1, 1)
                e = np.ones(r.shape)
                gr = g - r
                r_new = e * np.sum(gr) - N * gr
                rT = r_new.T
                a = 2 * (N - 1)
                b_val = -2  # constant value for off-diagonals
                ads = self.weights * N * 2 + a  # modified diagonal vector
                (diags, off_diags) = build_cholseky_factor(ads, b_val, N)
                c = -2 * rT
                y_sol = solve_lower_tri(diags, off_diags, -c.T)
                dx_star = solve_upper_tri(diags, off_diags, y_sol)
                updates.append(dx_star)
            elif self.opts['correction_strategy'] == 'pair':
                # "Pair" strategy: sort and compute differences using vectorized operations.
                r0 = X[:, dim]
                g0 = Y[:, dim]
                # Get sort indices and sort r0 and g0 accordingly.
                rix = np.argsort(r0)
                r = r0[rix].copy()
                g = g0[rix].copy()

                c = np.empty(N)
                # Vectorized computation for interior vertices.
                c[1:-1] = -(2 * g[1:-1] - g[:-2] - g[2:] - 2 * r[1:-1] + r[:-2] + r[2:])
                # Compute boundary conditions explicitly.
                c[0] = -(g[0] - g[1] - (r[0] - r[1]))
                c[-1] = -(g[-1] - g[-2] - (r[-1] - r[-2]))

                # Build the tridiagonal matrix A in sparse CSC format.
                d1 = -np.ones(N)
                d0 = 2 * np.ones(N) + 2 * (self.weights[rix])
                A = (1. / 2) * sparse.spdiags([d1, d0, d1], [-1, 0, 1], N, N, format='csc')
                L = sparse_cholesky(A).T
                b_sol = spsolve(L.T, c)
                y_sol = spsolve(L, b_sol)
                # Reorder the solution back to the original order.
                irix = np.empty_like(rix)
                irix[rix] = np.arange(rix.size)
                x_corr = y_sol[irix]
                updates.append(x_corr)

        dG = np.array(updates).T  # Each column corresponds to an update in one dimension
        return dG


def sparse_cholesky(A):
    """
    Performs a sparse Cholesky decomposition using splu on A.
    The input matrix A must be sparse, symmetric, and positive definite.

    Returns
    -------
    A sparse matrix representing the factor L multiplied by the square root of U's diagonal.
    """
    n = A.shape[0]
    LU = splinalg.splu(A, diag_pivot_thresh=0)
    # Multiply LU.L by diag(U_ii^0.5)
    return LU.L.dot(sparse.diags(LU.U.diagonal() ** 0.5))


def build_cholseky_factor(ads, b, N):
    """
    Builds the diagonal and off-diagonal factors for a recursive Cholesky factorization.

    Parameters
    ----------
    ads : ndarray, shape (N,)
        Vector of modified diagonal terms.
    b : scalar
        Constant used for the off-diagonals.
    N : int
        System dimension.

    Returns
    -------
    (diags, off_diags) : tuple of lists
        'diags' contains the diagonal factors; 'off_diags' contains the off-diagonal factors.
    """
    diags = []
    off_diags = []
    sum2_off_diags = 0
    for i in range(N):
        diags.append(np.sqrt(ads[i] - sum2_off_diags))
        if i == N - 1:
            break
        off_val = (b - sum2_off_diags) / diags[-1]
        off_diags.append(off_val)
        sum2_off_diags += off_val * off_val
    return (diags, off_diags)


def solve_lower_tri(alphas, betas, b):
    """
    Solves a lower-triangular linear system with a recursive structure.

    Parameters
    ----------
    alphas : list of floats
        Diagonal entries.
    betas : list of floats
        Off-diagonal entries.
    b : ndarray, shape (N, 1)
        Right-hand side.

    Returns
    -------
    y : ndarray, shape (N,)
        The solution to the lower-triangular system.
    """
    N = len(alphas)
    y = np.zeros(N)
    prevsum = 0
    for i in range(N):
        y[i] = (b[i] - prevsum) / alphas[i]
        if i != N - 1:
            prevsum += betas[i] * y[i]
    return y


def solve_upper_tri(alphas, betas, y):
    """
    Solves an upper-triangular linear system with a recursive structure.

    Parameters
    ----------
    alphas : list of floats
        Diagonal entries.
    betas : list of floats
        Off-diagonal coefficients.
    y : ndarray, shape (N,)
        Right-hand side obtained from solving the lower-triangular system.

    Returns
    -------
    x : ndarray, shape (N,)
        The solution to the upper-triangular system.
    """
    N = len(alphas)
    x = np.zeros(N)
    xsum = 0
    # Extend betas with a 0 for simplicity in the loop.
    betas_extended = betas + [0]
    for i in reversed(range(N)):
        x[i] = (y[i] - xsum * betas_extended[i]) / alphas[i]
        xsum += x[i]
    return x