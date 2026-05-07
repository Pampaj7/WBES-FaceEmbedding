import numpy as np
from scipy.spatial import distance, cKDTree
from tqdm import tqdm


class BaseDistanceComputer:
    """
    Abstract base class for distance computation between 3D meshes.
    """

    def __init__(self, opts=None):
        """
        Initialize with optional configuration parameters.

        Parameters
        ----------
        opts : dict, optional
            Configuration options for the distance computation.
        """
        self.opts = opts if opts is not None else {}

    def compute(self, X, Y, pidx=None, Xlmks=None, Ylmks=None):
        """
        Compute distance between two sets of points.
        Must be implemented in subclasses.

        Parameters
        ----------
        X : ndarray
            Source points (N x 3).
        Y : ndarray
            Target points (M x 3).
        pidx : ndarray, optional
            Indices of Y corresponding to each point in X.
        Xlmks : ndarray, optional
            Landmarks of X (L x 3).
        Ylmks : ndarray, optional
            Landmarks of Y (L x 3).

        Returns
        -------
        ndarray
            Distance values per point.
        """
        raise NotImplementedError("This method must be implemented in a subclass.")


class DenseP2PDistance(BaseDistanceComputer):
    """
    Computes point-to-point Euclidean distance between corresponding points.
    """

    def compute(self, X, Y, pidx, Xlmks=None, Ylmks=None):
        """
        Compute Euclidean distances between corresponding points.

        Parameters
        ----------
        X : ndarray (N x 3)
            Source mesh.
        Y : ndarray (M x 3)
            Target mesh.
        pidx : ndarray (N,)
            Indices mapping X to Y.

        Returns
        -------
        ndarray (N,)
            Euclidean distances per vertex.
        """
        pidx = np.asarray(pidx, dtype=int)  # Ensure pidx is a NumPy array
        return np.linalg.norm(X - Y[pidx], axis=1)  # Optimized Euclidean distance


def pointTriangleDistance(P, TRI):
    """
    Computes the shortest Euclidean distance between a point `P` and a given triangle `TRI` in 3D space.

    This function calculates the minimum distance between a given point and a triangle using the following steps:
    1. **Project the point onto the triangle’s plane.**
    2. **Check if the projection is inside the triangle using barycentric coordinates.**
    3. **If the projection is outside, compute the distance to the closest edge or vertex.**

    Parameters
    ----------
    P : ndarray, shape (3,)
        The query point in 3D space.
    TRI : ndarray, shape (3,3)
        The 3D coordinates of the three vertices of the triangle, each row representing a vertex.

    Returns
    -------
    float
        The shortest Euclidean distance from `P` to the triangle.

    Examples
    --------
    #>>> P = np.array([0.5, 0.5, 1.0])
    #>>> TRI = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    #>>> pointTriangleDistance(P, TRI)
    #1.0

    #>>> P = np.array([0.3, 0.3, 0.0])
    #>>> TRI = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    #>>> pointTriangleDistance(P, TRI)
    0.0  # The point lies inside the triangle

    Notes
    -----
    - If the projection of `P` is inside the triangle, the distance is the perpendicular distance to the plane.
    - If the projection is outside, the function finds the minimum distance to the edges and vertices.
    - If the triangle is degenerate (zero area), it returns the distance to the closest vertex.

    References
    ----------
    - David Eberly, "Distance Between Point and Triangle in 3D," Geometric Tools, LLC (1999).
      http://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
    -His version was obsoletete, this one is a more modern version of the same algorithm
    """

    # Extract triangle vertices
    A, B, C = TRI[0], TRI[1], TRI[2]

    # Compute triangle edge vectors
    AB = B - A
    AC = C - A
    AP = P - A

    # Compute the normal to the triangle plane
    N = np.cross(AB, AC)
    N_norm = np.linalg.norm(N)

    def pointLineDistance(P, A, B):
        """ Returns the perpendicular distance from P to the line passing through A and B. """
        AB = B - A
        return np.linalg.norm(np.cross(P - A, AB)) / np.linalg.norm(AB)

    # If the triangle is degenerate (collinear or nearly zero area)
    if N_norm == 0:
        if np.linalg.norm(AB) > 0:
            return pointLineDistance(P, A, B)
        elif np.linalg.norm(AC) > 0:
            return pointLineDistance(P, A, C)
        else:
            # all vertices are the same
            return np.linalg.norm(P - A)

    # Normalize the normal vector
    N = N / N_norm

    # Project the point onto the triangle's plane
    dist_to_plane = np.dot(AP, N)
    P_proj = P - dist_to_plane * N  # The projected point on the plane

    # Check if the projection lies inside the triangle using barycentric coordinates
    # (see: https://en.wikipedia.org/wiki/Barycentric_coordinate_system)
    v0 = C - A
    v1 = B - A
    v2 = P_proj - A

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    if (u >= 0) and (v >= 0) and (u + v <= 1):  # w=1-u-v
        return abs(dist_to_plane)  # If inside, return perpendicular distance

    # If the projection is outside the triangle, compute the minimum distance to the edges and vertices
    def segmentDistance(P, A, B):
        """
        Computes the shortest Euclidean distance between a point `P` and a line segment `AB`.

        Parameters
        ----------
        P : ndarray, shape (3,)
            The query point.
        A, B : ndarray, shape (3,)
            The endpoints of the line segment.

        Returns
        -------
        float
            The minimum distance from `P` to the segment `AB`.
        """
        AB = B - A
        t = np.dot(P - A, AB) / np.dot(AB, AB)
        t = np.clip(t, 0, 1)  # Clamp `t` to the segment bounds
        closest = A + t * AB  # Compute the closest point on the segment
        return np.linalg.norm(P - closest)  # Return the distance

    if u < 0:
        return segmentDistance(P, A, B)
    if v < 0:
        return segmentDistance(P, A, C)
    if u + v > 1:
        return segmentDistance(P, B, C)

    return np.linalg.norm(P - A)


def pointPlaneDistance(P, TRI):
    """
    Computes the shortest Euclidean distance from a point P to the plane
    defined by a triangle TRI in 3D space.

    Parameters
    ----------
    P : ndarray, shape (3,)
        The query point in 3D space.
    TRI : ndarray, shape (3,3)
        The 3D coordinates of the three vertices of the triangle, each row representing a vertex.

    Returns
    -------
    float
        The absolute shortest Euclidean distance from `P` to the plane extending from the triangle.

    Examples
    --------
    # P = np.array([0.5, 0.5, 1.0])
    # TRI = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    # pointPlaneDistance(P, TRI)
    1.0
    """

    # Extract triangle vertices
    A, B, C = TRI[0], TRI[1], TRI[2]

    # Compute triangle normal
    N = np.cross(B - A, C - A)
    N_norm = np.linalg.norm(N)

    def pointLineDistance(P, A, B):
        """
        Computes the shortest distance from a point `P` to the infinite line passing through `A` and `B`.
        """
        AB = B - A
        if np.linalg.norm(AB) == 0:  # If A and B are almost the same, return point-to-point distance
            return np.linalg.norm(P - A)
        return np.linalg.norm(np.cross(P - A, AB)) / np.linalg.norm(AB)

    # **Check for degenerate cases**
    if N_norm == 0:
        if np.linalg.norm(B - A) == 0:
            return pointLineDistance(P, A, B)  # Triangle is a line
        elif np.linalg.norm(C - A) == 0:
            return pointLineDistance(P, A, C)  # Triangle is a line
        else:
            return np.linalg.norm(P - A)  # Triangle is a single point

    # Normalize the normal
    N = N / N_norm

    # Compute the perpendicular distance from P to the plane
    distance = np.dot(P - A, N)

    return abs(distance)  # Return absolute distance


class DenseP2TriDistance(BaseDistanceComputer):
    """
    Computes point-to-triangle Euclidean distances by selecting the three nearest neighbors.
    """

    def compute(self, X, Y, pidx, Xlmks=None, Ylmks=None):
        """
        Compute distances from points in X to their nearest triangle in Y.

        Parameters
        ----------
        X : ndarray (N x 3)
            Source mesh.
        Y : ndarray (M x 3)
            Target mesh.
        pidx : ndarray (N,)
            Indices mapping X to Y.

        Returns
        -------
        ndarray (N,)
            Point-to-triangle distances.
        """
        N = X.shape[0]
        Yy = Y[pidx]

        # Use KDTree for efficient neighbor search
        tree = cKDTree(Yy)
        _, tidx = tree.query(X, k=3)  # Get 3 nearest neighbors for each point in X

        # Compute distances efficiently using vectorized operations
        return np.array([
            #pointPlaneDistance(X[i], Yy[tidx[i]])
            pointTriangleDistance(X[i], Yy[tidx[i]])
            for i in tqdm(range(N), desc="Computing P2Tri distances", unit="point")
        ])


class LandmarksDistance(BaseDistanceComputer):
    """
    Computes Euclidean distance between corresponding landmark points.
    """

    def compute(self, X, Y, pidx, Xlmks, Ylmks):
        """
        Compute Euclidean distances between corresponding landmarks.

        Parameters
        ----------
        X : ndarray (N x 3)
            Source mesh (not used).
        Y : ndarray (M x 3)
            Target mesh (not used).
        pidx : ndarray
            Indices of correspondence (not used).
        Xlmks : ndarray (L x 3)
            Landmark points from the source mesh.
        Ylmks : ndarray (L x 3)
            Landmark points from the target mesh.

        Returns
        -------
        ndarray (L,)
            Euclidean distances for each landmark pair.
        """
        Xlmks, Ylmks = np.asarray(Xlmks), np.asarray(Ylmks)  # Ensure NumPy arrays
        return np.linalg.norm(Xlmks - Ylmks, axis=1)  # Vectorized distance calculation
