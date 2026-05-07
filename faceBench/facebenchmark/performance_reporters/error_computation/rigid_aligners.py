import numpy as np
import open3d as o3d


class BaseRigidAligner:
    def __init__(self, opts=None):
        self.opts = opts if opts else {}

    def align(self, X, Y, Xlmks=None, Ylmks=None):
        raise NotImplementedError("This class is a base class and must be implemented.")


def procrustes(X, Y, scaling=True, reflection="best", tol=1e-8):
    """
    Computes the Procrustes transformation that best aligns `Y` to `X`.

    Parameters
    ----------
    X : ndarray (N, M)
        Target matrix of shape (N points, M dimensions).
    Y : ndarray (N, M)
        Input matrix to be transformed.
    scaling : bool, optional
        If False, disables scaling and forces it to 1.
    reflection : {'best', True, False}, optional
        If 'best' (default), allows reflection only if it reduces error.
        If True, forces reflection. If False, prevents reflection.
    tol : float, optional
        Tolerance to avoid division errors in degenerate cases.

    Returns
    -------
    d : float
        Residual sum of squared errors after transformation.
    Z : ndarray (N, M)
        The transformed version of `Y` after Procrustes alignment.
    tform : dict
        Transformation parameters containing:
        - 'rotation': optimal rotation matrix
        - 'scale': optimal scaling factor
        - 'translation': translation vector.
    """
    if X.shape != Y.shape:
        raise ValueError(f"Shape mismatch: X {X.shape} and Y {Y.shape} must be the same.")

    muX, muY = X.mean(axis=0), Y.mean(axis=0)
    X0, Y0 = X - muX, Y - muY

    normX, normY = np.linalg.norm(X0), max(np.linalg.norm(Y0), tol)  # Avoid division by zero

    X0 /= normX
    Y0 /= normY

    # Optimal rotation matrix using SVD
    U, s, Vt = np.linalg.svd(X0.T @ Y0)
    V = Vt.T
    T = V @ U.T

    # Handle reflection if needed
    if reflection != "best" and (np.linalg.det(T) < 0) != reflection:
        V[:, -1] *= -1
        T = V @ U.T

    traceTA = s.sum()

    # Compute transformation parameters
    b = traceTA * normX / normY if scaling else 1
    d = 1 - traceTA ** 2 if scaling else 1 + (normY ** 2 / normX ** 2) - 2 * traceTA * (normY / normX)
    Z = b * (Y0 @ T) + muX
    c = muX - b * (muY @ T)

    return d, Z, {"rotation": T, "scale": b, "translation": c}


class LandmarkBasedRigidAligner(BaseRigidAligner):
    """
    Performs rigid alignment of a point cloud based on specified landmark correspondences.
    """

    def __init__(self, opts):
        default_opts = {"ref_lmk_indices": [13, 19, 28, 31, 37]}
        self.opts = {**default_opts, **opts}
        if "ref_lmk_indices" not in self.opts:
            raise ValueError("Missing required 'ref_lmk_indices' in options.")

    def align(self, X, Y, Xlmks, Ylmks):
        """
        Aligns X to Y using Procrustes analysis based on selected landmarks.

        Parameters
        ----------
        X : ndarray (N, 3)
            Source point cloud.
        Y : ndarray (M, 3)
            Target point cloud.
        Xlmks : ndarray (L, 3)
            Landmarks from the source.
        Ylmks : ndarray (L, 3)
            Landmarks from the target.

        Returns
        -------
        X_aligned : ndarray (N, 3)
            Transformed source point cloud.
        Xlmks_aligned : ndarray (L, 3)
            Transformed source landmarks.
        """
        ref_ix = self.opts["ref_lmk_indices"]
        _, _, tform = procrustes(Ylmks[ref_ix, :], Xlmks[ref_ix, :])

        X_aligned = tform["scale"] * (X @ tform["rotation"]) + tform["translation"]
        Xlmks_aligned = tform["scale"] * (Xlmks @ tform["rotation"]) + tform["translation"]

        return X_aligned, Xlmks_aligned


class ICPRigidAligner(BaseRigidAligner):
    """
    Performs rigid alignment using the Iterative Closest Point (ICP) algorithm.

    Parameters
    ----------
    opts : dict
        Options for the alignment, including:
        - 'icp_threshold': Distance threshold for ICP (automatically computed if None).
        - 'icp_method': Type of ICP ('point_to_point' or 'point_to_plane').

    """

    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.land_align = LandmarkBasedRigidAligner({"ref_lmk_indices": [13, 19, 28, 31, 37]})

    def align(self, source_points, target_points, source_lmks_points, target_lmks_points):
        """
        Aligns the source points to the target using ICP.

        Parameters
        ----------
        source_points : ndarray (N, 3)
            Source point cloud.
        target_points : ndarray (M, 3)
            Target point cloud.
        source_lmks_points : ndarray (L, 3)
            Landmark points from the source.
        target_lmks_points : ndarray (L, 3)
            Landmark points from the target.

        Returns
        -------
        transformed_points : ndarray (N, 3)
            Transformed source points after alignment.
        transformed_landmarks : ndarray (L, 3)
            Transformed source landmarks after alignment.
        """

        # Step 1: Landmark-based initial alignment
        source_points, source_lmks_points = self.land_align.align(source_points, target_points,
                                                                  source_lmks_points, target_lmks_points)

        # Convert to Open3D point cloud
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(source_points)

        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(target_points)

        source_lmks = o3d.geometry.PointCloud()
        source_lmks.points = o3d.utility.Vector3dVector(source_lmks_points)

        # Choose this number because from experimentation it gave the best results
        threshold = 1000  # Adjust the threshold as needed

        # Apply ICP
        icp_values = o3d.pipelines.registration.registration_icp(
            source, target, threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        # Apply transformation
        transformation = icp_values.transformation
        source_new = source.transform(transformation)
        source_new_lmks = source_lmks.transform(transformation)

        return np.asarray(source_new.points), np.asarray(source_new_lmks.points)