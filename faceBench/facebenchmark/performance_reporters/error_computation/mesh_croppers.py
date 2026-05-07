import numpy as np


class BaseMeshCropper:
    """Abstract base class for mesh cropping operations."""

    def crop(self, X, Xlmks, Y=None, Ylmks=None):
        raise NotImplementedError("This class is abstract and must be subclassed.")


class PointBasedCropper(BaseMeshCropper):
    """
    Crops a 3D face mesh based on landmark distances.

    Parameters
    ----------
    opts : dict
        Options for cropping. Must contain:
        - 'dist_threshold_ratio': Threshold ratio for cropping.
        - 'ref_lmk_index': Index of the reference landmark.
        - 'leyec_index': Index of the left eye corner.
        - 'reyec_index': Index of the right eye corner.
    """

    def __init__(self, opts):
        self.opts = opts

        # Validate required options
        required_keys = ['dist_threshold_ratio', 'ref_lmk_index', 'leyec_index', 'reyec_index']
        for key in required_keys:
            if key not in opts:
                raise ValueError(f"Missing required option: {key}")

    def crop(self, X, Xlmks, Y=None, Ylmks=None):
        """
        Performs cropping based on the distance from a reference landmark.

        Parameters
        ----------
        X : ndarray, shape (N, 3)
            3D coordinates of the input mesh.
        Xlmks : ndarray, shape (L, 3)
            Landmarks of X.
        Y : ndarray, shape (M, 3), optional
            Target mesh (ignored in this implementation).
        Ylmks : ndarray, shape (M, 3), optional
            Landmarks of Y (ignored in this implementation).

        Returns
        -------
        ndarray
            Cropped mesh points.
        """

        # Extract options
        ref_ix = self.opts['ref_lmk_index']
        left_eye_ix = self.opts['leyec_index']
        right_eye_ix = self.opts['reyec_index']
        threshold = self.opts['dist_threshold_ratio']

        # Compute reference distance (Interpupillary Distance)
        iod = np.linalg.norm(Xlmks[right_eye_ix] - Xlmks[left_eye_ix])

        # Compute distances from the reference landmark
        dists = np.linalg.norm(X - Xlmks[ref_ix], axis=1)

        # Select points within the threshold
        mask = dists < (threshold * iod)
        return X[mask]