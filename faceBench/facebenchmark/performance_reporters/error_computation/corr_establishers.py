from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial import cKDTree


class BaseCorrespondenceEstablisher(ABC):
    """
    Abstract base class for correspondence establishment.
    """

    def __init__(self, opts):
        self.opts = opts

    @abstractmethod
    def establish(self, X, Y):
        """
        Establish correspondences between two sets of points.
        Must be implemented in subclasses.
        """
        pass


class ChamferCorrespondence(BaseCorrespondenceEstablisher):
    """
    Efficient nearest-neighbor correspondence using cKDTree.
    """

    def establish(self, X, Y):
        """
        Compute nearest correspondences between two meshes using cKDTree.
        """
        tree = cKDTree(Y)
        _, pidx = tree.query(X, k=1)
        return pidx.astype(int)


class IdentityCorrespondence(BaseCorrespondenceEstablisher):
    """
    Identity correspondence: assumes points in X already match Y.
    """

    def establish(self, X, Y):
        """
        Returns a direct index mapping assuming X and Y are already aligned.
        pidx : ndarray (N,)
            Indices mapping X to Y (identity mapping).
        """
        return np.arange(X.shape[0], dtype=int)
