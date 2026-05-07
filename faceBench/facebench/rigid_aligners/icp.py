import numpy as np
import open3d as o3d
from typing import List, Tuple, Optional
from facebench.rigid_aligners.landmark_based import landmark_based_align
from facebench.config import PrealignMethod


def icp_align(
        source_points: np.ndarray,
        target_points: np.ndarray,
        prealign: Optional[str] = "bbox",
        source_lmks: Optional[np.ndarray] = None,
        target_lmks: Optional[np.ndarray] = None,
        ref_lmk_indices: Optional[List[int]] = None,
        icp_threshold: float = 1000.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Perform rigid alignment of source_points to target_points using ICP.
    Optional pre-alignment using bounding box or landmark-based alignment.

    Parameters
    ----------
    source_points : (N, 3) ndarray
        Source mesh vertices.
    target_points : (M, 3) ndarray
        Target mesh vertices.
    prealign : PrealignMethod or None
        Pre-alignment strategy: 'bbox', 'landmark', or None.
    source_lmks : (L, 3) ndarray, optional
        Source landmarks.
    target_lmks : (L, 3) ndarray, optional
        Target landmarks.
    ref_lmk_indices : list of int, optional
        Landmark indices used for landmark-based pre-alignment.
    icp_threshold : float
        Maximum distance threshold for ICP.

    Returns
    -------
    aligned_points : (N, 3)
        Transformed source mesh.
    aligned_landmarks : (L, 3) or None
        Transformed landmarks if provided, else None.
    """
    aligned_src = source_points.copy()
    aligned_lmks = source_lmks.copy() if source_lmks is not None else None
    ref_lmk_indices = ref_lmk_indices or [13, 19, 28, 31, 37]

    # --- Optional pre-alignment ---
    if prealign == "landmark":
        if source_lmks is None or target_lmks is None:
            raise ValueError("Landmark pre-alignment requires both source_lmks and target_lmks.")
        aligned_src, aligned_lmks = landmark_based_align(
            aligned_src, target_points, source_lmks, target_lmks, ref_lmk_indices
        )

    elif prealign == "bbox":
        aligned_src = prealign_by_bbox(aligned_src, target_points)

    # --- Convert to Open3D PointCloud ---
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(aligned_src)

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points)

    if aligned_lmks is not None:
        lmks_pcd = o3d.geometry.PointCloud()
        lmks_pcd.points = o3d.utility.Vector3dVector(aligned_lmks)
    else:
        lmks_pcd = None

    # --- Run ICP ---
    result = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        icp_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    # --- Apply transformation ---
    transform = result.transformation
    source_pcd.transform(transform)
    if lmks_pcd:
        lmks_pcd.transform(transform)

    return np.asarray(source_pcd.points), (
        np.asarray(lmks_pcd.points) if lmks_pcd else None
    )


def prealign_by_bbox(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Roughly aligns source to target by centering and scaling using bounding box.

    Parameters
    ----------
    source : (N, 3)
        Source mesh.
    target : (M, 3)
        Target mesh.

    Returns
    -------
    aligned_source : (N, 3)
        Source mesh transformed to roughly align with target.
    """
    # Center
    source_centered = source - np.mean(source, axis=0)
    target_centered = target - np.mean(target, axis=0)

    # Scale
    source_scale = np.linalg.norm(source_centered, axis=1).max()
    target_scale = np.linalg.norm(target_centered, axis=1).max()

    source_scaled = source_centered / source_scale * target_scale

    return source_scaled + np.mean(target, axis=0)
