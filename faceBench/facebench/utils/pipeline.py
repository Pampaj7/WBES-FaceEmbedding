from typing import Optional
import numpy as np
import warnings
from facebench.config import (
    PipelineConfig,
)
from facebench.corrector.topology_consistency import topology_consistency_corrector
from facebench.correspondences.chamfer import chamfer_correspondence
from facebench.correspondences.identity import identity_correspondence
from facebench.distances.p2p import p2p_distance
from facebench.distances.p2tri import p2tri_distance
from facebench.mesh_croppers.point_based_cropper import point_based_crop
from facebench.nonrigid_aligners.landmark_elastic import landmark_elastic_align
from facebench.nonrigid_aligners.nonrigid_icp import nonrigid_icp_align
from facebench.rigid_aligners.icp import icp_align
from facebench.rigid_aligners.landmark_based import landmark_based_align
from typing import Tuple
from facebench.utils.warnings import warn_if_icp_no_prealign
from typing import Union
import facebench as fb


def run_pipeline(
        R: np.ndarray,
        G: np.ndarray,
        Rlmks: np.ndarray,
        Glmks: np.ndarray,
        mm: dict,
        config: PipelineConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Executes the complete FaceBench evaluation pipeline on a pair of meshes.

    This includes optional rigid and non-rigid alignment, correspondence estimation,
    topological correction, and final distance computation. The output is:
    - the per-vertex reconstruction error (after alignment and optional correction)
    - the aligned mesh used for evaluation.

    Parameters
    ----------
    R : np.ndarray, shape (N, 3)
        Reconstructed mesh vertices.
    G : np.ndarray, shape (M, 3)
        Ground-truth mesh vertices.
    Rlmks : np.ndarray, shape (L, 3)
        Landmarks on the reconstructed mesh.
    Glmks : np.ndarray, shape (L, 3)
        Landmarks on the ground-truth mesh.
    mm : dict
        Morphable model metadata, including landmark indices.
    config : PipelineConfig
        Configuration specifying which steps to apply (e.g., aligners, distance metric).

    Returns
    -------
    error : np.ndarray
        Per-vertex distance from aligned mesh to ground-truth (in meters).
    Rref : np.ndarray
        Final aligned mesh (after all alignments, ready for visualization).
    """

    # === Step 1: Mesh cropping (optional) ===
    # Removes irrelevant regions from the ground-truth mesh based on landmarks,
    # focusing the evaluation on a specific region (e.g., face only).
    if config.mesh_cropper:
        if config.mesh_cropper.method == "point_based":
            G = point_based_crop(
                G, Glmks,
                dist_threshold_ratio=config.mesh_cropper.dist_threshold_ratio,
                ref_lmk_index=config.mesh_cropper.ref_lmk_index,
                leyec_index=config.mesh_cropper.leyec_index,
                reyec_index=config.mesh_cropper.reyec_index
            )

    # === Step 2: Rigid alignment (optional) ===
    # Aligns R to G via landmarks or ICP (possibly both).
    if config.rigid_aligner:
        align_type = config.rigid_aligner.type
        prealign = config.rigid_aligner.prealign
        ref_lmk_indices = config.rigid_aligner.ref_lmk_indices or [13, 19, 28, 31, 37]

        if align_type == "icp":
            # Emit warning if no prealignment strategy defined
            warn_if_icp_no_prealign(prealign)

            R, Rlmks = icp_align(
                source_points=R,
                target_points=G,
                prealign=prealign,
                source_lmks=Rlmks,
                target_lmks=Glmks,
                ref_lmk_indices=ref_lmk_indices,
                icp_threshold=config.rigid_aligner.icp_threshold,
            )

        elif align_type == "landmark":
            R, Rlmks = landmark_based_align(
                R, G, Rlmks, Glmks, ref_lmk_indices=ref_lmk_indices
            )
    # === Step 3: Non-rigid alignment (optional) ===
    # Applies dense or landmark-driven deformation (e.g. Elastic or NICP).
    if config.nonrigid_aligner:
        method = config.nonrigid_aligner.type

        if method == "elastic":
            Rref = landmark_elastic_align(
                R.copy(),
                G,
                Glmks,
                lmk_indices=config.nonrigid_aligner.ref_lmk_indices or [13, 19, 28, 31, 37],
                gamma=config.nonrigid_aligner.gamma or 1.0,
                sel_lmk_ids=config.nonrigid_aligner.sel_lmk_ids or list(range(51)),
            )
        elif method == "nicp":
            Rref = nonrigid_icp_align(
                R.copy(),
                G,
                gamma=config.nonrigid_aligner.gamma or 1.0,
                alpha=config.nonrigid_aligner.alpha or 50.0,
                epsilon=config.nonrigid_aligner.epsilon or 1.0,
                source_point_lmks=Glmks,
                lmk_indices=config.nonrigid_aligner.ref_lmk_indices or [13, 19, 28, 31, 37],
                prealign=config.nonrigid_aligner.prealign or False,
            )
        else:
            raise ValueError(f"Unsupported nonrigid aligner: {method}")
    else:
        Rref = R  # fallback if no non-rigid alignment is used

    # === Step 4: Correspondence estimation ===
    # Maps points between Rref and G (e.g. via Chamfer or identity index).
    if config.corr_establisher:
        method = config.corr_establisher.type
        if method == "chamfer":
            pidx = chamfer_correspondence(Rref, G)
        elif method == "identity":
            pidx = identity_correspondence(Rref, G)
        else:
            raise ValueError(f"Unsupported correspondence method: {method}")
    else:
        warnings.warn("⚠️  Correspondence step skipped — using identity mapping.")
        pidx = identity_correspondence(Rref, G)

    # === Step 5: Topology correction (optional) ===
    # Smooths G to better match Rref when correspondence is noisy.
    if config.corrector:
        if config.corrector.type == "topology_consistency":
            correction_strategy = (
                str(config.corrector.correction_strategy.value)
                if config.corrector.correction_strategy else "pair"
            )
            weight_power = (
                str(config.corrector.weight_power.value)
                if config.corrector.weight_power else "sqrt"
            )
            weight_strategy = (
                str(config.corrector.weight_strategy.value)
                if config.corrector.weight_strategy else "mixed"
            )

            if np.any(pidx >= len(G)):
                warnings.warn("⚠️ Some correspondence indices exceed G shape — possible invalid matching.")

            G = topology_consistency_corrector(
                Rref,
                G,
                pidx,
                mm,
                correction_strategy=correction_strategy,
                weight_power=weight_power,
                weight_strategy=weight_strategy,
            )
        else:
            raise ValueError(f"Unsupported corrector: {config.corrector.type}")

    # === Step 6: Distance computation ===
    # Final step to compute error based on correspondences.
    dist_cfg = config.distance_computer
    if dist_cfg.type == "p2p":
        error = p2p_distance(R, G, pidx)
    elif dist_cfg.type == "p2tri":
        error = p2tri_distance(R, G, pidx)
    else:
        raise ValueError(f"Unknown distance computer type: {dist_cfg.type}")

    return error, Rref


def parse_pipeline_config(config: Union[PipelineConfig, dict]) -> PipelineConfig:
    if isinstance(config, PipelineConfig):
        return config

    if not isinstance(config, dict):
        raise ValueError("Config must be a PipelineConfig or a dict.")

    # Mapping per i tipi principali
    type_mapping = {
        "rigid": fb.RigidAlignerType.ICP,
        "landmark": fb.RigidAlignerType.LANDMARK,
        "elastic": fb.NonRigidAlignerType.ELASTIC,
        "nicp": fb.NonRigidAlignerType.NICP,
        "chamfer": fb.CorrEstablisherType.CHAMFER,
        "identity": fb.CorrEstablisherType.IDENTITY,
        "topology_consistency": fb.CorrectorType.TOPOLOGY_CONSISTENCY,
        "p2p": fb.DistanceComputerType.P2P,
        "p2tri": fb.DistanceComputerType.P2TRI,
    }

    # ========== BUILDING ==========
    # Rigid Aligner
    rigid_type_str = config.get("rigid_aligner")
    rigid_aligner = None
    if rigid_type_str:
        rigid_aligner = fb.RigidAlignerConfig(
            type=type_mapping[rigid_type_str],
            prealign=fb.PrealignMethod(config.get("rigid_prealign", fb.PrealignMethod.BBOX)),
            ref_lmk_indices=config.get("rigid_ref_lmk_indices"),
            icp_threshold=config.get("rigid_icp_threshold", 1000.0),
        )

    # NonRigid Aligner
    nonrigid_type_str = config.get("nonrigid_aligner")
    nonrigid_aligner = None
    if nonrigid_type_str:
        nonrigid_aligner = fb.NonRigidAlignerConfig(
            type=type_mapping[nonrigid_type_str],
            gamma=config.get("nonrigid_gamma", 1.0),
            sel_lmk_ids=config.get("nonrigid_sel_lmk_ids"),
            ref_lmk_indices=config.get("nonrigid_ref_lmk_indices"),
            alpha=config.get("nonrigid_alpha", 50.0),
            epsilon=config.get("nonrigid_epsilon", 1.0),
            prealign=config.get("nonrigid_prealign", False),
        )

    # Correspondence Establisher
    corr_type_str = config.get("corr_establisher")
    corr_establisher = None
    if corr_type_str:
        corr_establisher = fb.CorrEstablisherConfig(
            type=type_mapping[corr_type_str],
        )

    # Distance Computer
    dist_type_str = config.get("distance_computer")
    distance_computer = None
    if dist_type_str:
        distance_computer = fb.DistanceComputerConfig(
            type=type_mapping[dist_type_str],
        )

    # Corrector
    corrector_type_str = config.get("corrector")
    corrector = None
    if corrector_type_str:
        corrector = fb.CorrectorConfig(
            type=type_mapping[corrector_type_str],
            correction_strategy=fb.CorrectionStrategy(config.get("corrector_correction_strategy", fb.CorrectionStrategy.PAIR)),
            weight_power=fb.WeightPower(config.get("corrector_weight_power", fb.WeightPower.SQRT)),
            weight_strategy=fb.WeightStrategy(config.get("corrector_weight_strategy", fb.WeightStrategy.MIXED)),
        )

    # Mesh Cropper
    mesh_cropper = None
    if config.get("mesh_cropper"):
        mesh_cropper = fb.MeshCropperConfig(
            method=fb.MeshCropperType.POINT_BASED,
            dist_threshold_ratio=config.get("cropper_dist_threshold_ratio", 1.0),
            ref_lmk_index=config.get("cropper_ref_lmk_index", 28),
            leyec_index=config.get("cropper_leyec_index", 36),
            reyec_index=config.get("cropper_reyec_index", 45),
        )

    return PipelineConfig(
        mesh_cropper=mesh_cropper,
        rigid_aligner=rigid_aligner,
        nonrigid_aligner=nonrigid_aligner,
        corr_establisher=corr_establisher,
        corrector=corrector,
        distance_computer=distance_computer,
    )