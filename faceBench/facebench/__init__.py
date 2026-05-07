# Distances
from .distances.p2p import p2p_distance
from .distances.p2tri import p2tri_distance
from .distances.landmarks import landmark_distance

# Correspondences
from .correspondences.chamfer import chamfer_correspondence
from .correspondences.identity import identity_correspondence

# Corrector
from .corrector.topology_consistency import topology_consistency_corrector

# Mesh croppers
from .mesh_croppers.point_based_cropper import point_based_crop

# Rigid aligners
from .rigid_aligners.icp import icp_align
from .rigid_aligners.landmark_based import landmark_based_align

# Non-rigid aligners
from .nonrigid_aligners.landmark_elastic import landmark_elastic_align
from .nonrigid_aligners.nonrigid_icp import nonrigid_icp_align

# Evaluation utilities
from .utils.utils import run_pipeline_batch

#Visualization
from .visualization.correlation import compute_correlation, plot_correlation
from .visualization.results_table import generate_results_table
from .visualization.heatmap_plain import plot_face_heatmap
from .visualization.space_heatmap import plot_face_3d_error, plot_face_3d_error_interactive, \
    plot_point_clouds_with_error_interactive

# Configurations
from .config import (
    PipelineConfig,
    RigidAlignerConfig,
    NonRigidAlignerConfig,
    CorrEstablisherConfig,
    CorrectorConfig,
    DistanceComputerConfig,
    MeshCropperConfig,
    # Enums
    MeshCropperType,
    RigidAlignerType,
    NonRigidAlignerType,
    PrealignMethod,
    CorrEstablisherType,
    CorrectorType,
    CorrectionStrategy,
    WeightPower,
    WeightStrategy,
    DistanceComputerType,
)

__all__ = [
    # Distances
    "p2p_distance",
    "p2tri_distance",
    "landmark_distance",

    # Correspondences
    "chamfer_correspondence",
    "identity_correspondence",

    # Corrector
    "topology_consistency_corrector",

    # Mesh croppers
    "point_based_crop",

    # Aligners
    "icp_align",
    "landmark_based_align",
    "landmark_elastic_align",
    "nonrigid_icp_align",

    # Evaluation
    "run_pipeline_batch",

    # Visualization
    "compute_correlation",
    "plot_correlation",

    # Configs
    "PipelineConfig",
    "MeshCropperConfig",
    "RigidAlignerConfig",
    "NonRigidAlignerConfig",
    "CorrEstablisherConfig",
    "CorrectorConfig",
    "DistanceComputerConfig",

    # Enums
    "MeshCropperType",
    "RigidAlignerType",
    "PrealignMethod",
    "NonRigidAlignerType",
    "CorrEstablisherType",
    "CorrectorType",
    "CorrectionStrategy",
    "WeightPower",
    "WeightStrategy",
    "DistanceComputerType",
]
