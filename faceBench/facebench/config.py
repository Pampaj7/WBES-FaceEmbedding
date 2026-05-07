from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


# =============================
# FaceBench Configuration Models
# =============================

# NOTE:
# - All configuration classes use PascalCase naming (e.g., CorrectorConfig)
#   in accordance with Python PEP8 conventions for class names.
# - Enum values and configuration fields inside the classes remain in snake_case.
# - This naming scheme distinguishes configuration containers (PascalCase)
#   from function arguments or local variables (snake_case).

class MeshCropperType(str, Enum):
    POINT_BASED = "point_based"


class RigidAlignerType(str, Enum):
    ICP = "icp"
    LANDMARK = "landmark"


class PrealignMethod(str, Enum):
    BBOX = "bbox"
    NONE = "none"
    LANDMARK = "landmark"


class NonRigidAlignerType(str, Enum):
    ELASTIC = "elastic"
    NICP = "nicp"


class CorrEstablisherType(str, Enum):
    CHAMFER = "chamfer"
    IDENTITY = "identity"


class CorrectorType(str, Enum):
    TOPOLOGY_CONSISTENCY = "topology_consistency"


class CorrectionStrategy(str, Enum):
    PAIR = "pair"
    TRACE = "trace"


class WeightPower(str, Enum):
    SQRT = "sqrt"
    SQUARE = "square"


class WeightStrategy(str, Enum):
    MIXED = "mixed"
    MIN = "min"
    MEAN = "mean"


class DistanceComputerType(str, Enum):
    P2P = "p2p"
    P2TRI = "p2tri"


@dataclass
class MeshCropperConfig:
    method: MeshCropperType
    dist_threshold_ratio: float = 1.0
    ref_lmk_index: int = 28
    leyec_index: int = 36
    reyec_index: int = 45


@dataclass
class RigidAlignerConfig:
    type: RigidAlignerType
    prealign: Optional[PrealignMethod] = PrealignMethod.BBOX
    ref_lmk_indices: Optional[List[int]] = None
    icp_threshold: float = 1000.0


@dataclass
class NonRigidAlignerConfig:
    type: NonRigidAlignerType

    # elastic
    gamma: Optional[float] = 1.0
    sel_lmk_ids: Optional[List[int]] = None
    ref_lmk_indices: Optional[List[int]] = None

    # nicp
    alpha: Optional[float] = 50.0
    epsilon: Optional[float] = 1.0
    prealign: Optional[bool] = False


@dataclass
class CorrEstablisherConfig:
    type: CorrEstablisherType


@dataclass
class CorrectorConfig:
    type: CorrectorType = CorrectorType.TOPOLOGY_CONSISTENCY
    correction_strategy: CorrectionStrategy = CorrectionStrategy.PAIR
    weight_power: WeightPower = WeightPower.SQRT
    weight_strategy: WeightStrategy = WeightStrategy.MIXED


@dataclass
class DistanceComputerConfig:
    type: DistanceComputerType


@dataclass
class PipelineConfig:
    mesh_cropper: Optional[MeshCropperConfig] = None
    rigid_aligner: Optional[RigidAlignerConfig] = None
    nonrigid_aligner: Optional[NonRigidAlignerConfig] = None
    corr_establisher: Optional[CorrEstablisherConfig] = None
    corrector: Optional[CorrectorConfig] = None
    distance_computer: Optional[DistanceComputerConfig] = None
