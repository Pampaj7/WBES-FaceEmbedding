from __future__ import annotations

import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
ROBUSTNESS_DIR = THIS_FILE.parent
INTRINSIC_DIR = ROBUSTNESS_DIR.parent
REPO_ROOT = THIS_FILE.parents[5]
AUTOENCODER_DIR = REPO_ROOT / "face_embedding" / "gt_encdec" / "autoencoder"

DEFAULT_DATA_DIR = REPO_ROOT / "datasets" / "REMESH" / "npz_data_topo_500_withops"
DEFAULT_DIST_NPZ = (
    REPO_ROOT
    / "face_embedding"
    / "gt_encdec"
    / "autoencoder"
    / "latent_analysis"
    / "gt_distance_matrix"
    / "normalized_matrix_distances.npz"
)
TRAIN_RUNS_ROOT = INTRINSIC_DIR / "perturbated"


def ensure_autoencoder_dir_on_syspath() -> None:
    if str(AUTOENCODER_DIR) not in sys.path:
        sys.path.append(str(AUTOENCODER_DIR))


ensure_autoencoder_dir_on_syspath()
