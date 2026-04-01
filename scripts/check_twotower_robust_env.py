from __future__ import annotations

import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic" / "train_twotower_dn_spec_robust.py"


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(SCRIPT_PATH.parent))

    from face_embedding.gt_encdec.autoencoder.path_setup import resolve_diffusion_net_src

    diffusion_net_src = resolve_diffusion_net_src()
    sys.path.insert(0, str(diffusion_net_src))

    imported = {}
    for module_name in (
        "numpy",
        "scipy",
        "sklearn",
        "torch",
        "tqdm",
        "igl",
        "potpourri3d",
        "robust_laplacian",
        "tensorboard",
        "diffusion_net",
        "robustness.train_runner",
    ):
        imported[module_name] = importlib.import_module(module_name)

    train_runner = imported["robustness.train_runner"]
    print(f"Python executable: {sys.executable}")
    print(f"diffusion-net src: {diffusion_net_src}")
    print(f"torch: {imported['torch'].__version__}")
    print(f"cuda available: {imported['torch'].cuda.is_available()}")
    print(f"default data dir exists: {Path(train_runner.DEFAULT_DATA_DIR).exists()}")
    print(f"default dist npz exists: {Path(train_runner.DEFAULT_DIST_NPZ).exists()}")
    print("Environment import-check OK for train_twotower_dn_spec_robust.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
