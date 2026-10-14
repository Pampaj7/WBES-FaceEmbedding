#!/usr/bin/env bash
#SBATCH --job-name=wbes-ddfa3-setup
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#
# Venv per 3DDFA-V3 (external/3DDFA-V3, aau/recon/tddfa_v3_run.py):
#   external/venvs/ddfa3   torch e torchvision del container + scikit-image, albumentations, torch-summary,
#                          opencv headless
#
#   aau/submit.sh recon/tddfa_v3_setup_env.sbatch
#
# Niente nvdiffrast, tensorflow e mtcnn: servono ai render del demo e al detector alternativo, non alla geometria
# (tddfa_v3_run.py li sostituisce con stub che sollevano se usati). numpy<2 come negli altri venv.
# Prerequisito (frontend): il clone external/3DDFA-V3 e gli asset in external/3DDFA-V3/assets.
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

export VENV_DDFA3="$WBES_ROOT/external/venvs/ddfa3"
echo "[ddfa3-setup] host=$(hostname) job=${SLURM_JOB_ID:-none} CONTAINER=$CONTAINER"
[[ -d "$WBES_ROOT/external/3DDFA-V3/assets" ]] || { echo "ERRORE: mancano external/3DDFA-V3/assets" >&2; exit 1; }

singularity exec "$CONTAINER" bash -c '
set -euo pipefail
export PIP_CONSTRAINT="$WBES_ROOT/external/venvs/pip-constraints.txt"
[[ -x "$VENV_DDFA3/bin/python3" ]] || python3 -m venv --system-site-packages --without-pip "$VENV_DDFA3"
source "$VENV_DDFA3/bin/activate"
python3 -m pip install --no-input scikit-image albumentations torch-summary "opencv-python-headless<5"
# albumentations puo tirare dentro opencv-python (libGL, assente nel container): via, headless reinstallata.
python3 -m pip uninstall -y opencv-python opencv-contrib-python || true
python3 -m pip install --no-input --force-reinstall --no-deps "opencv-python-headless<5"
python3 -m pip list --local
'

echo "[ddfa3-setup] --- import test ---"
"$AAU_DIR/recon/run_recon.sh" ddfa3 -c "
import numpy, torch, torchvision, cv2, skimage, albumentations, torchsummary
print('numpy', numpy.__version__, 'torch', torch.__version__, 'cv2', cv2.__version__, 'albumentations', albumentations.__version__)
"
echo "[ddfa3-setup] OK"
