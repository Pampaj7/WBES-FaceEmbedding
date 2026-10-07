#!/usr/bin/env bash
#SBATCH --job-name=wbes-mica-setup
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#
# Venv per MICA (external/MICA, aau/recon/mica_run.py):
#   external/venvs/mica   torch del container + insightface, onnxruntime, face-alignment,
#                         trimesh, loguru, yacs, scikit-image
#
#   aau/submit.sh recon/mica_setup_env.sbatch
#
# Niente pytorch3d: lo importano solo micalib/renderer.py e micalib/tester.py, non la catena
# di demo.py.  face-alignment serve solo perche' utils/landmark_detector.py lo importa in
# testa (il detector usato e' RetinaFace di insightface).  numpy<2 come negli altri venv:
# numpy 2.x nel venv shadowa quella del container e rompe torch NGC.
# Prerequisito (frontend): il clone external/MICA e gli asset di mica_run.py.
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

export VENV_MICA="$WBES_ROOT/external/venvs/mica"
echo "[mica-setup] host=$(hostname) job=${SLURM_JOB_ID:-none} CONTAINER=$CONTAINER"
[[ -d "$WBES_ROOT/external/MICA" ]] || { echo "ERRORE: manca external/MICA" >&2; exit 1; }

singularity exec "$CONTAINER" bash -c '
set -euo pipefail
printf "numpy<2\ntyping_extensions>=4.8\n" > /tmp/mica-constraints.txt
export PIP_CONSTRAINT=/tmp/mica-constraints.txt
[[ -x "$VENV_MICA/bin/python3" ]] || python3 -m venv --system-site-packages --without-pip "$VENV_MICA"
source "$VENV_MICA/bin/activate"
python3 -m pip install --no-input "insightface==0.7.3" onnxruntime face-alignment trimesh loguru yacs \
    scikit-image "opencv-python-headless<5"
# Qualche dipendenza tira dentro opencv-python, che carica libGL (assente nel container) e si
# installa sopra la headless nello stesso pacchetto cv2: via, e headless reinstallata.
python3 -m pip uninstall -y opencv-python opencv-contrib-python || true
python3 -m pip install --no-input --force-reinstall --no-deps "opencv-python-headless<5"
python3 -m pip list --local
'

echo "[mica-setup] --- import test ---"
"$AAU_DIR/recon/run_recon.sh" mica -c "
import sys; sys.path.insert(0, 'external/MICA')
import numpy, torch, insightface, onnxruntime, face_alignment, trimesh, loguru, yacs, skimage
from configs.config import get_cfg_defaults
from utils.landmark_detector import LandmarksDetector, detectors
print('numpy', numpy.__version__, 'torch', torch.__version__, 'insightface', insightface.__version__)
"
echo "[mica-setup] OK"
