#!/usr/bin/env bash
#SBATCH --job-name=wbes-ugt-setup
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#
# Venv della GT unificata (v3_work/unified_gt): .venv_ugt.
#   sbatch v3_work/unified_gt/setup_env.sbatch
#
# Eredita .venv_open3d (open3d 0.18 per il ray casting, cvxpy per facebench, igl, trimesh) con
# un .pth invece di reinstallarlo, e aggiunge solo:
#  - mediapipe 0.10.14 (FaceMesh "legacy", 468 landmark): il detector 2D denso;
#  - h5py: AI-NEXT-Shape-NoAug.mat di HIFI3D e' un .mat v7.3.
# numpy<2 come negli altri venv (numpy 2 nel venv rompe torch NGC). mediapipe tira dentro
# opencv-contrib-python, che vuole libGL: si sostituisce con la headless.
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
export VENV="$WBES_ROOT/.venv_ugt"
export O3D_VENV="$WBES_ROOT/.venv_open3d"
cd "$WBES_ROOT"

echo "[ugt-setup] host=$(hostname) job=${SLURM_JOB_ID:-none} VENV=$VENV"
singularity exec "$CONTAINER" bash -c '
set -euo pipefail
if [[ ! -x "$VENV/bin/python3" ]]; then
    python3 -m venv --system-site-packages --without-pip "$VENV"
fi
SP=$(ls -d "$VENV"/lib/python3*/site-packages)
O3D_SP=$(ls -d "$O3D_VENV"/lib/python3*/site-packages)
echo "$O3D_SP" > "$SP/zz_venv_open3d.pth"
source "$VENV/bin/activate"
printf "numpy<2\n" > "$VENV/pip-constraints.txt"
export PIP_CONSTRAINT="$VENV/pip-constraints.txt"
python3 -m pip install --no-input "mediapipe==0.10.14" h5py
python3 -m pip uninstall -y opencv-contrib-python opencv-python || true
python3 -m pip install --no-input --no-deps "opencv-contrib-python-headless<4.11"
python3 -m pip list --path "$SP"
'

echo "[ugt-setup] --- prova d import ---"
"$WBES_ROOT/v3_work/unified_gt/run.sh" -c "
import numpy, open3d, cv2, h5py, mediapipe as mp, trimesh, igl
fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)
print('numpy', numpy.__version__, 'open3d', open3d.__version__, 'cv2', cv2.__version__, 'mediapipe', mp.__version__)
"
echo "[ugt-setup] OK"
