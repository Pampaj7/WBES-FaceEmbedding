#!/usr/bin/env bash
#SBATCH --job-name=wbes-hrn-setup
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#
# Venv per HRN (external/HRN, aau/recon/hrn_run.py), dopo hrn_pytorch3d_build.sbatch (pytorch3d gia' compilato):
#   external/venvs/hrn   torch del container + pytorch3d + nvdiffrast (dal sorgente) + tensorflow-cpu (solo il .pb
#                        di segmentazione, grafo TF1), face-alignment 1.3.5 (LandmarksType._3D, tolto nelle 1.4),
#                        kornia, einops, trimesh, numba, scikit-image, albumentations, torchsummary, imageio
# Asset (cartella Google Drive del README di HRN, nessuna registrazione) in external/HRN/assets con gdown, e i pesi di
# face-alignment nella cache di torch hub (li scarica il primo FaceAlignment). Su una L40S: nvdiffrast compila il plugin
# CUDA al primo uso, e la prova d'importazione lo fa.
#
#   aau/submit.sh recon/hrn_setup_env.sbatch
set -euo pipefail

export AAU_NV="--nv"
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

export VENV_HRN="$WBES_ROOT/external/venvs/hrn"
echo "[hrn-setup] host=$(hostname) job=${SLURM_JOB_ID:-none} CONTAINER=$CONTAINER"
[[ -d "$WBES_ROOT/external/HRN" ]] || { echo "ERRORE: manca external/HRN" >&2; exit 1; }
[[ -x "$VENV_HRN/bin/python3" ]] || { echo "ERRORE: manca $VENV_HRN (hrn_pytorch3d_build.sbatch)" >&2; exit 1; }

singularity exec --nv "$CONTAINER" bash -c '
set -euo pipefail
export PIP_CONSTRAINT="$WBES_ROOT/external/venvs/pip-constraints.txt"
source "$VENV_HRN/bin/activate"
export TORCH_CUDA_ARCH_LIST="8.6;8.9" MAX_JOBS="${SLURM_CPUS_PER_TASK:-16}"
python3 -m pip install --no-input --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"
python3 -m pip install --no-input "tensorflow-cpu==2.15.1" "face-alignment==1.3.5" kornia einops trimesh numba \
    scikit-image albumentations torchsummary imageio gdown "opencv-python-headless<5"
python3 -m pip uninstall -y opencv-python opencv-contrib-python || true
python3 -m pip install --no-input --force-reinstall --no-deps "opencv-python-headless<5"
python3 -m pip list --local
if [[ ! -f "$WBES_ROOT/external/HRN/assets/pretrained_models/hrn_v1.1/epoch_10.pth" ]]; then
    DL="$WBES_ROOT/external/HRN/assets_gdrive"
    mkdir -p "$DL"
    gdown --folder "https://drive.google.com/drive/folders/1qI3-5GKxEDbQBSUCwDWmH15uWU3Ka5LS" -O "$DL"
    find "$DL" -maxdepth 3 | head -50
fi
'
echo "[hrn-setup] OK (asset da sistemare in external/HRN/assets se scaricati in assets_gdrive)"
