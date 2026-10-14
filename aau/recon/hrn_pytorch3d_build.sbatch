#!/usr/bin/env bash
#SBATCH --job-name=wbes-hrn-p3d
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:10:00
#
# HRN, prova di ammissione (aau/runs/evidence/now_ranking/REGOLE.md, sez. 1): HRN entra solo se pytorch3d si
# compila entro 1 h. Venv external/venvs/hrn (torch del container), pytorch3d dal sorgente (tag stable) con le
# estensioni CUDA per sm_89 (L40S) e sm_86 (A10/A40); nessuna GPU serve per compilare.
#
#   aau/submit.sh recon/hrn_pytorch3d_build.sbatch
#
# Esito nell'ultima riga del log: "[hrn-p3d] AMMESSO in <s> s" oppure "[hrn-p3d] ESCLUSO (...)".
set -uo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

export VENV_HRN="$WBES_ROOT/external/venvs/hrn"
echo "[hrn-p3d] host=$(hostname) job=${SLURM_JOB_ID:-none} CONTAINER=$CONTAINER"

t0=$(date +%s)
timeout 3600 singularity exec "$CONTAINER" bash -c '
set -euo pipefail
export PIP_CONSTRAINT="$WBES_ROOT/external/venvs/pip-constraints.txt"
[[ -x "$VENV_HRN/bin/python3" ]] || python3 -m venv --system-site-packages --without-pip "$VENV_HRN"
source "$VENV_HRN/bin/activate"
export FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.6;8.9" MAX_JOBS="${SLURM_CPUS_PER_TASK:-16}"
nvcc --version | tail -1
python3 -m pip install --no-input --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
python3 -c "import pytorch3d, pytorch3d._C; print(\"pytorch3d\", pytorch3d.__version__)"
'
rc=$?
dt=$(( $(date +%s) - t0 ))
if (( rc == 0 && dt <= 3600 )); then
    echo "[hrn-p3d] AMMESSO in $dt s"
else
    echo "[hrn-p3d] ESCLUSO (rc=$rc, $dt s)"
fi
