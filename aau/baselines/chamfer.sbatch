#!/usr/bin/env bash
#SBATCH --job-name=wbes-bl-chamfer
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# Chamfer raw: la baseline che valida tutta la pipeline (deve dare lo 0.729 del paper).
# Default = variante faceBench, quella con cui il numero del paper e' stato prodotto:
# cKDTree su CPU, nessuna GPU.
#   aau/submit.sh baselines/chamfer.sbatch
#
# L'altra variante, quella dell'eval del repo (chamfer_sq), vuole una GPU:
#   WBES_BL_ARGS="--variant eval_utils" aau/submit.sh baselines/chamfer.sbatch --gres=gpu:l40s:1
#
# Gira nel venv normale (.venv_aau): servono solo scipy, torch e il codice del repo.
set -euo pipefail

if [[ -z "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}${GPU_DEVICE_ORDINAL:-}" ]]; then
    export AAU_NV=""
fi
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

"$AAU_DIR/run.sh" aau/baselines/chamfer_matrix.py \
    --workers "${SLURM_CPUS_PER_TASK:-8}" ${WBES_BL_ARGS:-}
