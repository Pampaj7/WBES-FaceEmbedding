#!/usr/bin/env bash
#SBATCH --job-name=wbes-bl-perc
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#
# ArcFace / CLIP / DINOv2 / LPIPS sui render. Gira nel venv delle baseline.
# Prerequisiti: aau/submit.sh baselines/setup_env.sbatch e baselines/render.sbatch.
#   aau/submit.sh baselines/perceptual.sbatch
#   WBES_BL_ARGS="--extractors lpips" aau/submit.sh baselines/perceptual.sbatch
set -euo pipefail

# ArcFace gira su onnxruntime CPU: lo si sottomette a parte, senza GPU e su piu' processi
#   WBES_BL_ARGS="--extractors arcface --device cpu --embed-workers 16" \
#     aau/submit.sh baselines/perceptual.sbatch --gres=NONE --cpus-per-task=16
if [[ -z "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}${GPU_DEVICE_ORDINAL:-}" ]]; then
    export AAU_NV=""
fi
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

"$AAU_DIR/baselines/run_bl.sh" aau/baselines/perceptual_matrix.py --device cuda ${WBES_BL_ARGS:-}
