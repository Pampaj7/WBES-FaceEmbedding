#!/usr/bin/env bash
#SBATCH --job-name=wbes-fvn-summary
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#
# Riepilogo FaceVerse neutra contro espressioni (fvn_summary.py), dopo fvn_embed.sbatch e le baseline (launch.sh).
#   FVN_JOBS="id ..." sbatch v3_work/faceverse_neutral/fvn_summary.sbatch
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
echo "[fvn-summary] host=$(hostname) job=${SLURM_JOB_ID:-none}"
OMP_NUM_THREADS=1 AAU_NV= "$AAU_DIR/run.sh" v3_work/faceverse_neutral/fvn_summary.py \
  --workers "${SLURM_CPUS_PER_TASK:-8}" --jobs "${FVN_JOBS:-}${FVN_JOBS:+, }riepilogo ${SLURM_JOB_ID:-manual}"
