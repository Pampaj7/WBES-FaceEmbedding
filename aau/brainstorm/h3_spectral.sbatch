#!/usr/bin/env bash
#SBATCH --job-name=wbes-bs-h3
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%A_%a.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%A_%a.err
#SBATCH --partition=cpu
#SBATCH --qos=unprivileged
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --array=0-15
#
# H3 del brainstorm: spettri e HKS in 5 varianti di operatore, 50 soggetti held-out BFM,
# un soggetto ~2 min.  Ripartibile: i soggetti gia' scritti in
# aau/runs/brainstorm/h3_parts/ vengono saltati, quindi unprivileged va bene.
#
#   sbatch aau/brainstorm/h3_spectral.sbatch
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export AAU_NV=""
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK \
       OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

"$AAU_DIR/run.sh" aau/brainstorm/h3_spectral.py --shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
