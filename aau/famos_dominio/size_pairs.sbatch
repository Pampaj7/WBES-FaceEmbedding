#!/usr/bin/env bash
#SBATCH --job-name=wbes-size-pairs
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#
# Effetto della taglia per ogni concorrente su FaMoS, con IC appaiato (aau/famos_dominio/size_pairs.py). Solo CPU.
#   sbatch aau/famos_dominio/size_pairs.sbatch
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export AAU_NV=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "[size_pairs] host=$(hostname) job=${SLURM_JOB_ID:-none} $(date +%T)"
aau/run.sh -u aau/famos_dominio/size_pairs.py
echo "[size_pairs] fine $(date +%T)"
