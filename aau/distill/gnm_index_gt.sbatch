#!/usr/bin/env bash
#SBATCH --job-name=wbes-gnm-index-gt
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#
# Indice dei tar e GT d'identita' (L2 maxabs fra le neutre) delle identita' GNM (aau/distill/gnm_index_gt.py).
#   aau/submit.sh distill/gnm_index_gt.sbatch
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
echo "[gnm-index-gt] host=$(hostname) job=${SLURM_JOB_ID:-none}"
OMP_NUM_THREADS=1 "$AAU_DIR/run.sh" aau/distill/gnm_index_gt.py --shards-dir datasets/GNM_DISTILL/shards \
    --gt-dir datasets/GNM_DISTILL/gt --n-proc "${SLURM_CPUS_PER_TASK:-8}"
echo "[gnm-index-gt] OK"
