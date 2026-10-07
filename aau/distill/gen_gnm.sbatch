#!/usr/bin/env bash
#SBATCH --job-name=wbes-distill-gnm
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#
# Identita' GNM Head per la distillazione v2 (aau/distill/gen_gnm_shard.py): 10.100 identita' in 41 shard
# di sola geometria in datasets/GNM_DISTILL/shards (gli ultimi 100 = validazione).
#
#   aau/submit.sh distill/gen_gnm.sbatch
#   WBES_GNM_SHARDS=0-0 aau/submit.sh distill/gen_gnm.sbatch --time=00:20:00     prova su uno shard
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORK="/tmp/wbes_gnm_${SLURM_JOB_ID:-manual}"
trap 'rm -rf "$WORK"' EXIT

echo "[gnm] host=$(hostname) job=${SLURM_JOB_ID:-none} shards=${WBES_GNM_SHARDS:-0-40}"
"$AAU_DIR/run.sh" aau/distill/gen_gnm_shard.py --shards "${WBES_GNM_SHARDS:-0-40}" \
    --out-dir "${WBES_GNM_OUT:-datasets/GNM_DISTILL/shards}" --work-dir "$WORK" --n-cores "${SLURM_CPUS_PER_TASK:-8}"
echo "[gnm] OK"
