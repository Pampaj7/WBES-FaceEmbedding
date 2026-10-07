#!/usr/bin/env bash
#SBATCH --job-name=wbes-distill-v2
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#
# Distillazione v2 (aau/distill/train_distill_v2.py): operatori calcolati nel job (64 processi), cache
# RAM compatta, poi training su una GPU. Run dir aau/runs/distill_v2/student_<job>. Argomenti extra in
# WBES_DISTILL_ARGS (es. --sources bfm,ict5k,ictscale per il braccio senza GNM).
#
#   aau/submit.sh distill/train_distill_v2.sbatch
#   WBES_DISTILL_ARGS="--limit-subjects 20 --steps 40 --steps-per-epoch 20 --val-every 1" \
#     aau/submit.sh distill/train_distill_v2.sbatch --mem=48G --cpus-per-task=16 --time=00:40:00   smoke
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv
aau_require_diffusion_net

RUN_DIR="${WBES_DISTILL_RUN_DIR:-$AAU_RUNS/distill_v2/student_${SLURM_JOB_ID:-manual}}"
STAGE="/tmp/wbes_distill_v2_${SLURM_JOB_ID:-manual}"
rm -rf "$STAGE"
trap 'rm -rf "$STAGE"' EXIT
CORES="${SLURM_CPUS_PER_TASK:-8}"

echo "[v2] host=$(hostname) job=${SLURM_JOB_ID:-none} run_dir=$RUN_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# un thread per processo negli operatori; il training ne usa 8 (torch.set_num_threads)
# shellcheck disable=SC2086
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$AAU_DIR/run.sh" aau/distill/train_distill_v2.py \
    --stage "$STAGE" --run-dir "$RUN_DIR" --n-proc "$CORES" ${WBES_DISTILL_ARGS:-}
echo "[v2] OK"
