#!/usr/bin/env bash
#SBATCH --job-name=wbes-distill-train
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#
# Pilota della distillazione (aau/distill/train_distill.py): studente DiffusionNet, bersaglio ArcFace su
# normal map delle original (aau/runs/distill_pilot/teacher/{bfm,ict}). Run dir:
# aau/runs/distill_pilot/student_<job>. Argomenti extra del trainer in WBES_DISTILL_ARGS.
#
#   aau/submit.sh distill/train_distill.sbatch
#   WBES_DISTILL_TEACHER=aau/runs/distill_pilot/teacher_limit24 WBES_DISTILL_ARGS="--epochs 2" \
#     aau/submit.sh distill/train_distill.sbatch --mem=32G --time=00:30:00          smoke
#
# Memoria: cache RAM non pinned di tutte le mesh d'ingresso (CachedDataset), dimensionata sullo smoke.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv
aau_require_diffusion_net

TEACHER="${WBES_DISTILL_TEACHER:-$AAU_RUNS/distill_pilot/teacher}"
RUN_DIR="${WBES_DISTILL_RUN_DIR:-$AAU_RUNS/distill_pilot/student_${SLURM_JOB_ID:-manual}}"
STAGE="/tmp/wbes_distill_${SLURM_JOB_ID:-manual}"
rm -rf "$STAGE"
trap 'rm -rf "$STAGE"' EXIT
THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "[distill] host=$(hostname) job=${SLURM_JOB_ID:-none} teacher=$TEACHER run_dir=$RUN_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# shellcheck disable=SC2086
OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" "$AAU_DIR/run.sh" aau/distill/train_distill.py \
    --teacher-dirs "$TEACHER/bfm" "$TEACHER/ict" --stage "$STAGE" --run-dir "$RUN_DIR" \
    --threads "$THREADS" --cache-workers "$THREADS" ${WBES_DISTILL_ARGS:-}
echo "[distill] OK"
