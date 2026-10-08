#!/usr/bin/env bash
#SBATCH --job-name=wbes-e9-fwd
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=7G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#
# E9 (b): forward + backward a gruppi su 1 L40S (fwd_bench.py). --mem=7G: l'unico nodo L40S con GPU
# libere l'8 ottobre (a768-l40s-02) ha ~7.9 GB di RAM non allocata. Basta: i dati vanno subito in GPU,
# l'host tiene solo torch/CUDA (~3 GB). Il programma si ferma da solo a 52 minuti (--deadline-min).
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
OUT="$AAU_RUNS/evidence/e9/fwd"
WORK="/tmp/${SLURM_JOB_ID:-manual}/fwd"
mkdir -p "$OUT" "$WORK"
trap 'rm -rf "/tmp/${SLURM_JOB_ID:-manual}"' EXIT
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv,noheader > "$OUT/gpu_${SLURM_JOB_ID:-manual}.txt" || true

OMP_NUM_THREADS=8 "$AAU_DIR/run.sh" aau/evidence/e9_bench/fwd_bench.py \
    --work-dir "$WORK" --out-dir "$OUT" ${E9_FWD_ARGS:-}
echo "[e9-fwd] OK"
