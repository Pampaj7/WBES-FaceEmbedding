#!/usr/bin/env bash
#SBATCH --job-name=wbes-e10
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#
# E10: operatori DiffusionNet su GPU. Una GPU, 16 CPU (thread di lettura/scrittura e analisi
# simbolica di cuDSS). Il tipo di GPU si sceglie al lancio, il default e' V100 (fp64 forte; container
# 24.10 <= 24.11, quindi ok su sm_70):
#   sbatch aau/evidence/e10_gpu_ops/e10.sbatch                         # check + bench su V100
#   E10_STEPS=bench sbatch --gres=gpu:a40:1 aau/evidence/e10_gpu_ops/e10.sbatch
#   E10_CHECK_K=128 E10_CHECK_ARGS="--variants bk" sbatch --gres=gpu:l40s:1 ... e10.sbatch
#   E10_BENCH_ARGS="--parts classes --variant bk_1pass" E10_STEPS=bench sbatch ... e10.sbatch
# Richiede .venv_e10 (cuDSS): vedi aau/evidence/e10_gpu_ops/README.md.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export VENV="$WBES_ROOT/.venv_e10"
OUT="$AAU_RUNS/evidence/e10"
WORK="/tmp/${SLURM_JOB_ID:-manual}/e10"
mkdir -p "$OUT/bench" "$WORK"
trap 'rm -rf "/tmp/${SLURM_JOB_ID:-manual}"' EXIT
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')"
{ echo "host=$(hostname) job=${SLURM_JOB_ID:-none} gpu=$GPU cpus=${SLURM_CPUS_PER_TASK:-?} date=$(date -Is)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  lscpu | grep -E "Model name|^CPU\(s\)"
} > "$OUT/bench/node_${SLURM_JOB_ID:-manual}.txt"

for step in ${E10_STEPS:-check bench}; do
    case "$step" in
        check)
            for k in ${E10_CHECK_K:-128 64}; do
                "$AAU_DIR/run.sh" aau/evidence/e10_gpu_ops/check_ops.py --work-dir "$WORK/check_k$k" --k "$k" \
                    --out-json "$OUT/check_k${k}_${GPU}.json" --n-proc "${SLURM_CPUS_PER_TASK:-16}" ${E10_CHECK_ARGS:-}
                rm -rf "$WORK/check_k$k"
            done ;;
        bench)
            "$AAU_DIR/run.sh" aau/evidence/e10_gpu_ops/bench.py --work-dir "$WORK/bench" \
                --out-dir "$OUT/bench" --tag "_${GPU}" ${E10_BENCH_ARGS:-} ;;
    esac
done
echo "[e10] OK"
