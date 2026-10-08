#!/usr/bin/env bash
#SBATCH --job-name=wbes-v3-bench
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=34G
#SBATCH --time=01:00:00
#
# Throughput del passo di training su L40S: sequenziale (v2) contro packed, batch reali BFM/ICT/GNM.
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
STAGE="/tmp/${SLURM_JOB_ID:-manual}"
trap 'rm -rf "$STAGE/bench"' EXIT
mkdir -p "$STAGE"
EV="$AAU_RUNS/evidence/trainer_v3"
OUT="$EV${WBES_V3_SUITE_TAG:+/$WBES_V3_SUITE_TAG}"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$AAU_DIR/run.sh" v3_work/trainer/tests/bench_forward_v3.py \
    --testdata "$EV/testdata" --stage-root "$STAGE/bench" --out "$OUT/bench_forward.json" \
    --poolings meanmax,area_attn --bucket-ratios 1.5 --reps 20
