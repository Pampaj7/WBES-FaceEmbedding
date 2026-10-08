#!/usr/bin/env bash
#SBATCH --job-name=wbes-v3-ac
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --gres-flags=disable-binding
#SBATCH --time=03:00:00
#
# Test anti-collasso delle loss (tests/anti_collapse.py): v2 (riferimento) e le varianti log con ancora di scala.
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
STAGE="/tmp/${SLURM_JOB_ID:-manual}"
trap 'rm -rf "$STAGE/ac"' EXIT
mkdir -p "$STAGE"
EV="$AAU_RUNS/evidence/trainer_v3"
OUT="$EV${WBES_V3_SUITE_TAG:+/$WBES_V3_SUITE_TAG}"
mkdir -p "$OUT"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$AAU_DIR/run.sh" v3_work/trainer/tests/anti_collapse.py \
    --losses "${WBES_V3_AC_LOSSES:-v2,log,log+inv,log+inv+nbr}" --steps "${WBES_V3_AC_STEPS:-6000}" \
    --testdata "$EV/testdata" --stage-root "$STAGE/ac" --out "$OUT/anti_collapse_${WBES_V3_AC_TAG:-${SLURM_JOB_ID:-manual}}.json"
