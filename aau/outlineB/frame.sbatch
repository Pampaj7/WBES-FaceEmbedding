#!/usr/bin/env bash
#SBATCH --job-name=wbes-oB-frame
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=30
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#
# Chamfer nei frame maxabs / area / rms / global e D_GT nei frame raw / maxabs / area / rms /
# global, held-out BFM (aau/outlineB/frame_matrix.py). Solo CPU, nessun open3d.
#   aau/submit.sh outlineB/frame.sbatch
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "[oB-frame] host=$(hostname) job=${SLURM_JOB_ID:-none}"
"$AAU_DIR/run.sh" aau/outlineB/frame_matrix.py \
    --out-root "$AAU_RUNS/outlineB/bfm_heldout" --workers "${SLURM_CPUS_PER_TASK:-8}" ${OB_ARGS:-}
echo "[oB-frame] OK"
