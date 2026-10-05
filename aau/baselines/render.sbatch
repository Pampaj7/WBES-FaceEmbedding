#!/usr/bin/env bash
#SBATCH --job-name=wbes-bl-render
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#
# Render 3 viste x 500 mesh con camera unica. Solo numpy+PIL: niente GPU.
#   aau/submit.sh baselines/render.sbatch
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

"$AAU_DIR/run.sh" aau/baselines/render_cache.py \
    --workers "${SLURM_CPUS_PER_TASK:-8}" ${WBES_BL_ARGS:-}
