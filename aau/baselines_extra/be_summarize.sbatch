#!/usr/bin/env bash
#SBATCH --job-name=wbes-be-sum
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#
# Tabelle di CLIP e DINOv2 (vfm_summarize.py) per i domini di WBES_BE_DOMAINS, uno dopo l'altro, poi
# summary.md ricomposto in aau/runs/baselines_extra/.
#
#   aau/submit.sh baselines_extra/be_summarize.sbatch --dependency=afterok:<embed>
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "[be-sum] host=$(hostname) job=${SLURM_JOB_ID:-none}"
for dom in ${WBES_BE_DOMAINS:-fv_expr hifi3d bfm_heldout}; do
    "$AAU_DIR/run.sh" aau/baselines_extra/vfm_summarize.py --domain "$dom" --eval-seed "${WBES_EVAL_SEED:-1234}"
done
echo "[be-sum] OK"
