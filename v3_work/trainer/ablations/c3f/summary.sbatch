#!/usr/bin/env bash
#SBATCH --job-name=wbes-v3-c3f-summary
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#
# Ablazioni v3 C3F: tabella finale e regola di adozione (v3_work/trainer/tools/ablation_summary.py) in
# aau/runs/evidence/trainer_v3/ablations/results.md. Niente GPU.
#   aau/submit.sh v3_work/trainer/ablations/c3f/summary.sbatch [--dependency=afterok:<eval>,...]
set -euo pipefail
export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
echo "[v3-sum] host=$(hostname) job=${SLURM_JOB_ID:-none}"
"$AAU_DIR/run.sh" v3_work/trainer/tools/ablation_summary.py
