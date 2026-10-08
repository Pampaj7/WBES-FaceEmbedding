#!/usr/bin/env bash
#SBATCH --job-name=wbes-v3-c3f-eval
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --partition=aicentre-a100
#SBATCH --qos=unprivileged
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#
#   WBES_V3_ARM=<braccio> aau/submit.sh v3_work/trainer/ablations/c3f/eval.sbatch --dependency=afterok:<training>
# Corpo in eval_body.sh, copiato all'avvio.
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
B="/tmp/v3c3f_eval_body_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}.sh"
cp "$WBES_ROOT/v3_work/trainer/ablations/c3f/eval_body.sh" "$B"
exec bash "$B"
