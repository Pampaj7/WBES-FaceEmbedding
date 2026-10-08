#!/usr/bin/env bash
#SBATCH --job-name=wbes-v3-c3f
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --partition=aicentre-a100
#SBATCH --qos=unprivileged
#SBATCH --requeue
#SBATCH --nodelist=nv-ai-04
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=130G
#SBATCH --time=12:00:00
#
#   WBES_V3_ARM=ctrl|area|arearobust|bal|ugtmix aau/submit.sh v3_work/trainer/ablations/c3f/train.sbatch
# Corpo in train_body.sh, copiato all'avvio (un file modificato dopo non tocca il job in corso).
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
B="/tmp/v3c3f_train_body_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}.sh"
cp "$WBES_ROOT/v3_work/trainer/ablations/c3f/train_body.sh" "$B"
exec bash "$B"
