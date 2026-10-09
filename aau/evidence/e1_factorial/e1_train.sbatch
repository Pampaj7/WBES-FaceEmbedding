#!/usr/bin/env bash
#SBATCH --job-name=wbes-e1-train
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#
# E1 (paper/PLAN_MASSIVE.md, sezione 13): una cella del fattoriale. Tutte le celle su V100 (decisione del PI,
# 8 ottobre): container PyTorch <= 24.11, controllato nel corpo. Il trainer non ha ripresa da checkpoint
# (train_runner.run_training riparte dall'epoca 1; --init_checkpoint carica solo i pesi): niente A100
# prelazionabili per il training, e --no-requeue. Lo lancia e1_launch.sh, che sceglie --mem e --time per cella:
#
#   WBES_E1_CELL=<cella> aau/submit.sh evidence/e1_factorial/e1_train.sbatch --mem=... --time=...
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
# copia privata del corpo, presa all'AVVIO del job: bash legge lo script mentre lo esegue, e una modifica
# del file durante il job lo corromperebbe
B="/tmp/e1_train_body_${SLURM_JOB_ID:-manual}.sh"
cp "$WBES_ROOT/aau/evidence/e1_factorial/e1_train_body.sh" "$B"
exec bash "$B"
