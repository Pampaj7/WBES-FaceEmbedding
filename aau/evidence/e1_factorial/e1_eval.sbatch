#!/usr/bin/env bash
#SBATCH --job-name=wbes-e1-eval
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:30:00
#
# E1: valutazione dei due checkpoint (10.548 e 21.096 passi) di una cella sui domini di test, con le
# pipeline esistenti chiamate cosi' come sono (corpo in e1_eval_body.sh, letto all'avvio del job):
#
#   WBES_E1_CELL=c2m|c2f|c3f|g1|c3m [WBES_E1_EVAL_STEPS="hifi fv flame now"] \
#     aau/submit.sh evidence/e1_factorial/e1_eval.sbatch [--dependency=afterok:<training>]
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
exec bash "$WBES_ROOT/aau/evidence/e1_factorial/e1_eval_body.sh"
