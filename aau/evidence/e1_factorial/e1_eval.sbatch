#!/usr/bin/env bash
#SBATCH --job-name=wbes-e1-eval
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=aicentre-a100
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=05:00:00
#SBATCH --requeue
#
# E1: valutazione dei due checkpoint (10.548 e 21.096 passi) di una cella sui domini di test, con le pipeline
# esistenti chiamate cosi' come sono (corpo in e1_eval_body.sh). Tutte le eval sulle A100 di nv-ai-04
# (eccezione autorizzata dall'utente, 8 ottobre): QoS unprivileged, quindi PRELAZIONABILE; --requeue, e il
# corpo riprende da dove era: le pipeline saltano i bracci gia' completi (.done) e i latenti NoW gia' calcolati.
#
#   WBES_E1_CELL=<cella> [WBES_E1_EVAL_STEPS="hifi devfs fv flame now"] \
#     aau/submit.sh evidence/e1_factorial/e1_eval.sbatch [--dependency=afterok:<training>]
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
# copia privata del corpo, presa all'avvio (vedi e1_train.sbatch)
B="/tmp/e1_eval_body_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}.sh"
cp "$WBES_ROOT/aau/evidence/e1_factorial/e1_eval_body.sh" "$B"
exec bash "$B"
