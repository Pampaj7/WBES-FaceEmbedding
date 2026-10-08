#!/usr/bin/env bash
#SBATCH --job-name=wbes-e1-train
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=430G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#
# E1 (paper/PLAN_MASSIVE.md, sezione 13): una cella del fattoriale varieta' x quantita'. Involucro: il
# corpo (e1_train_body.sh) e' letto all'AVVIO del job, cosi' un errore trovato dopo la sottomissione si
# corregge senza perdere il posto in coda. Lo lancia e1_launch.sh, che sceglie --mem e --time per cella:
#
#   WBES_E1_CELL=c2m|c2f|c3f|g1 aau/submit.sh evidence/e1_factorial/e1_train.sbatch --mem=... --time=...
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
exec bash "$WBES_ROOT/aau/evidence/e1_factorial/e1_train_body.sh"
