#!/usr/bin/env bash
#SBATCH --job-name=wbes-e1-gate-smoke
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#
# E1: cancello dei training su V100 (e1_gate_smoke.py). Parte dopo lo smoke V100 (afterany) e lo fa passare solo
# se l'epoca 1 su V100 torna con quella su L40S; i training dipendono da questo job (afterok).
#   WBES_E1_SMOKE_RUN=<run dir dello smoke> aau/submit.sh evidence/e1_factorial/e1_gate_smoke.sbatch --dependency=afterany:<smoke>
set -euo pipefail
cd "${WBES_ROOT:-$PWD}"
python3 aau/evidence/e1_factorial/e1_gate_smoke.py "${WBES_E1_SMOKE_RUN:?}"
