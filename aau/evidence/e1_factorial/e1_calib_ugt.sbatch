#!/usr/bin/env bash
#SBATCH --job-name=wbes-e1-calib-ugt
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#
# E1: GT unificata tarata sulle mediane della maxabs (e1_calib_ugt.py build), poi il controllo col loader del
# trainer (check -> datasets/UNIFIED_GT/train/check_calib.json). Memoria: due matrici 65.600^2 float32 (17.2 GB
# ciascuna) piu' 6 GB per la mediana del blocco ICT; nel controllo la matrice in float64 (34.4 GB) come nel trainer.
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export AAU_NV="" OMP_NUM_THREADS=8
"$AAU_DIR/run.sh" aau/evidence/e1_factorial/e1_calib_ugt.py build
"$AAU_DIR/run.sh" aau/evidence/e1_factorial/e1_calib_ugt.py check
echo "[calib] OK"
