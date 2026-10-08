#!/usr/bin/env bash
#SBATCH --job-name=wbes-ugt-train-gt
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#
# E11, dati: GT unificata delle 65.600 identita' del run su scala (train_gt.py), poi il controllo col
# loader del trainer v2 in sola lettura (check_train_gt.py, venv del trainer).
#
#   sbatch v3_work/unified_gt/train_gt.sbatch
#
# Memoria: la matrice 65.600^2 float32 e' 17.2 GB; il loader del trainer la converte in float64
# (34.4 GB) come fa nel run, piu' la copia float32 durante la conversione.
set -euo pipefail
export UGT_THREADS=32
v3_work/unified_gt/run.sh v3_work/unified_gt/train_gt.py
export OMP_NUM_THREADS=8
aau/run.sh v3_work/unified_gt/check_train_gt.py
echo "[train-gt] OK"
