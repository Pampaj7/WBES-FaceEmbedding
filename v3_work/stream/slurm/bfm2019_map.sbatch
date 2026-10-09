#!/usr/bin/env bash
#SBATCH --job-name=wbes-stream-bfm19map
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#
# Mappa unificata di BFM 2019 (bfm2019_map.py, venv .venv_ugt), una volta sola: datasets/STREAM/maps/bfm2019.npz.
set -euo pipefail
export AAU_NV=
export UGT_THREADS=8
v3_work/unified_gt/run.sh v3_work/stream/bfm2019_map.py
