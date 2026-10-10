#!/usr/bin/env bash
#SBATCH --job-name=wbes-stream-smoke2
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:2
#SBATCH --cpus-per-gpu=20
#SBATCH --mem-per-gpu=80G
#SBATCH --time=00:25:00
#
# Variante a 2 L40S di trial_validated.sbatch (stesse misure), da lanciare prima della prova a 12 GPU:
# 1000 passi da 100, eval online ogni 5 epoche.
#   sbatch v3_work/stream/slurm/smoke_validated_2gpu.sbatch
set -euo pipefail
export STREAM_STEPS="${STREAM_STEPS:-1000}" STREAM_SPE="${STREAM_SPE:-100}" STREAM_EVAL_EVERY="${STREAM_EVAL_EVERY:-5}"
export STREAM_OUT="${STREAM_OUT:-/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/stream/smoke_validated_${SLURM_JOB_ID:-manual}}"
exec bash v3_work/stream/slurm/trial_validated.sbatch
