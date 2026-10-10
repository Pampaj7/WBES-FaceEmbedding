#!/usr/bin/env bash
#SBATCH --job-name=wbes-stream-trial12
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:6
#SBATCH --cpus-per-gpu=20
#SBATCH --mem-per-gpu=80G
#SBATCH --time=00:45:00
#
# Prova breve della configurazione del run massivo con il preset validated (bfm2019, ict, gnm) e la ricetta di
# C3F/C3M scalata (massive_node.sh, STREAM_RECIPE c3m): 12 L40S = 6 + 6, QoS normal, nessun --requeue, un solo
# tentativo di torchrun.
#   sbatch v3_work/stream/slurm/trial_validated.sbatch                 (12 GPU, ~40 min)
#   sbatch v3_work/stream/slurm/smoke_validated_2gpu.sbatch            (2 GPU, ~20 min: da lanciare per prima)
# Cosa misura (aau/runs/evidence/stream/trial_validated_<job>/node0, summarize_run.py a fine job): s/passo e mesh/s
# per epoca, riuso delle viste (a regime e cumulato), loss per epoca (deve scendere), lr per epoca (deve restare 1e-4),
# batch per dominio e per fonti su tutto il run (nessun batch misto, nemmeno per fonti), quota d'espressioni per
# dominio dei produttori, dominio e fonti delle identita' dei primi 200 batch per rank
# (runs/*/batch_domains_rank<r>.csv). Variabili di massive.sbatch sovrascrivibili (STREAM_STEPS ...).
set -euo pipefail
export STREAM_SOURCES=validated
export STREAM_STEPS="${STREAM_STEPS:-2400}" STREAM_SPE="${STREAM_SPE:-200}" STREAM_EVAL_EVERY="${STREAM_EVAL_EVERY:-6}"
export STREAM_ATTEMPTS="${STREAM_ATTEMPTS:-0}"
export STREAM_OUT="${STREAM_OUT:-/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/stream/trial_validated_${SLURM_JOB_ID:-manual}}"
exec bash v3_work/stream/slurm/massive.sbatch
