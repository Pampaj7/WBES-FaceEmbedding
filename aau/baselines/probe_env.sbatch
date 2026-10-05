#!/usr/bin/env bash
#SBATCH --job-name=wbes-bl-probe
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#
# Ricognizione: dipendenze, rete dal nodo, dati, tempi per dimensionare i job veri.
#   aau/submit.sh baselines/probe_env.sbatch
# Su GPU (per verificare che anche i nodi prioritized abbiano rete):
#   aau/submit.sh baselines/probe_env.sbatch --partition=prioritized --gres=gpu:l40s:1
set -euo pipefail

# Se non e' stata chiesta una GPU, singularity va invocato senza --nv.
if [[ -z "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}${GPU_DEVICE_ORDINAL:-}" ]]; then
    export AAU_NV=""
fi
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

"$AAU_DIR/run.sh" aau/baselines/probe_env.py
