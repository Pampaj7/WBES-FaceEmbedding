#!/usr/bin/env bash
#SBATCH --job-name=wbes-ws2-views
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#
# WS2, tabella cross-3DMM: ricostruisce gli split di training dei tre modelli, sceglie i
# soggetti di eval di ogni cella (solo held-out del modello valutato, controllo leak con
# assert) e scrive le viste a symlink in datasets/WS2_CROSS3DMM/. Dettagli in ws2_views.py.
# Serve numpy e il codice del training, quindi gira nel container, ma senza GPU.
#
#   aau/submit.sh cross3dmm/ws2_views.sbatch
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv

# Nodo senza GPU: --nv farebbe fallire singularity per mancanza di libnvidia.
export AAU_NV=""

echo "[ws2-views] host=$(hostname) job=${SLURM_JOB_ID:-none}"
"$AAU_DIR/run.sh" aau/cross3dmm/ws2_views.py "$@"
