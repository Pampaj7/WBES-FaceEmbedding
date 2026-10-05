#!/usr/bin/env bash
#SBATCH --job-name=wbes-ws3b-an
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#
# WS3b, passo 4: classifica dei tre metodi per criterio, Kendall tau fra criteri con il
# metro dell'ipotesi nulla, split-half di affidabilita', AUC di identita'.
#
#   aau/submit.sh recon/ws3b_analysis.sbatch
#
# Niente GPU e niente venv delle baseline: legge solo i csv di ws3b_geometric e ws3b_latent.
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv

echo "[ws3b-an] host=$(hostname) job=${SLURM_JOB_ID:-none}"

"$AAU_DIR/run.sh" aau/recon/ws3b_analysis.py ${WBES_WS3B_ARGS:-} "$@"

echo "[ws3b-an] OK"
