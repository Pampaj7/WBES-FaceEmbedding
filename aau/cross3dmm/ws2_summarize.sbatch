#!/usr/bin/env bash
#SBATCH --job-name=wbes-ws2-summarize
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#
# Tabella cross-3DMM di WS2 + riga espressioni + controllo leak sugli output, con i CI
# bootstrap del paper. Niente GPU: legge solo csv e json.
#
#   aau/submit.sh cross3dmm/ws2_summarize.sbatch --dependency=afterok:<job delle celle>
#
# WBES_WS2_SUM_BOOTSTRAP=200 per CI piu' grossolani e veloci. Il costo e' il bootstrap:
# 6 celle x (30 coppie di topologie + 2 aggregati + subject-pair-mean) x 2 metriche, piu'
# le espressioni, 1000 ricampionamenti ciascuno.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv

# Nodo senza GPU: --nv farebbe fallire singularity per mancanza di libnvidia.
export AAU_NV=""

opts=()
[[ -n "${WBES_WS2_SUM_BOOTSTRAP:-}" ]] && opts+=(--n-bootstrap "$WBES_WS2_SUM_BOOTSTRAP")

echo "[ws2-sum] host=$(hostname) job=${SLURM_JOB_ID:-none} opts=${opts[*]:-(default)}"
"$AAU_DIR/run.sh" aau/cross3dmm/ws2_summarize.py "${opts[@]}"
