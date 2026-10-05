#!/usr/bin/env bash
#SBATCH --job-name=wbes-ws3a-analysis
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#
# WS3a, tabella finale: AUC same/different con CI bootstrap per soggetto sui csv delle
# distanze.  Va lanciato DOPO ws3a_geometric / ws3a_perceptual / ws3a_latent; le metriche
# che non hanno ancora il csv vengono saltate con un avviso, quindi si puo' lanciare anche
# a meta' strada per guardare quello che c'e'.
#
#   aau/submit.sh multiface/ws3a_analysis.sbatch
#   aau/submit.sh multiface/ws3a_analysis.sbatch --n-bootstrap 10000
#   WBES_WS3A_PAIRS=hard WBES_WS3A_SUMMARY=summary_hard WBES_WS3A_BASELINE=tracked,tracked \
#       aau/submit.sh multiface/ws3a_analysis.sbatch
#
# Il nome del summary e la coppia di riferimento passano da variabili d'ambiente e non da
# riga di comando: `submit.sh` mette i suoi argomenti extra fra le opzioni di sbatch, non
# fra quelle dello script, quindi `--summary-name` li' verrebbe letto da sbatch.
#
# Serve un job solo perche' il frontend non ha numpy (niente pip, niente conda): il conto
# in se' e' un minuto scarso, sono 24 csv x 3 confronti x 1000 repliche di bincount.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv

SUMMARY_NAME="${WBES_WS3A_SUMMARY:-summary}"
BASELINE_PAIR="${WBES_WS3A_BASELINE:-}"

echo "[ws3a-an] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[ws3a-an] out=${WBES_WS3A_OUT:-$AAU_RUNS/multiface_ws3a} coppie=${WBES_WS3A_PAIRS:-clean}"
echo "[ws3a-an] summary=$SUMMARY_NAME baseline='$BASELINE_PAIR'"

"$AAU_DIR/run.sh" aau/multiface/ws3a_analysis.py \
    --summary-name "$SUMMARY_NAME" --baseline-pair "$BASELINE_PAIR" "$@"

echo "[ws3a-an] OK"
