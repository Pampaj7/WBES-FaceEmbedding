#!/usr/bin/env bash
#SBATCH --job-name=wbes-ir-sum
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#
# Riconoscimento in dominio: tabelle e tempi (aau/indomain/ir_summarize.py) ->
# aau/runs/indomain_recog/summary.md. Dopo ir_embed, ir_facebench (tutti gli shard), ir_arcface e
# ir_timing.
#
#   aau/submit.sh indomain/ir_summarize.sbatch --dependency=afterok:<...>
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv
if [[ ! -f "$AAU_RUNS/indomain_recog/protocol.md" ]]; then
    echo "ERRORE: aau/runs/indomain_recog/protocol.md assente: il protocollo va dichiarato prima" >&2
    exit 1
fi
echo "[ir-sum] host=$(hostname) job=${SLURM_JOB_ID:-none}"
"$AAU_DIR/run.sh" aau/indomain/ir_summarize.py ${WBES_IR_SUM_ARGS:-}
