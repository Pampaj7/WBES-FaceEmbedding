#!/usr/bin/env bash
#SBATCH --job-name=wbes-d3e2
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#
# Diagnostica D3, emendamento 2 (POST HOC, PROTOCOL_D3_emendamento_2.md), passi CPU:
#   D3E2_STEP=test                                   sbatch aau/diagnostics/d3_e2.sbatch   (prove di test_d3_e2.py)
#   D3E2_STEP=cv D3E2_ARM=<braccio>                  sbatch aau/diagnostics/d3_e2.sbatch   (uno per braccio, ripartibile)
#   D3E2_EXTRA="--tag _b --reverse": aiutante su un altro nodo (cache a parte, dai fit piu' veloci), poi eval li unisce
#   D3E2_STEP=eval                                   sbatch aau/diagnostics/d3_e2.sbatch   (dopo i tre cv)
# File nuovo: d3.sbatch e i moduli importati dalla catena dell'ablazione 2x2 (diag.py, d1_stats.py) restano invariati.
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
STEP="${D3E2_STEP:?D3E2_STEP: test | cv | eval}"
W="${SLURM_CPUS_PER_TASK:-8}"
export AAU_NV=""
D=aau/diagnostics
echo "[d3e2] host=$(hostname) job=${SLURM_JOB_ID:-none} passo=$STEP braccio=${D3E2_ARM:-} $(date +%T)"
t0=$(date +%s)
case "$STEP" in
    test) aau/run.sh $D/test_d3_e2.py ;;
    cv)   UGT_THREADS=1 v3_work/unified_gt/run.sh $D/d3_e2.py cv --arm "${D3E2_ARM:?D3E2_ARM}" --workers "$W" ${D3E2_EXTRA:-} ;;
    eval) UGT_THREADS=1 v3_work/unified_gt/run.sh $D/d3_e2.py eval --workers "$W" ;;
    *)    echo "passo sconosciuto: $STEP"; exit 2 ;;
esac
echo "[d3e2] passo $STEP finito in $(( $(date +%s) - t0 ))s"
