#!/usr/bin/env bash
#SBATCH --job-name=wbes-d5
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#
# Diagnostica D5 (PROTOCOL_D5.md), passi CPU, sul nodo della valutazione di D3 e2 per K0 (--nodelist=i256-a10-10):
#   D5_STEP=test  sbatch --cpus-per-task=4 --mem=16G --time=00:20:00 aau/diagnostics/d5.sbatch   (prove di test_d5.py)
#   D5_STEP=fit   sbatch --nodelist=i256-a10-10 aau/diagnostics/d5.sbatch   (teste in d5/heads, ripartibile)
#   D5_STEP=eval  sbatch --nodelist=i256-a10-10 aau/diagnostics/d5.sbatch   (dopo fit)
#   D5_STEP=all   fit poi eval nello stesso job
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
STEP="${D5_STEP:?D5_STEP: test | fit | eval | all}"
W="${SLURM_CPUS_PER_TASK:-8}"
export AAU_NV=""
D=aau/diagnostics
echo "[d5] host=$(hostname) job=${SLURM_JOB_ID:-none} passo=$STEP $(date +%T)"
t0=$(date +%s)
case "$STEP" in
    test) aau/run.sh $D/test_d5.py ;;
    fit)  UGT_THREADS=1 v3_work/unified_gt/run.sh $D/d5_fewshot.py fit --workers "$W" ;;
    eval) UGT_THREADS=1 v3_work/unified_gt/run.sh $D/d5_fewshot.py eval --workers "$W" ;;
    all)  UGT_THREADS=1 v3_work/unified_gt/run.sh $D/d5_fewshot.py fit --workers "$W"
          UGT_THREADS=1 v3_work/unified_gt/run.sh $D/d5_fewshot.py eval --workers "$W" ;;
    *)    echo "passo sconosciuto: $STEP"; exit 2 ;;
esac
echo "[d5] passo $STEP finito in $(( $(date +%s) - t0 ))s"
