#!/usr/bin/env bash
#SBATCH --job-name=wbes-e12
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#
# E12, GT canonica: tutta la catena su CPU, nel venv della GT unificata (v3_work/unified_gt/run.sh).
#
#   sbatch v3_work/canonical_gt/run.sbatch                     # tutto
#   E12_STEPS="methods report" sbatch v3_work/canonical_gt/run.sbatch
#
#   gt       gt.py        GT canonica, verifiche (trasformazioni, unita', CV), Spearman fra GT
#   methods  methods.py   metodi esistenti rivalutati con la GT canonica
#   report   report.py    aau/runs/evidence/e12/summary.md
set -euo pipefail

STEPS="${E12_STEPS:-gt methods report}"
RUN="v3_work/unified_gt/run.sh"
export UGT_THREADS="${UGT_THREADS:-8}"
echo "[e12] host=$(hostname) job=${SLURM_JOB_ID:-none} steps: $STEPS"
for step in $STEPS; do
    t0=$(date +%s)
    case "$step" in
        gt)      $RUN v3_work/canonical_gt/gt.py ${E12_GT_ARGS:-} ;;
        methods) UGT_THREADS=1 $RUN v3_work/canonical_gt/methods.py ${E12_METHODS_ARGS:-} ;;
        report)  $RUN v3_work/canonical_gt/report.py ;;
        *) echo "passo sconosciuto: $step" >&2; exit 2 ;;
    esac
    echo "[e12] $step: $(( $(date +%s) - t0 )) s"
done
