#!/usr/bin/env bash
#SBATCH --job-name=wbes-e23-sum
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#
# Riepiloghi (solo CPU, nessun calcolo su mesh): E2 (summarize_e2.py), E3b (summarize_e3b.py), E3 ed E3c
# (breakdown.py). WBES_SUM_PARTS sceglie quali (default: tutti e tre).
#
#   aau/submit.sh evidence/e3_breakdown/summary.sbatch
#   WBES_SUM_PARTS=e3 aau/submit.sh evidence/e3_breakdown/summary.sbatch
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export AAU_NV= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for part in ${WBES_SUM_PARTS:-e2 e3b e3}; do
    case "$part" in
        e2)  "$AAU_DIR/run.sh" aau/evidence/e2_canon/summarize_e2.py ;;
        e3b) "$AAU_DIR/run.sh" aau/evidence/e3_breakdown/summarize_e3b.py ;;
        e3)  "$AAU_DIR/run.sh" aau/evidence/e3_breakdown/breakdown.py ;;
        *) echo "ERRORE: parte $part" >&2; exit 2 ;;
    esac 2>&1 | grep -v "valid test operator"
done
