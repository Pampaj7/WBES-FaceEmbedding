#!/usr/bin/env bash
#SBATCH --job-name=wbes-fd
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#
# FaMoS come confine di dominio (aau/famos_dominio/fd.py; regole in aau/runs/evidence/famos_dominio/REGOLE.md):
# statistiche delle grezze e delle patch, analisi H1/H2/H3 sulle distanze di esterna_famos_id, figura. Solo CPU.
#   sbatch aau/famos_dominio/fd.sbatch                 (FD_STEP=all | stats | analyze | fig)
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export AAU_NV=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
STEP="${FD_STEP:-all}"
W="${SLURM_CPUS_PER_TASK:-8}"
echo "[fd] host=$(hostname) job=${SLURM_JOB_ID:-none} passo=$STEP cpu=$W $(date +%T)"
if [[ $STEP == all || $STEP == stats ]]; then
    aau/run.sh -u aau/famos_dominio/fd.py stats --workers "$W"
fi
if [[ $STEP == all || $STEP == analyze ]]; then
    aau/run.sh -u aau/famos_dominio/fd.py analyze
fi
if [[ $STEP == all || $STEP == fig ]]; then
    aau/run.sh -u aau/runs/evidence/famos_dominio/fig_famos_expr.py
fi
echo "[fd] fine $(date +%T)"
