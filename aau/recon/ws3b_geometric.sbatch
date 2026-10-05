#!/usr/bin/env bash
#SBATCH --job-name=wbes-ws3b-geom
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#
# WS3b, passo 3a: errore stile NoW (con allineamento) e Chamfer grezza (senza), per tutte
# le ricostruzioni contro la GT, piu' la Chamfer sulle 8000 coppie dell'AUC di identita'.
#
#   aau/submit.sh recon/ws3b_geometric.sbatch
#   aau/submit.sh recon/ws3b_geometric.sbatch --time=02:00:00   # opzioni di SBATCH in coda
#   WBES_WS3B_ARGS="--mode gt --overwrite" aau/submit.sh recon/ws3b_geometric.sbatch
#
# Gli argomenti dello SCRIPT passano da WBES_WS3B_ARGS: quello che si scrive dopo il nome
# del file in submit.sh finisce fra le opzioni di sbatch, non fra quelle del job.
#
# Niente GPU: cKDTree, Procrustes e l'AABB di libigl.  Ogni csv gia' presente viene
# saltato (senza --overwrite), quindi il job e' ripartibile.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv

WORKERS="${WBES_WS3B_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[ws3b-geom] host=$(hostname) job=${SLURM_JOB_ID:-none} workers=$WORKERS"

"$AAU_DIR/run.sh" aau/recon/ws3b_geometric.py --workers "$WORKERS" ${WBES_WS3B_ARGS:-} "$@"

echo "[ws3b-geom] OK"
