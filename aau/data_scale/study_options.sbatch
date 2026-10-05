#!/usr/bin/env bash
#SBATCH --job-name=wbes-ds-study
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#
# Studio delle opzioni di scala dati (aau/data_scale/PLAN.md), 90 mesh ESISTENTI:
# nessun dato nuovo. Le viste temporanee (al massimo ~0.8 GB alla volta, una variante per
# volta, cancellata subito) stanno su /tmp del nodo, che e' RAM e conta contro --mem; su
# CephFS va solo aau/data_scale/study_options.json (pochi KB).
#
#   aau/submit.sh data_scale/study_options.sbatch
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv
aau_require_diffusion_net
export AAU_NV=

# i processi di ricalcolo sono a thread singolo; il forward del modello alza i thread da se'
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

WORK="/tmp/wbes-ds-study-${SLURM_JOB_ID:-manual}"
trap 'rm -rf "$WORK"' EXIT
echo "[study] host=$(hostname) job=${SLURM_JOB_ID:-none} work=$WORK"
"$AAU_DIR/run.sh" aau/data_scale/study_options.py --work-dir "$WORK" \
    --out-json "$AAU_DIR/data_scale/study_options.json" --n-cores "${SLURM_CPUS_PER_TASK:-16}"
