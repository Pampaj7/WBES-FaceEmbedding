#!/usr/bin/env bash
#SBATCH --job-name=wbes-mf-prep
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#
# WS3a, passo 1: le 2848 mesh tracciate di Multiface in sei topologie (17088 npz).
#
#   aau/submit.sh multiface/multiface_prep.sbatch
#   WBES_MF_FRAMES=20 aau/submit.sh multiface/multiface_prep.sbatch   # prova rapida
#
# Le sei topologie sono tracked/remesh/down (gentili) e crop/noisy/up (dure): lo script
# costruisce solo le varianti mancanti, quindi aggiungere le tre dure a una prep gia'
# fatta non ritocca le altre.
#
# 8 core e non 24 perche' la partizione cpu ha UN SOLO nodo da 32 core, condiviso con il
# precompute degli operatori (ICT e Multiface). Riavviabile: salta i frame gia' completi.
# Alla fine gira il self-check sui primi 6 frame.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv

IN_DIR="${WBES_MF_FULL_DIR:-$WBES_ROOT/datasets/Multiface/full}"
OUT_DIR="${WBES_MF_PREP_DIR:-$WBES_ROOT/datasets/Multiface/prep}"
N_CORES="${WBES_MF_CORES:-${SLURM_CPUS_PER_TASK:-8}}"
N_FRAMES="${WBES_MF_FRAMES:-0}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[mf-prep] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[mf-prep] in=$IN_DIR out=$OUT_DIR cores=$N_CORES frames=$N_FRAMES"

"$AAU_DIR/run.sh" aau/multiface/prepare_multiface.py \
    --full-dir "$IN_DIR" --out-dir "$OUT_DIR" \
    --n-cores "$N_CORES" --n-frames "$N_FRAMES"

for topology in tracked remesh down crop noisy up; do
    n=$(find "$OUT_DIR/$topology" -maxdepth 1 -name '*.npz' -type f | wc -l)
    echo "[mf-prep] $topology: $n npz, $(du -sh "$OUT_DIR/$topology" | cut -f1)"
done

echo "[mf-prep] verifica sui primi 6 frame"
"$AAU_DIR/run.sh" aau/multiface/prepare_multiface.py --demo --out-dir "$OUT_DIR"
