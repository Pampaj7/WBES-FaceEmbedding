#!/usr/bin/env bash
#SBATCH --job-name=wbes-distill-teacher
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#
# Etichette dell'insegnante (ArcFace su normal map delle original) per il pilota della distillazione,
# BFM e ICT-5000 in serie (aau/distill/teacher_labels.py). Output in
# aau/runs/distill_pilot/teacher<suffisso>/<dominio>; render su /tmp, cancellati in uscita.
#
#   WBES_TEACHER_LIMIT=24 aau/submit.sh distill/teacher_labels.sbatch --time=00:20:00   misura del costo
#   WBES_TEACHER_FRAME_CHECK=4 ...                                                     solo controllo frame
#   aau/submit.sh distill/teacher_labels.sbatch                                         tutto
#   WBES_TEACHER_DOMAINS=gnm WBES_TEACHER_OUT=aau/runs/distill_v2/teacher \
#     WBES_TEACHER_EXTRA="--camera-sample 3000" aau/submit.sh distill/teacher_labels.sbatch   v2
#
# CPU e niente GPU, come zs_arcface.sbatch: un thread per processo onnxruntime (arcface_fixed).
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

LIMIT="${WBES_TEACHER_LIMIT:-0}"
FC="${WBES_TEACHER_FRAME_CHECK:-0}"
SUFFIX=""
(( LIMIT > 0 )) && SUFFIX="_limit$LIMIT"
(( FC > 0 )) && SUFFIX="_framecheck"
OUT="${WBES_TEACHER_OUT:-$AAU_RUNS/distill_pilot/teacher$SUFFIX}"
STAGE="/tmp/wbes_teacher_${SLURM_JOB_ID:-manual}"
CORES="${SLURM_CPUS_PER_TASK:-8}"
rm -rf "$STAGE"
trap 'rm -rf "$STAGE"' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "[teacher] host=$(hostname) job=${SLURM_JOB_ID:-none} out=$OUT limit=$LIMIT frame_check=$FC"
for dom in ${WBES_TEACHER_DOMAINS:-bfm ict}; do
    # shellcheck disable=SC2086
    "$AAU_DIR/baselines/run_bl.sh" aau/distill/teacher_labels.py --domain "$dom" --stage "$STAGE/$dom" \
        --out-dir "$OUT/$dom" --workers "$CORES" --limit "$LIMIT" --frame-check "$FC" ${WBES_TEACHER_EXTRA:-}
    echo "[teacher] $dom: /tmp $(du -sh "$STAGE/$dom" | cut -f1)"
    rm -rf "$STAGE/$dom"
done
echo "[teacher] OK"
