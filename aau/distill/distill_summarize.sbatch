#!/usr/bin/env bash
#SBATCH --job-name=wbes-distill-sum
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#
# Tabelle del pilota della distillazione (aau/distill/distill_summarize.py), entrambi i domini.
#
#   aau/submit.sh distill/distill_summarize.sbatch
#   WBES_DISTILL_OUT=<dir con protocol.md> WBES_DISTILL_STUDENT_ROOT=<eval dir> ...   prova su altri dati
#   v2: WBES_DISTILL_OUT=aau/runs/distill_v2 WBES_DISTILL_ABLATION_ROOT=aau/runs/distill_v2/eval_nognm \
#       WBES_DISTILL_SUM_ARGS="--reference ict" WBES_DISTILL_STUDENT_LABEL="Studente v2" ...
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv

OUT="${WBES_DISTILL_OUT:-aau/runs/distill_pilot}"
echo "[distill-sum] host=$(hostname) job=${SLURM_JOB_ID:-none} out=$OUT"
for dom in ${WBES_DISTILL_DOMAINS:-fv_expr hifi3d}; do
    extra=()
    [[ -n "${WBES_DISTILL_STUDENT_ROOT:-}" ]] && extra=(--student-root "$WBES_DISTILL_STUDENT_ROOT/$dom")
    [[ -n "${WBES_DISTILL_ABLATION_ROOT:-}" ]] && extra+=(--ablation-root "$WBES_DISTILL_ABLATION_ROOT/$dom")
    [[ -n "${WBES_DISTILL_STUDENT_LABEL:-}" ]] && extra+=(--student-label "$WBES_DISTILL_STUDENT_LABEL")
    # shellcheck disable=SC2086
    "$AAU_DIR/run.sh" aau/distill/distill_summarize.py --domain "$dom" --out-dir "$OUT" "${extra[@]}" ${WBES_DISTILL_SUM_ARGS:-}
done
echo "[distill-sum] OK"
