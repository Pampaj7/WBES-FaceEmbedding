#!/usr/bin/env bash
#SBATCH --job-name=wbes-v3-resume
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#
# Ripresa da checkpoint del trainer v3 con un VERO requeue di Slurm (sulle A100: -p aicentre-a100
# --qos=unprivileged --gres=gpu:a100:1). Dati di prova a 2 blocchi, 60 passi da 10 per epoca, eval ogni 3 epoche,
# EMA, cache compatta, pre-pass con grad_vec, checkpoint di ripresa ogni 6 s (anche a meta' epoca).
#   1. primo avvio: run I in background; al primo checkpoint A META' EPOCA oltre il passo 20,
#      `scontrol requeue` di questo job (Slurm lo uccide e lo rimette in coda);
#   2. riavvio: run I riparte da solo (--resume auto, stessa run dir) e finisce; poi R e R2 senza interruzioni;
#   3. tests/compare_resume.py: I contro R (ripresa) e R2 contro R (pavimento di non determinismo della GPU).
set -uo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
source "$AAU_DIR/data_scale/recipe_v1.sh"
TD="$AAU_RUNS/evidence/trainer_v3/testdata"
OUT="$AAU_RUNS/evidence/trainer_v3${WBES_V3_SUITE_TAG:+/$WBES_V3_SUITE_TAG}/resume_${WBES_V3_RESUME_TAG:+${WBES_V3_RESUME_TAG}_}${SLURM_JOB_ID:-manual}"
rm -rf "/tmp/${SLURM_JOB_ID:-manual}"_r*          # /tmp di un tentativo ucciso dal requeue (il trap non gira)
STAGE="/tmp/${SLURM_JOB_ID:-manual}_r${SLURM_RESTART_COUNT:-0}"
mkdir -p "$OUT" "$STAGE"
trap 'rm -rf "$STAGE"' EXIT
echo "[resume] $(date +%T) avvio, restart=${SLURM_RESTART_COUNT:-0} host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader || true
run() {  # $1 nome
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$AAU_DIR/run.sh" v3_work/trainer/train_v3.py \
    --total-steps 60 --steps-per-epoch "${WBES_V3_RESUME_S:-10}" --split-json "$TD/${WBES_V3_RESUME_SPLIT:-split.json}" --data-spec "$TD/spec.json" \
    --stage-root "$STAGE/$1" --prepass-proc 12 --train-threads 8 --eval_domain bfm --gt-keep-scale \
    --lr-steps 45:5e-5 --cache-workers 8 --cache-max-gb 60 "${RECIPE_V1[@]}" --epochs 6 --eval_every 3 \
    --save_every 3 --seed 1234 --dist_npz "$TD/gt.npz" --runs_root "$OUT/runs_$1" --ema-decay 0.999 \
    --compact-cache --fast-data --ckpt-minutes 0.1 --log-every 5
}
if (( ${SLURM_RESTART_COUNT:-0} == 0 )); then
  run I >> "$OUT/I.log" 2>&1 &
  PID=$!
  until grep -qE "checkpoint di ripresa: epoca [0-9]+, batch [1-9][0-9]*, passo ([2-9][0-9])" "$OUT/I.log" 2> /dev/null; do
    sleep 2
    kill -0 "$PID" 2> /dev/null || { echo "[resume] ERRORE: I finito prima del requeue"; exit 1; }
  done
  grep -E "checkpoint di ripresa" "$OUT/I.log" | tail -1
  echo "[resume] $(date +%T) scontrol requeue $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID"
  sleep 600
  exit 0
fi
echo "[resume] $(date +%T) ripresa di I"
run I >> "$OUT/I.log" 2>&1
grep -E "RIPRESA|generatori" "$OUT/I.log"
echo "[resume] $(date +%T) R e R2 senza interruzioni"
run R > "$OUT/R.log" 2>&1
run R2 > "$OUT/R2.log" 2>&1
"$AAU_DIR/run.sh" v3_work/trainer/tests/compare_resume.py --i "$OUT/runs_I" --r "$OUT/runs_R" --r2 "$OUT/runs_R2" \
  --out "$OUT/compare.json" | tee "$OUT/compare.txt"
for f in "$OUT"/*.log; do tr '\r' '\n' < "$f" | grep -v '%|' > "$f.tmp" && mv "$f.tmp" "$f"; done
