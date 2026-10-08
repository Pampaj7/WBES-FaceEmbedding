#!/usr/bin/env bash
#SBATCH --job-name=wbes-v3-storedbg
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=aicentre-a100
#SBATCH --qos=unprivileged
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#
# Traccia del primo passo (tests/debug_store_step.py) con lo staging (A) e con lo store (B), stessi argomenti
# di test_store.sbatch: le righe TRACE devono coincidere.
set -uo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
source "$AAU_DIR/data_scale/recipe_v1.sh"
TD="$AAU_RUNS/evidence/trainer_v3/testdata"
OUT="$AAU_RUNS/evidence/trainer_v3/a100/store_debug"
STAGE="/tmp/${SLURM_JOB_ID:-manual}"
rm -rf "$OUT"; mkdir -p "$OUT" "$STAGE"
trap 'rm -rf "$STAGE"' EXIT
"$AAU_DIR/run.sh" v3_work/trainer/tools/build_store.py --spec "$TD/spec.json" --split "$TD/split_views.json" \
  --out "$STAGE/store" --workers 8 --shards 8 --verify 0 2>&1 | grep '\[store\]'
run() {  # $1 nome, resto: flag
  local n="$1"; shift
  OMP_NUM_THREADS=8 "$AAU_DIR/run.sh" v3_work/trainer/tests/debug_store_step.py --total-steps 6 --steps-per-epoch 3 \
    --split-json "$TD/split_views.json" --data-spec "$TD/spec.json" --prepass-proc 8 --train-threads 8 \
    --eval_domain bfm --gt-keep-scale --cache-workers 8 --cache-max-gb 60 "${RECIPE_V1[@]}" --epochs 2 \
    --eval_every 0 --save_every 1 --seed 1234 --dist_npz "$TD/gt.npz" --runs_root "$STAGE/runs_$n" --fast-data \
    --compact-cache --ema-decay 0.999 --ckpt-minutes 0 "$@" > "$OUT/$n.log" 2>&1
  grep '^TRACE' "$OUT/$n.log" > "$OUT/$n.trace"
}
run A --stage-root "$STAGE/stage"
run B --store "$STAGE/store"
echo "righe TRACE: A $(wc -l < "$OUT/A.trace"), B $(wc -l < "$OUT/B.trace")"
diff "$OUT/A.trace" "$OUT/B.trace" > "$OUT/diff.txt" && echo "TRACCE IDENTICHE" || { echo "TRACCE DIVERSE"; head -20 "$OUT/diff.txt"; }
