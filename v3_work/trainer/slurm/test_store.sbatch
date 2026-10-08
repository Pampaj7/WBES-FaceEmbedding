#!/usr/bin/env bash
#SBATCH --job-name=wbes-v3-store
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=aicentre-a100
#SBATCH --qos=unprivileged
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#
# Store in mmap contro staging normale: stessi dati di prova su viste (operatori su disco, identici), stesso run
# v3 (24 passi, 2 blocchi, 4 eval) letto (A) con lo staging a blocchi e (B) dallo store (tools/build_store.py):
# log, eval e pesi devono coincidere.
set -uo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
source "$AAU_DIR/data_scale/recipe_v1.sh"
TD="$AAU_RUNS/evidence/trainer_v3/testdata"
OUT="$AAU_RUNS/evidence/trainer_v3/a100/store_test"
STAGE="/tmp/${SLURM_JOB_ID:-manual}"
rm -rf "$OUT"; mkdir -p "$OUT" "$STAGE"
trap 'rm -rf "$STAGE"' EXIT
"$AAU_DIR/run.sh" v3_work/trainer/tools/build_store.py --spec "$TD/spec.json" --split "$TD/split_views.json" \
  --out "$STAGE/store" --workers 8 --shards 8 --verify 200 2>&1 | grep -v Warning | tee "$OUT/build.log"
run() {  # $1 nome, resto: flag
  local n="$1"; shift
  OMP_NUM_THREADS=8 "$AAU_DIR/run.sh" v3_work/trainer/train_v3.py --total-steps 24 --steps-per-epoch 3 \
    --split-json "$TD/split_views.json" --data-spec "$TD/spec.json" --prepass-proc 8 --train-threads 8 \
    --eval_domain bfm --gt-keep-scale --cache-workers 8 --cache-max-gb 60 "${RECIPE_V1[@]}" --epochs 8 \
    --eval_every 2 --save_every 2 --seed 1234 --dist_npz "$TD/gt.npz" --runs_root "$OUT/runs_$n" --fast-data \
    --compact-cache --ema-decay 0.999 "$@" > "$OUT/$n.log" 2>&1
}
run A --stage-root "$STAGE/stage"
run B --store "$STAGE/store"
"$AAU_DIR/run.sh" v3_work/trainer/tests/compare_runs.py --a "$OUT/runs_A" --b "$OUT/runs_B" --out "$OUT/compare.json" \
  | tee "$OUT/compare.txt"
