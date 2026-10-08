#!/usr/bin/env bash
#SBATCH --job-name=wbes-v3-eq2
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#
# Equivalenza v2 = v3, livello 2 (traiettoria): la catena v2 VERA (v2_work/fastio/train_steps.py, flag del run
# su scala) due volte e il trainer v3 con i flag di default, sugli stessi dati di prova a 2 blocchi
# (testdata/: 15 BFM, 30 ICT-5000, 30 ICT_SCALE, 30 GNM; eval online 16 BFM + 16 GNM come il run su scala),
# 40 passi in 4 epoche da 10, eval ogni 2 epoche, lr 1e-4 -> 5e-5 dal passo 31, cambio di blocco all'epoca 3.
# Poi tests/compare_runs.py: log per epoca, eval online, pesi alle epoche 2 e 4.
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
source "$AAU_DIR/data_scale/recipe_v1.sh"
STAGE="/tmp/${SLURM_JOB_ID:-manual}"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE"
TD="$AAU_RUNS/evidence/trainer_v3/testdata"
OUT="${WBES_V3_OUT:-$AAU_RUNS/evidence/trainer_v3${WBES_V3_SUITE_TAG:+/$WBES_V3_SUITE_TAG}/equivalence_l2}"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

common() {  # $1 = nome del run
  echo --total-steps "${WBES_V3_L2_T:-40}" --steps-per-epoch "${WBES_V3_L2_S:-10}" \
    --split-json "$TD/${WBES_V3_L2_SPLIT:-split.json}" --data-spec "$TD/spec.json" \
    --stage-root "$STAGE/$1/stage" --prepass-proc 7 --train-threads 8 --domain-blocked --eval_domain bfm \
    --gt-keep-scale --lr-steps "${WBES_V3_L2_LR:-30:5e-5}" --cache-residency ram --cache-workers 16 --cache-max-gb 60 \
    --frame current "${RECIPE_V1[@]}" --epochs "${WBES_V3_L2_EPOCHS:-4}" --eval_every "${WBES_V3_L2_EVAL_EVERY:-2}" \
    --save_every 2 --seed 1234 \
    --data_dir "$STAGE/$1/stage" --dist_npz "$TD/gt.npz" --runs_root "$STAGE/$1/runs"
}
for r in v2a v2b; do
  echo "[eq2] $r: catena v2 (train_steps.py)"
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$AAU_DIR/run.sh" v2_work/fastio/train_steps.py $(common $r) \
    > "$OUT/$r.log" 2>&1
done
echo "[eq2] v3: train_v3.py, flag v3 di default"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$AAU_DIR/run.sh" v3_work/trainer/train_v3.py $(common v3) --prepass-grad orig --ckpt-minutes 0 \
    > "$OUT/v3.log" 2>&1
"$AAU_DIR/run.sh" v3_work/trainer/tests/compare_runs.py --a "$STAGE/v2a/runs" --b "$STAGE/v3/runs" \
    --floor "$STAGE/v2b/runs" --out "$OUT/compare.json" | tee "$OUT/compare.txt"
for r in v2a v2b v3; do
  d=$(dirname "$(find "$STAGE/$r/runs" -name train_log.csv | head -1)")
  mkdir -p "$OUT/$r"; cp "$d"/*.csv "$d"/*.txt "$OUT/$r/" 2> /dev/null || true
done
for r in v2a v2b v3; do tr '\r' '\n' < "$OUT/$r.log" | grep -v '%|' > "$OUT/$r.log.tmp" && mv "$OUT/$r.log.tmp" "$OUT/$r.log"; done
