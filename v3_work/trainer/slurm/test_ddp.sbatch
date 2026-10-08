#!/usr/bin/env bash
#SBATCH --job-name=wbes-v3-ddp
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=01:30:00
#
# DDP del trainer v3 su 2 GPU dello stesso nodo, torchrun elastico, dati di prova (testdata/, 2 blocchi):
#   ddp_default  flag v3 di default, 2 rank, 40 passi da 4 per epoca; il rank 1 esce al passo 18 al primo
#                tentativo (--test-crash-at-step): torchrun riavvia tutti, che riprendono da last.pth
#   ddp_new      tutti i flag nuovi insieme: packed, bilanciato alpha 0, area_attn, sqrt(area), EMA, cache
#                compatta; 40 passi da 10
#   resume_w1    ripresa di ddp_default su UNA GPU (world 2 -> 1) fino a 60 passi
# Ogni epoca in DDP controlla che i pesi siano identici sui rank (train_v3.check_sync).
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
source "$AAU_DIR/data_scale/recipe_v1.sh"
STAGE="/tmp/${SLURM_JOB_ID:-manual}"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE"
TD="$AAU_RUNS/evidence/trainer_v3/testdata"
OUT="${WBES_V3_OUT:-$AAU_RUNS/evidence/trainer_v3${WBES_V3_SUITE_TAG:+/$WBES_V3_SUITE_TAG}/ddp}"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

common() {  # $1 nome, $2 passi totali, $3 passi per epoca
  echo --total-steps "$2" --steps-per-epoch "$3" --split-json "$TD/split.json" --data-spec "$TD/spec.json" \
    --stage-root "$STAGE/$1/stage" --prepass-proc 7 --train-threads 4 --eval_domain bfm --gt-keep-scale \
    --cache-workers 8 --cache-max-gb 60 "${RECIPE_V1[@]}" --epochs 4 --eval_every "${WBES_V3_EVAL_EVERY:-5}" --save_every 5 --seed 1234 \
    --dist_npz "$TD/gt.npz" --runs_root "$OUT/runs_$1" --log-every 5
}
TRUN=(-m torch.distributed.run --standalone --nproc-per-node 2 --max-restarts 1)
X=()   # senza GPU (logica DDP con gloo su CPU) le attivazioni stanno in RAM: 3 mesh per soggetto invece di 6
if ! nvidia-smi -L > /dev/null 2>&1; then
  X=(--max_meshes_per_subject_train 3); echo "[ddp] nessuna GPU: gloo su CPU, ${X[*]}"
fi

echo "[ddp] ddp_default (crash al passo 18, riavvio elastico)"
"$AAU_DIR/run.sh" "${TRUN[@]}" v3_work/trainer/train_v3.py $(common ddp_default 40 4) "${X[@]}" --test-crash-at-step 18 \
    > "$OUT/ddp_default.log" 2>&1 || echo "[ddp] ddp_default rc=$?"
echo "[ddp] ddp_new (packed, bilanciato, area_attn, sqrt_area, EMA, cache compatta)"
"$AAU_DIR/run.sh" "${TRUN[@]}" v3_work/trainer/train_v3.py $(common ddp_new 40 10) "${X[@]}" --forward packed \
    --sampler balanced --domain-alpha 0.0 --pooling area_attn --input-norm sqrt_area --ema-decay 0.999 \
    --compact-cache > "$OUT/ddp_new.log" 2>&1 || echo "[ddp] ddp_new rc=$?"
LAST=$(find "$OUT/runs_ddp_default" -name last.pth | head -1)
echo "[ddp] resume_w1 da $LAST (world 1)"
CUDA_VISIBLE_DEVICES=0 "$AAU_DIR/run.sh" v3_work/trainer/train_v3.py $(common resume_w1 60 4) "${X[@]}" --resume "$LAST" \
    > "$OUT/resume_w1.log" 2>&1 || echo "[ddp] resume_w1 rc=$?"
for f in "$OUT"/*.log; do tr '\r' '\n' < "$f" | grep -v '%|' > "$f.tmp" && mv "$f.tmp" "$f"; done
grep -h -E "RIPRESA|crash di prova|epoca [0-9]+:|passi eseguiti|DDP|Error|error" "$OUT"/*.log | head -80
