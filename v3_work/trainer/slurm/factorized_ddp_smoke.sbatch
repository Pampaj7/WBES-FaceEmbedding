#!/usr/bin/env bash
#SBATCH --job-name=wbes-v3-fact-ddp
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=aicentre-a100
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#
# DDP del trainer v3 con la modalita' fattorizzata completa (come il run C3M: global + factorized + scala + arearobust +
# bal + EMA + cache compatta), dati di prova, 2 GPU, 40 passi da 4: crash di prova al passo 18 (torchrun riavvia,
# ripresa da last.pth) e controllo dei pesi identici sui rank a ogni epoca (train_v3.check_sync).
set -uo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
source "$AAU_DIR/data_scale/recipe_v1.sh"
TD="$AAU_RUNS/evidence/trainer_v3/testdata"
OUT="$AAU_RUNS/evidence/trainer_v3/factorized/ddp_smoke_${SLURM_JOB_ID:-manual}"
STAGE="/tmp/${SLURM_JOB_ID:-manual}_fddp"
mkdir -p "$OUT" "$STAGE"
trap 'rm -rf "$STAGE"' EXIT
OMP_NUM_THREADS=4 "$AAU_DIR/run.sh" -m torch.distributed.run --standalone --nproc-per-node 2 --max-restarts 1 \
  v3_work/trainer/train_v3.py --total-steps 40 --steps-per-epoch 4 --split-json "$TD/split.json" \
  --data-spec "$TD/spec.json" --stage-root "$STAGE/stage" --prepass-proc 14 --train-threads 4 --eval_domain bfm \
  --gt-keep-scale --cache-workers 8 --cache-max-gb 60 "${RECIPE_V1[@]}" --epochs 10 --eval_every 5 --save_every 5 \
  --seed 1234 --dist_npz "$TD/factorized/gt_shape.npz" --runs_root "$OUT/runs" --log-every 5 --test-crash-at-step 18 \
  --fast-data --compact-cache --ema-decay 0.999 --area robust --area-robust smooth --sampler balanced --domain-alpha 0.0 \
  --input-norm global --scale-table "$TD/scale_table.npz" --head factorized --size-table "$TD/factorized/size.npz" \
  --scale-aug 0.8,1.25 --gt-scale 1.0 > "$OUT/ddp.log" 2>&1
echo "[fact-ddp] rc=$?"
tr '\r' '\n' < "$OUT/ddp.log" | grep -v '%|' > "$OUT/ddp.log.tmp" && mv "$OUT/ddp.log.tmp" "$OUT/ddp.log"
grep -h -E "RIPRESA|crash di prova|epoca [0-9]+:|passi eseguiti|DDP|Error|error|Epoch" "$OUT/ddp.log" | head -60
