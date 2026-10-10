#!/usr/bin/env bash
#SBATCH --job-name=wbes-v3-dual-short
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=aicentre-a100
#SBATCH --qos=unprivileged
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#
# Testa doppia (--head dual), run breve di stabilita' sui dati di prova: z_F su GT-FR tarata (come ctrlfr), u su GT-SR
# grezza x kappa (come factorized), pesi --lambda-form/--lambda-shape (WBES_V3_DUAL_LF/LS, default 1/1), ricetta dei
# bracci di factorized_protocol.md (arearobust, bal, ingresso globale, dropout 0), WBES_V3_SHORT_T passi (1000).
set -uo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
source "$AAU_DIR/data_scale/recipe_v1.sh"
TD="$AAU_RUNS/evidence/trainer_v3/testdata"
C="$AAU_RUNS/evidence/trainer_v3/ablations/c3f"
KAPPA=$(python3 -c "import json; print(repr(json.load(open('$C/fr_params.json'))['kappa']))")
T="${WBES_V3_SHORT_T:-1000}"
OUT="$AAU_RUNS/evidence/trainer_v3/factorized/dual_short_${SLURM_JOB_ID:-manual}"
STAGE="/tmp/${SLURM_JOB_ID:-manual}_dual"
mkdir -p "$OUT" "$STAGE"
trap 'rm -rf "$STAGE"' EXIT
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$AAU_DIR/run.sh" v3_work/trainer/train_v3.py \
  --total-steps "$T" --steps-per-epoch 100 --split-json "$TD/split.json" --data-spec "$TD/spec.json" \
  --stage-root "$STAGE/stage" --prepass-proc 14 --train-threads 8 --eval_domain bfm --gt-keep-scale \
  --cache-workers 8 --cache-max-gb 40 "${RECIPE_V1[@]}" --epochs $(( T / 100 )) --eval_every 5 --save_every 10 \
  --seed 1234 --dist_npz "$TD/gt_frcal.npz" --runs_root "$OUT/runs" --forward sequential --fast-data \
  --compact-cache --ema-decay 0.999 --log-every 100 --ckpt-minutes 0 --dropout 0.0 \
  --area robust --area-robust smooth --sampler balanced --domain-alpha 0.0 \
  --input-norm global --scale-table "$TD/scale_table.npz" --head dual --dist-npz-shape "$TD/gt_sr.npz" \
  --gt-scale-shape "$KAPPA" --lambda-form "${WBES_V3_DUAL_LF:-1.0}" --lambda-shape "${WBES_V3_DUAL_LS:-1.0}" \
  > "$OUT/train.log" 2>&1
echo "[dual-short] rc=$?"
tr '\r' '\n' < "$OUT/train.log" | grep -v '%|' > "$OUT/train.log.tmp" && mv "$OUT/train.log.tmp" "$OUT/train.log"
grep -h "^Epoch\|Traceback\|Error" "$OUT/train.log" | cut -c1-260
