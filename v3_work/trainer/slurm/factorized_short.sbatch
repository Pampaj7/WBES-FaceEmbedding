#!/usr/bin/env bash
#SBATCH --job-name=wbes-v3-fact-short
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# Modalita' fattorizzata, run breve di stabilita' sui dati di prova (testdata/: 15 BFM, 30 ICT-5000, 30 ICT nuovi,
# 30 GNM, 2 blocchi; pre-pass su /tmp): la combinazione del braccio C3F (--area robust, bal, ingresso globale,
# testa fattorizzata, augmentation di scala 0.8-1.25, GT shape e centroid size di build_factorized_targets.py),
# WBES_V3_SHORT_T passi (2000) da 100, eval online ogni 5 epoche, EMA 0.999. Poi tests/check_equivariance.py sui
# pesi EMA finali (BFM dell'eval online e ICT-5000 held-out, mai visti in training).
set -uo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
source "$AAU_DIR/data_scale/recipe_v1.sh"
TD="$AAU_RUNS/evidence/trainer_v3/testdata"
T="${WBES_V3_SHORT_T:-2000}"
OUT="$AAU_RUNS/evidence/trainer_v3/factorized/short_${SLURM_JOB_ID:-manual}"
STAGE="/tmp/${SLURM_JOB_ID:-manual}_fact"
mkdir -p "$OUT" "$STAGE"
trap 'rm -rf "$STAGE"' EXIT
nvidia-smi --query-gpu=name --format=csv,noheader || true
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$AAU_DIR/run.sh" v3_work/trainer/train_v3.py \
  --total-steps "$T" --steps-per-epoch 100 --split-json "$TD/split.json" --data-spec "$TD/spec.json" \
  --stage-root "$STAGE/stage" --prepass-proc 14 --train-threads 8 --eval_domain bfm --gt-keep-scale \
  --cache-workers 8 --cache-max-gb 40 "${RECIPE_V1[@]}" --epochs $(( T / 100 )) --eval_every 5 --save_every 10 \
  --seed 1234 --dist_npz "$TD/factorized/gt_shape.npz" --runs_root "$OUT/runs" --forward sequential --fast-data \
  --compact-cache --ema-decay 0.999 --log-every 100 --ckpt-minutes 0 \
  --area robust --area-robust smooth --sampler balanced --domain-alpha 0.0 \
  --input-norm global --scale-table "$TD/scale_table.npz" --head factorized \
  --size-table "$TD/factorized/size.npz" --scale-aug 0.8,1.25 ${WBES_V3_SHORT_EXTRA:-} \
  > "$OUT/train.log" 2>&1
echo "[fact-short] training rc=$?"
tr '\r' '\n' < "$OUT/train.log" | grep -v '%|' > "$OUT/train.log.tmp" && mv "$OUT/train.log.tmp" "$OUT/train.log"
CK=$(ls "$OUT"/runs/v3_*/checkpoints/epoch*_ema.pth 2>/dev/null | sort | tail -1)
[[ -n "$CK" ]] || { echo "ERRORE: nessun checkpoint EMA"; exit 1; }
"$AAU_DIR/run.sh" v3_work/trainer/tests/check_equivariance.py --ckpt "$CK" --out "$OUT/equivariance.json" \
  > "$OUT/equivariance.log" 2>&1
echo "[fact-short] equivarianza rc=$? ($CK)"
