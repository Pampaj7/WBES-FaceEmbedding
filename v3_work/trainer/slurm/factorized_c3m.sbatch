#!/usr/bin/env bash
#SBATCH --job-name=wbes-v3-fact-c3m
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --partition=prioritized
#SBATCH --nodelist=a768-l40s-06
#SBATCH --gres=gpu:l40s:6
#SBATCH --cpus-per-task=180
#SBATCH --mem=700G
#SBATCH --time=2-12:00:00
#SBATCH --requeue
#
# factorized a scala piena (protocollo aau/runs/evidence/trainer_v3/factorized_protocol.md, run C3M, descrittivo):
# split del run su scala (64.400 identita' di training), spec del run su scala (viste BFM e ICT-5000, 241 tar, K 46,
# BFM residente all'8.87%), T 60.000 passi da 293 per rank, DDP su 6 L40S (torchrun elastico), EMA 0.999,
# checkpoint di ripresa ogni 20 minuti nella home (--resume auto: un requeue riparte da last.pth).
# Prima del training ASPETTA: datasets/CANONICAL_GT/train/READY (GT di E12 verificate) e la tabella di scala C3M.
# kappa = media geometrica dei fattori di taratura per dominio di SR (gt_sr_bfm_ict_gnm_calib.json di E12).
set -uo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
source "$AAU_DIR/data_scale/recipe_v1.sh"
CGT="$WBES_ROOT/datasets/CANONICAL_GT/train"
ST="$AAU_RUNS/evidence/trainer_v3/factorized/scale_tables/c3m.npz"
OUT="$AAU_RUNS/evidence/trainer_v3/factorized/c3m"
SPEC_SRC="$AAU_RUNS/data_scale_runs/scale_bfm_ict_gnm_s1234_nocanon_noaug_20261007_1411/spec.json"
T="${WBES_V3_C3M_T:-60000}"
W="${WBES_V3_C3M_WORLD:-6}"
mkdir -p "$OUT"
echo "[c3m] $(date +%F_%T) host=$(hostname) job=${SLURM_JOB_ID:-none} restart=${SLURM_RESTART_COUNT:-0}: attendo GT e tabella" \
  | tee -a "$OUT/train.log"
until [[ -f "$CGT/READY" && -f "$ST" && -f "$CGT/gt_sr_bfm_ict_gnm.npz" && -f "$CGT/centroid_size_bfm_ict_gnm.npz" ]]; do
  sleep 60
done
KAPPA=$(python3 -c "
import json, math
f = json.load(open('$CGT/gt_sr_bfm_ict_gnm_calib.json'))['factor']
print(repr(math.exp(sum(math.log(f[d]) for d in ('bfm', 'ict', 'gnm')) / 3)))")
echo "[c3m] $(date +%F_%T) GT pronte (READY: $(head -c 200 "$CGT/READY")), kappa=$KAPPA" | tee -a "$OUT/train.log"
cp -n "$SPEC_SRC" "$OUT/spec.json"
STAGE="/tmp/${SLURM_JOB_ID:-manual}_r${SLURM_RESTART_COUNT:-0}"
rm -rf "/tmp/${SLURM_JOB_ID:-manual}"_r*
mkdir -p "$STAGE"
trap 'rm -rf "$STAGE"' EXIT
CMD=(v3_work/trainer/train_v3.py --total-steps "$T" --steps-per-epoch 293
  --split-json "$AAU_DIR/data_scale/split_scale_all.json" --frozen-heldout "$AAU_DIR/data_scale/heldout_frozen.json"
  --data-spec "$OUT/spec.json" --stage-root "$STAGE/stage" --prepass-proc 150 --train-threads 4
  --domain-blocked --eval_domain bfm --gt-keep-scale --lr-steps 81747:5e-5 --cache-workers 8 --cache-max-gb 240
  "${RECIPE_V1[@]}" --epochs $(( (T + 292) / 293 )) --eval_every 10 --save_every 41 --seed 1234
  --dist_npz "$CGT/gt_sr_bfm_ict_gnm.npz" --gt-scale "$KAPPA" --runs_root "$OUT/runs"
  --forward sequential --fast-data --compact-cache --ema-decay 0.999 --ckpt-minutes 20 --log-every 500
  --area robust --area-robust smooth --sampler balanced --domain-alpha 0.0
  --input-norm global --scale-table "$ST" --head factorized --size-table "$CGT/centroid_size_bfm_ict_gnm.npz"
  --scale-aug 0.8,1.25 --size-mask-domains bfm --dropout 0.0)
printf '%q ' "${CMD[@]}" > "$OUT/launch.txt"; echo >> "$OUT/launch.txt"
nvidia-smi --query-gpu=name --format=csv,noheader | head -2 || true
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$AAU_DIR/run.sh" -m torch.distributed.run --standalone --nproc-per-node "$W" \
  --max-restarts 2 "${CMD[@]}" 2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v '%|' >> "$OUT/train.log"
echo "[c3m] $(date +%F_%T) fine rc=${PIPESTATUS[0]}" | tee -a "$OUT/train.log"
