#!/usr/bin/env bash
# Un braccio delle ablazioni v3 sulla cella E1 C3F (protocollo: aau/runs/evidence/trainer_v3/ablation_protocol.md).
# Stessi dati, blocchi, ordine, passi e ricetta di e1_train_body.sh (cella c3f): split_c3f.json, spec del run su
# scala con 4 blocchi, T 21.096, S 293, eval ogni 6 epoche, checkpoint alle epoche 36 e 72, seme 1234. Operatori
# dallo store condiviso (build_store.sbatch). Forward sequenziale con --fast-data (= v2), EMA 0.999.
# Ripresa: run dir fissa per braccio, --resume auto da checkpoints/last.pth (checkpoint ogni 20 minuti).
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
source "$AAU_DIR/data_scale/recipe_v1.sh"
ARM="${WBES_V3_ARM:?WBES_V3_ARM=ctrl|area|arearobust|bal|ugtmix}"
C="$AAU_RUNS/evidence/trainer_v3/ablations/c3f"
STORE="${WBES_V3_STORE:-/tmp/wbes_v3_store_c3f}"
GT="$C/gt.npz"
case "$ARM" in
  ctrl)       EXTRA=() ;;
  area)       EXTRA=(--area on) ;;
  arearobust) EXTRA=(--area robust --area-robust smooth) ;;
  bal)        EXTRA=(--sampler balanced --domain-alpha 0.0) ;;
  ugtmix)     EXTRA=(--gt unified --sampler balanced --domain-alpha 1.0 --batch-domains mixed --no-domain-blocked)
              GT="$C/gt_unified.npz" ;;
  loginv)     EXTRA=(--loss log+inv) ;;   # emendamento del 9 ottobre (ablation_protocol_emendamento_2026-10-09.md)
  *) echo "ERRORE: braccio $ARM" >&2; exit 2 ;;
esac
[[ -f "$STORE/index.npz" && -f "$STORE/manifest.json" ]] || { echo "ERRORE: store $STORE incompleto su $(hostname)" >&2; exit 3; }
RUNS_ROOT="$AAU_RUNS/evidence/trainer_v3/ablations/c3f_runs/$ARM"
mkdir -p "$RUNS_ROOT"
CMD=(v3_work/trainer/train_v3.py --total-steps 21096 --steps-per-epoch 293
  --split-json "$C/split.json" --data-spec "$C/spec.json" --store "$STORE"
  --frozen-heldout "$AAU_DIR/data_scale/heldout_frozen.json" --train-threads 8
  --domain-blocked --eval_domain bfm --gt-keep-scale --lr-steps 81747:5e-5 --cache-workers 8
  "${RECIPE_V1[@]}" --epochs 72 --eval_every 6 --save_every 36 --seed 1234
  --dist_npz "$GT" --runs_root "$RUNS_ROOT" --forward sequential --fast-data --ema-decay 0.999
  --ckpt-minutes 20 --log-every 500 "${EXTRA[@]}")
printf '%q ' "${CMD[@]}" > "$RUNS_ROOT/launch.txt"; echo >> "$RUNS_ROOT/launch.txt"
echo "[c3f] $(date +%F_%T) braccio=$ARM host=$(hostname) job=${SLURM_JOB_ID:-none} restart=${SLURM_RESTART_COUNT:-0}" \
  | tee -a "$RUNS_ROOT/train.log"
nvidia-smi --query-gpu=name --format=csv,noheader || true
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$AAU_DIR/run.sh" "${CMD[@]}" 2>&1 \
  | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v '%|' >> "$RUNS_ROOT/train.log"
grep -E "passi eseguiti|^Best" "$RUNS_ROOT/train.log" | tail -4 || true
[[ "$0" == /tmp/v3c3f_train_body_* ]] && rm -f "$0"
