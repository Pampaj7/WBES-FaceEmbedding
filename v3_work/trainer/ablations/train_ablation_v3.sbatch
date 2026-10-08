#!/usr/bin/env bash
#SBATCH --job-name=wbes-v3-abl
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --time=05:30:00
#SBATCH --no-requeue
#
# Ablazione del piano massivo (PLAN_MASSIVE §13, ridotte dal critic l'8 ottobre) con il trainer v3, UN braccio
# per job:   WBES_V3_ARM=<braccio> aau/submit.sh v3_work/trainer/ablations/train_ablation_v3.sbatch
#   ctrl        configurazione attuale (= v2): centro e pooling medio PER VERTICE, scala maxabs
#   PRIORITA' 1, E7 (E3: il 74% degli errori di riconoscimento su HIFI3D cade sulle coppie con down8k, perche'
#   centro e pooling medio dipendono dalla densita'):
#   area        --area on: centro e pooling medio pesati per area (massa), scala sqrt(area)
#   arearobust  --area robust: come area con aree robuste al rumore (passa-basso sui primi 64 autovettori:
#               sulle mesh vere l'area della noisy passa da 2.25x a 0.96x l'original, tests/test_area.py)
#   Altri bracci, se servono:
#   bal         campionatore bilanciato, alpha 0 = uniforme fra domini
#   ugt         target = GT unificata (datasets/UNIFIED_GT, invariante alla similarita'), batch monodominio
#   ugtmix      GT unificata + batch multi-dominio (bilanciato alpha 1 = domini in proporzione ai soggetti)
# La loss resta v2 in tutti (E3c: la struttura locale di e108 non e' peggiore delle baseline).
# Tutto il resto e' identico fra i bracci: dati, passi, seme (quindi, per area/arearobust, gli stessi
# batch, lo stesso rumore e gli stessi pesi iniziali dell'encoder: confronto appaiato), EMA 0.999, eval online.
#
# Dati: sottoinsieme del run su scala (aau/runs/evidence/trainer_v3/ablations/data, make_testdata.py seme
# 2026) con le sue proporzioni: 392 BFM (tutti), 4.390 ICT (326 dalla vista ICT-5000 + 4.064 dagli shard
# ICT_SCALE), 810 GNM = 5.592 identita'; eval online come il run su scala (16 BFM + 16 GNM). 8 blocchi
# stratificati, BFM residente in tutti; cache compatta senza perdita (int32, niente L).
# Memoria: vedi ablations/budget.json e budget_K16.json (tools/budget_v3.py).
# Sulle A100 (eccezione autorizzata, prelazionabili): aggiungere -p aicentre-a100 --qos=unprivileged
# --gres=gpu:a100:1 --requeue (launch_ablations.sh con WBES_V3_A100=1).
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv
aau_require_diffusion_net
source "$AAU_DIR/data_scale/recipe_v1.sh"

ARM="${WBES_V3_ARM:?WBES_V3_ARM=ctrl|area|arearobust|bal|ugt|ugtmix}"
GT_NAME=gt
case "$ARM" in
  ctrl)     EXTRA=() ;;
  area)     EXTRA=(--area on) ;;
  arearobust) EXTRA=(--area robust --area-robust smooth) ;;
  ugt)      EXTRA=(--gt unified); GT_NAME=gt_unified ;;
  ugtmix)   EXTRA=(--gt unified --sampler balanced --domain-alpha 1.0 --batch-domains mixed --no-domain-blocked)
            GT_NAME=gt_unified ;;
  bal)      EXTRA=(--sampler balanced --domain-alpha 0.0) ;;
  *) echo "ERRORE: braccio $ARM" >&2; exit 2 ;;
esac
D="$AAU_RUNS/evidence/trainer_v3/ablations/data"
T="${WBES_V3_STEPS:?WBES_V3_STEPS: passi totali, uguali per tutti i bracci (vedi summary.md)}"
# 8 blocchi: cache 93 GiB (misurata, smoke 1061910), RSS dopo il caricamento 128 GiB + /tmp del pre-pass
# successivo ~50 GiB sulle L40S = ~178 GiB (--mem 220G, 3 bracci per nodo); 16 blocchi: ~124 GiB (sottomettere con
# --mem=150G, 4 per nodo), epoche piu' corte. Stima: tools/budget_v3.py -> ablations/budget*.json
K="${WBES_V3_BLOCKS:-8}"
case "$K" in
  8)  SPEC="$D/spec.json";     S_DEF=140 ;;   # <= 142: un'epoca senza ripetizioni nel blocco (ICT e GNM)
  16) SPEC="$D/spec_K16.json"; S_DEF=70 ;;    # <= 71
  *) echo "ERRORE: WBES_V3_BLOCKS=$K (8|16)" >&2; exit 2 ;;
esac
S="${WBES_V3_STEPS_PER_EPOCH:-$S_DEF}"
# Operatori gia' calcolati sul disco locale di nv-ai-04 (precompute_ops_local.sbatch): la spec li usa come vista
# e niente pre-pass per braccio. Sottomettere con --nodelist=nv-ai-04.
if [[ -n "${WBES_V3_LOCAL_OPS:-}" ]]; then
  [[ -d "$WBES_V3_LOCAL_OPS" ]] || { echo "ERRORE: $WBES_V3_LOCAL_OPS assente su $(hostname)" >&2; exit 2; }
  LSPEC="/tmp/spec_local_${SLURM_JOB_ID:-manual}.json"
  "$AAU_DIR/run.sh" -c "import json,sys; s=json.load(open('$SPEC')); s['views']=s['views']+['$WBES_V3_LOCAL_OPS']; \
s['tars']=[]; s.pop('tar_index',None); s['view_bytes_json']=None; json.dump(s,open('$LSPEC','w'))"
  SPEC="$LSPEC"
fi
# --fast-data: stessi tensori di v2 (loss e gradienti identici, units_cpu.json), passo x1.8-2.2 su L40S
# (bench_forward.json). Su A100 conviene WBES_V3_FORWARD=groups (x2.3-2.5 contro x1.5-2.0 del sequenziale).
FWD="${WBES_V3_FORWARD:-sequential}"
EVAL_EVERY="${WBES_V3_EVAL_EVERY:-24}"
EPOCHS=$(( (T + S - 1) / S ))
SAVE_EVERY=$(( (EPOCHS + 3) / 4 ))           # 25/50/75/100%
LR_DROP=$(( T * 775 / 1000 ))                # stesso punto relativo del run su scala (81.747/105.480)
SEED="${WBES_V3_SEED:-1234}"
RUNS_ROOT="$AAU_RUNS/evidence/trainer_v3/ablations/runs/${ARM}_K${K}_s${SEED}_${SLURM_JOB_ID:-manual}"
STAGE="/tmp/${SLURM_JOB_ID:-manual}"
mkdir -p "$RUNS_ROOT" "$STAGE"
trap 'rm -rf "$STAGE"' EXIT
echo "$ARM" > "$RUNS_ROOT/arm.txt"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
CMD=(v3_work/trainer/train_v3.py --total-steps "$T" --steps-per-epoch "$S"
  --split-json "$D/split.json" --data-spec "$SPEC" --frozen-heldout "$AAU_DIR/data_scale/heldout_frozen.json"
  --stage-root "$STAGE/stage" --prepass-proc 22 --prepass-grad vec --prepass-tolerate 50 --train-threads 8 --domain-blocked --eval_domain bfm
  --gt-keep-scale --lr-steps "$LR_DROP:5e-5" --cache-workers 16 --cache-max-gb 150 --compact-cache
  --forward "$FWD" --fast-data --ema-decay 0.999 "${RECIPE_V1[@]}"
  --epochs "$EPOCHS" --eval_every "$EVAL_EVERY" --save_every "$SAVE_EVERY" --seed "$SEED"
  --dist_npz "$D/$GT_NAME.npz" --runs_root "$RUNS_ROOT" --max-hours "${WBES_V3_MAX_HOURS:-5.1}" --log-every 500
  "${EXTRA[@]}")
printf '%q ' "${CMD[@]}" > "$RUNS_ROOT/launch.txt"; echo >> "$RUNS_ROOT/launch.txt"
echo "[abl] braccio=$ARM blocchi=$K T=$T S=$S epoche=$EPOCHS forward=$FWD host=$(hostname) runs_root=$RUNS_ROOT"
# Prelazione (A100 aicentre, --qos=unprivileged --requeue): il job riparte con lo stesso id, quindi stessa run dir,
# e train_v3 --resume auto riprende da checkpoints/last.pth (ultima epoca finita). Il log si accoda.
echo "[abl] avvio $(date +%F_%T) restart=${SLURM_RESTART_COUNT:-0}" >> "$RUNS_ROOT/train.log"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$AAU_DIR/run.sh" "${CMD[@]}" 2>&1 \
  | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v '%|' >> "$RUNS_ROOT/train.log"
grep -E "passi eseguiti|^Best|max-hours" "$RUNS_ROOT/train.log" || true
