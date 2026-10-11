#!/usr/bin/env bash
#SBATCH --job-name=wbes-abl2x2-eval
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --partition=aicentre-a100
#SBATCH --qos=unprivileged
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#
# Valutazione di una cella dell'ablazione 2x2 (aau/runs/evidence/stream/ablation_2x2/PROTOCOL.md, sez. 4) sull'ultimo
# checkpoint EMA (epoch100_ema.pth), dopo il suo training. Job breve e ripartibile su una A100 di nv-ai-04
# (unprivileged, --requeue):
#   1. c sugli held-out sintetici: v3_work/trainer/slurm/calib_ckpt.sbatch (fact_calib.py calib-ckpt) -> eval/calib;
#   2. embedding [s, u] dei test: passo ``form`` di v3_work/trainer/ablations/c3f/eval_body.sh, copiato all'avvio con la
#      sola directory di uscita cambiata (eval/form_hifi, form_devfs, form_devfs_expr, form_fv_expr, form/): HIFI3D, dev
#      FaceScape neutra (e con espressioni, non letta), FaceVerse con espressioni; nessun passo famos ne' now;
#   3. solo A e C: embedding degli insiemi FLAME di D1 rigenerati (ablation_2x2_flame.sbatch) -> eval/flame.
#   CELL=A sbatch --dependency=afterok:<training A>:<flame> v3_work/stream/slurm/ablation_2x2_eval.sbatch
# Prova su un altro checkpoint: CELL=T ABL_RUNS=<dir con v3_*/checkpoints> ABL_EPOCH=<NNN> ABL_OUT=<dir> ABL_FLAME=1.
# Ripartibile: calibrazione e FLAME saltati se gia' scritti, i bracci zs con .done saltati (eval_body.sh).
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
CELL="${CELL:?CELL=A|B|C|D}"
ABL="$AAU_RUNS/evidence/stream/ablation_2x2"
RUNS="${ABL_RUNS:-$ABL/$CELL/runs}"
EP="${ABL_EPOCH:-100}"
OUT="${ABL_OUT:-$ABL/$CELL/eval}"
FL="${ABL_FLAME:-$([[ "$CELL" == A || "$CELL" == C ]] && echo 1 || echo 0)}"
shopt -s nullglob
CKS=("$RUNS"/v3_*/checkpoints/epoch${EP}_ema.pth)
shopt -u nullglob
(( ${#CKS[@]} == 1 )) || { echo "[abl-eval] ERRORE: ${#CKS[@]} checkpoint epoch${EP}_ema.pth in $RUNS" >&2; exit 1; }
CK="${CKS[0]}"
mkdir -p "$OUT"
echo "[abl-eval] $(date +%T) cella $CELL host=$(hostname) job=${SLURM_JOB_ID:-none} riavvio=${SLURM_RESTART_COUNT:-0}" \
     "checkpoint $CK -> $OUT (FLAME: $FL)" | tee -a "$OUT/eval.log"

# 1. calibrazione c (held-out bfm REMESH, ict, gnm)
if [[ ! -f "$OUT/calib/calib.json" ]]; then
  WBES_CALIB_CKPT="$CK" WBES_CALIB_OUT="$OUT/calib" WBES_CALIB_ARM=factorized bash v3_work/trainer/slurm/calib_ckpt.sbatch
fi
echo "[abl-eval] $(date +%T) c: $(python3 -c "import json, sys; d = json.load(open(sys.argv[1])); print(d['c_median'], d['c_ls'])" "$OUT/calib/calib.json")" \
  | tee -a "$OUT/eval.log"

# 2. embedding dei test: passo form di eval_body.sh con OUT = $OUT
B="/tmp/abl2x2_eval_body_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}.sh"
sed 's|^OUT="\$AAU_RUNS/evidence/trainer_v3/ablations/c3f_eval"$|OUT="$ABL_EVAL_OUT"|' \
  "$WBES_ROOT/v3_work/trainer/ablations/c3f/eval_body.sh" > "$B"
trap 'rm -f "$B"' EXIT
[[ $(grep -cx 'OUT="$ABL_EVAL_OUT"' "$B") == 1 ]] || { echo "[abl-eval] ERRORE: OUT di eval_body.sh non sostituito" >&2; exit 1; }
ABL_EVAL_OUT="$OUT" WBES_V3_ARM="factorizedabl2x2${CELL,,}" WBES_V3_EVAL_RR="$RUNS" WBES_V3_EVAL_EPOCHS="$EP" \
  WBES_V3_EVAL_STEPS=form bash "$B"
echo "[abl-eval] $(date +%T) embedding dei test fatti" | tee -a "$OUT/eval.log"

# 3. FLAME 2023 di D1 (solo A e C): zs_embed come aau/diagnostics/embed.sbatch
if [[ "$FL" == 1 && ! -f "$OUT/flame/embeddings.npz" ]]; then
  D1F="$ABL/d1flame"
  OPS="$WBES_ROOT/datasets/V3_OPS_CACHE/abl2x2_d1"
  [[ -f "$OPS/.done" && -f "$D1F/check.json" ]] || { echo "[abl-eval] ERRORE: prima ablation_2x2_flame.sbatch" >&2; exit 1; }
  mkdir -p "$OUT/flame"
  WBES_V3_SCALE_TABLES="$D1F/scale_table.npz" WBES_V3_FACTORIZED_OUT=full WBES_EVAL_SCENARIOS=clean \
    "$AAU_DIR/run.sh" v3_work/trainer/eval_v3.py -- aau/zs3dmm/zs_embed.py \
      --model_path "$CK" --checkpoint_selector best_by_clean --data_dir "$OPS/ops" --dist_npz "$D1F/gt_all_sr.npz" \
      --subject_split all --eval_fraction 0.2 --max_subjects 0 --max_meshes_per_subject_eval 10 --preload_workers 8 \
      --pair_mode cross_topology --seed 1234 --embeddings_out "$OUT/flame/.embeddings.tmp.npz" 2>&1 \
    | tee "$OUT/flame/embed.log"
  [[ -f "$OUT/flame/.embeddings.tmp.npz" ]] || { echo "[abl-eval] ERRORE: embedding FLAME assenti" >&2; exit 1; }
  mv "$OUT/flame/.embeddings.tmp.npz" "$OUT/flame/embeddings.npz"
fi
echo "[abl-eval] $(date +%T) cella $CELL fatta" | tee -a "$OUT/eval.log"
