#!/usr/bin/env bash
#SBATCH --job-name=wbes-abl2x2-flemb
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --partition=aicentre-a100
#SBATCH --qos=unprivileged
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#
# Ablazione 2x2, emendamento 1 (sez. 4): SOLO l'embedding degli insiemi FLAME 2023 di D1 rigenerati per una cella che
# l'ha avuto in training (B, D; poi B'), per il controllo "B e D hanno imparato FLAME". Il passo 3 di
# ablation_2x2_eval.sbatch, copiato: zs_embed come aau/diagnostics/embed.sbatch, sull'ultimo checkpoint EMA della
# cella; nessuna calibrazione, nessun embedding dei test. Dopo la valutazione della cella (stesso checkpoint).
#   CELL=B sbatch --dependency=afterok:<valutazione di B> v3_work/stream/slurm/ablation_2x2_flame_embed.sbatch
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
CELL="${CELL:?CELL=B|D|...}"
ABL="$AAU_RUNS/evidence/stream/ablation_2x2"
RUNS="${ABL_RUNS:-$ABL/$CELL/runs}"
EP="${ABL_EPOCH:-100}"
OUT="${ABL_OUT:-$ABL/$CELL/eval}"
shopt -s nullglob
CKS=("$RUNS"/v3_*/checkpoints/epoch${EP}_ema.pth)
shopt -u nullglob
(( ${#CKS[@]} == 1 )) || { echo "[abl-flemb] ERRORE: ${#CKS[@]} checkpoint epoch${EP}_ema.pth in $RUNS" >&2; exit 1; }
CK="${CKS[0]}"
D1F="$ABL/d1flame"
OPS="$WBES_ROOT/datasets/V3_OPS_CACHE/abl2x2_d1"
[[ -f "$OPS/.done" && -f "$D1F/check.json" ]] || { echo "[abl-flemb] ERRORE: prima ablation_2x2_flame.sbatch" >&2; exit 1; }
echo "[abl-flemb] $(date +%T) cella $CELL host=$(hostname) job=${SLURM_JOB_ID:-none} checkpoint $CK -> $OUT/flame" \
  | tee -a "$OUT/eval.log"
if [[ ! -f "$OUT/flame/embeddings.npz" ]]; then
  mkdir -p "$OUT/flame"
  WBES_V3_SCALE_TABLES="$D1F/scale_table.npz" WBES_V3_FACTORIZED_OUT=full WBES_EVAL_SCENARIOS=clean \
    "$AAU_DIR/run.sh" v3_work/trainer/eval_v3.py -- aau/zs3dmm/zs_embed.py \
      --model_path "$CK" --checkpoint_selector best_by_clean --data_dir "$OPS/ops" --dist_npz "$D1F/gt_all_sr.npz" \
      --subject_split all --eval_fraction 0.2 --max_subjects 0 --max_meshes_per_subject_eval 10 --preload_workers 8 \
      --pair_mode cross_topology --seed 1234 --embeddings_out "$OUT/flame/.embeddings.tmp.npz" 2>&1 \
    | tee "$OUT/flame/embed.log"
  [[ -f "$OUT/flame/.embeddings.tmp.npz" ]] || { echo "[abl-flemb] ERRORE: embedding FLAME assenti" >&2; exit 1; }
  mv "$OUT/flame/.embeddings.tmp.npz" "$OUT/flame/embeddings.npz"
fi
echo "[abl-flemb] $(date +%T) cella $CELL fatta" | tee -a "$OUT/eval.log"
