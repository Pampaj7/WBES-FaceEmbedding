#!/usr/bin/env bash
#SBATCH --job-name=wbes-fact-calib
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#
# Calibrazione di d_P sugli held-out sintetici (factorized_protocol emendamento 4, tools/fact_calib.py):
#   1. stage: mesh grezze dei 300 soggetti held-out su /tmp, tabella di scala in $OPS (una volta);
#   2. operatori areanorm k_eig 128 in $OPS/ops (cache in datasets/V3_OPS_CACHE, ripartibile: i file esistenti si saltano);
#   3. embedding [s, u] (dual: [z_F, u]) di ogni checkpoint con eval_v3 + zs_embed, come i domini di test;
#   4. fact_calib.py calib -> aau/runs/evidence/trainer_v3/factorized_calibration.csv.
#   WBES_CALIB_KEYS="factorized_s1234 ..." aau/submit.sh v3_work/trainer/ablations/c3f/calib_heldout.sbatch
# Chiavi assenti (checkpoint non ancora scritto, es. dual in training) si saltano; embedding gia' scritti pure.
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
T=v3_work/trainer/tools/fact_calib.py
OPS="$WBES_ROOT/datasets/V3_OPS_CACHE/heldout_calib"
EMB="$AAU_RUNS/evidence/trainer_v3/factorized/calib_heldout"
STAGE="/tmp/wbes_calib_${SLURM_JOB_ID:-manual}"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$OPS/ops" "$EMB"
exec 9> "$OPS.lock"; flock 9
if [[ ! -f "$OPS/.staged" ]]; then
  echo "[calib] $(date +%T) stage"
  "$AAU_DIR/run.sh" "$T" stage --in-dir "$STAGE/in" --table "$OPS/scale_table.npz"
  N=$(find "$STAGE/in" -name "*.npz" | wc -l)
  pids=()
  for (( i = 0; i < 31; i++ )); do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$AAU_DIR/run.sh" v2_work/potential/areanorm_operators.py \
      --input-dir "$STAGE/in" --output-dir "$OPS/ops" --k-eig 128 --shard "$i/31" > "$STAGE.shard$i.log" 2>&1 & pids+=($!)
  done
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "ERRORE: shard $i"; cat "$STAGE.shard$i.log"; exit 1; }; done
  rm -f "$STAGE".shard*.log
  M=$(find "$OPS/ops" -name "*.npz" ! -name ".*" | wc -l)
  echo "[calib] $(date +%T) operatori $M / $N, $(du -sh "$OPS/ops" | cut -f1)"
  (( M == N )) || { echo "ERRORE: operatori incompleti"; exit 1; }
  touch "$OPS/.staged"
fi
flock -u 9
GT="$AAU_RUNS/evidence/trainer_v3/ablations/c3f/gt_sr.npz"
for key in ${WBES_CALIB_KEYS:-$("$AAU_DIR/run.sh" "$T" keys)}; do
  out="$EMB/$key/embeddings.npz"
  [[ -f "$out" ]] && { echo "[calib] $key: embedding gia' presenti"; continue; }
  ck=$("$AAU_DIR/run.sh" "$T" ckpt "$key" 2>/dev/null) || { echo "[calib] $key: checkpoint assente, saltato"; continue; }
  echo "[calib] $(date +%T) $key: $ck"
  mkdir -p "$EMB/$key"
  WBES_V3_SCALE_TABLES="$OPS/scale_table.npz" WBES_V3_FACTORIZED_OUT=full WBES_EVAL_SCENARIOS=clean \
    "$AAU_DIR/run.sh" v3_work/trainer/eval_v3.py -- aau/zs3dmm/zs_embed.py \
      --model_path "$ck" --checkpoint_selector best_by_clean --data_dir "$OPS/ops" --dist_npz "$GT" \
      --subject_split all --eval_fraction 0.2 --max_subjects 0 --max_meshes_per_subject_eval 10 --preload_workers 8 \
      --pair_mode cross_topology --seed 1234 --embeddings_out "$out" 2>&1 | tee "$EMB/$key/embed.log"
  [[ -f "$out" ]] || { echo "ERRORE: $key senza embedding"; exit 1; }
done
"$AAU_DIR/run.sh" "$T" calib
