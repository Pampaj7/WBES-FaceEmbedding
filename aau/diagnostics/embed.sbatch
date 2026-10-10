#!/usr/bin/env bash
#SBATCH --job-name=wbes-diag-embed
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=aicentre-a100
#SBATCH --qos=unprivileged
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#
# Diagnostica D, D1 (PROTOCOL_D.md sez. 3.2): embedding [s, u] dei bracci sugli insiemi nuovi (rigenerati e FLAME),
# con eval_v3 + zs_embed e gli argomenti di v3_work/trainer/ablations/c3f/calib_heldout.sbatch. Dopo DIAG_STEP=gen
# (operatori in datasets/V3_OPS_CACHE/diag_d1/ops, tabella e GT in aau/runs/evidence/diagnostics/d1).
# Job breve sulle A100 di nv-ai-04 (unprivileged, autorizzato per job brevi): ripartibile, gli embedding gia'
# scritti si saltano (--requeue).
#   sbatch aau/diagnostics/embed.sbatch
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
OPS="$WBES_ROOT/datasets/V3_OPS_CACHE/diag_d1/ops"
D1="$AAU_RUNS/evidence/diagnostics/d1"
[[ -f "$D1/scale_table.npz" && -f "$D1/gt_all_sr.npz" ]] || { echo "ERRORE: prima DIAG_STEP=gen"; exit 1; }
echo "[embed] host=$(hostname) job=${SLURM_JOB_ID:-none} $(date +%T), $(find "$OPS" -name '*.npz' | wc -l) operatori"
for key in factorized_s1234 factorized_s2345 factorizedc3m_e123; do
  out="$D1/emb/$key/embeddings.npz"
  [[ -f "$out" ]] && { echo "[embed] $key: embedding gia' presenti"; continue; }
  ck=$("$AAU_DIR/run.sh" v3_work/trainer/tools/fact_calib.py ckpt "$key" 2>/dev/null | tail -1)
  echo "[embed] $(date +%T) $key: $ck"
  mkdir -p "$D1/emb/$key"
  WBES_V3_SCALE_TABLES="$D1/scale_table.npz" WBES_V3_FACTORIZED_OUT=full WBES_EVAL_SCENARIOS=clean \
    "$AAU_DIR/run.sh" v3_work/trainer/eval_v3.py -- aau/zs3dmm/zs_embed.py \
      --model_path "$ck" --checkpoint_selector best_by_clean --data_dir "$OPS" --dist_npz "$D1/gt_all_sr.npz" \
      --subject_split all --eval_fraction 0.2 --max_subjects 0 --max_meshes_per_subject_eval 10 --preload_workers 8 \
      --pair_mode cross_topology --seed 1234 --embeddings_out "$out" 2>&1 | tee "$D1/emb/$key/embed.log"
  [[ -f "$out" ]] || { echo "ERRORE: $key senza embedding"; exit 1; }
done
echo "[embed] $(date +%T) fatto"
