#!/usr/bin/env bash
#SBATCH --job-name=wbes-calib-ckpt
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#
# c di d_P su un checkpoint QUALUNQUE (es. un epochNNN_ema.pth del run massivo sullo stream), sugli held-out
# sintetici di calib_heldout.sbatch (bfm 100 REMESH, ict 100, gnm 100; operatori gia' in datasets/V3_OPS_CACHE/
# heldout_calib, nessuno stage qui), nessun dominio di test:
#   1. embedding [s, u] con eval_v3 + zs_embed, come calib_heldout.sbatch;
#   2. tools/fact_calib.py calib-ckpt -> $WBES_CALIB_OUT/calib.json (c, c_LS, c per dominio, sha256).
#   WBES_CALIB_CKPT=<.../epochNNN_ema.pth> WBES_CALIB_OUT=<dir> [WBES_CALIB_ARM=factorized] \
#     sbatch v3_work/trainer/slurm/calib_ckpt.sbatch
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
CK="${WBES_CALIB_CKPT:?checkpoint (WBES_CALIB_CKPT)}"
OUT="${WBES_CALIB_OUT:?directory di uscita (WBES_CALIB_OUT)}"
OPS="$WBES_ROOT/datasets/V3_OPS_CACHE/heldout_calib"
GT="$AAU_RUNS/evidence/trainer_v3/ablations/c3f/gt_sr.npz"
[[ -f "$OPS/.staged" ]] || { echo "ERRORE: held-out non preparati ($OPS): prima calib_heldout.sbatch"; exit 1; }
mkdir -p "$OUT"
EMB="$OUT/embeddings.npz"
if [[ ! -f "$EMB" ]]; then
  echo "[calib] $(date +%T) embedding di $CK"
  WBES_V3_SCALE_TABLES="$OPS/scale_table.npz" WBES_V3_FACTORIZED_OUT=full WBES_EVAL_SCENARIOS=clean \
    "$AAU_DIR/run.sh" v3_work/trainer/eval_v3.py -- aau/zs3dmm/zs_embed.py \
      --model_path "$CK" --checkpoint_selector best_by_clean --data_dir "$OPS/ops" --dist_npz "$GT" \
      --subject_split all --eval_fraction 0.2 --max_subjects 0 --max_meshes_per_subject_eval 10 --preload_workers 8 \
      --pair_mode cross_topology --seed 1234 --embeddings_out "$EMB" 2>&1 | tee "$OUT/embed.log"
  [[ -f "$EMB" ]] || { echo "ERRORE: embedding assenti"; exit 1; }
fi
"$AAU_DIR/run.sh" v3_work/trainer/tools/fact_calib.py calib-ckpt --ckpt "$CK" --arm "${WBES_CALIB_ARM:-factorized}" \
  --emb "$EMB" --out "$OUT/calib.json"
