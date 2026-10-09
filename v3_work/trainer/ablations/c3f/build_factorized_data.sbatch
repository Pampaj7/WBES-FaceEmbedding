#!/usr/bin/env bash
#SBATCH --job-name=wbes-v3-c3f-factdata
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#
# Dati del braccio C3F ``factorized`` (niente GPU, ripartibile: le uscite gia' scritte si saltano):
#   1. tabella di scala delle mesh dello store C3F (tools/build_scale_table.py: area in mm^2 nel frame canonico
#      dalla geometria grezza, verificata contro le viste) -> ablations/c3f/scale_table.npz;
#   2. tabelle delle viste di eval (HIFI3D eval_view, dev FaceScape eval_view ed expr_view, FaceVerse expr_view)
#      -> factorized/scale_tables/<vista>.npz (lette da eval_v3 via WBES_V3_SCALE_TABLES);
#   3. GT shape (kappa * dP, nomi e ordine di c3f/gt.npz) e centroid size dei 6.993 soggetti
#      (tools/build_factorized_targets.py, venv della GT unificata) -> ablations/c3f/factorized/.
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export AAU_NV=
C="$AAU_RUNS/evidence/trainer_v3/ablations/c3f"
ST="$AAU_RUNS/evidence/trainer_v3/factorized/scale_tables"
mkdir -p "$ST" "$C/factorized"
W="${SLURM_CPUS_PER_TASK:-16}"
[[ -f "$C/scale_table.npz" ]] || OMP_NUM_THREADS=1 "$AAU_DIR/run.sh" v3_work/trainer/tools/build_scale_table.py \
  --spec "$C/spec.json" --split "$C/split.json" --out "$C/scale_table.npz" --workers "$W"
for v in hifi3d_eval:HIFI3D/eval_view:hifi3d devfs_eval:DEV_FACESCAPE/eval_view:facescape \
         devfs_expr:DEV_FACESCAPE/expr_view:facescape fv_expr:FACEVERSE_ZS/expr_view:faceverse; do
  IFS=: read -r name dir dom <<< "$v"
  [[ -f "$ST/$name.npz" ]] || OMP_NUM_THREADS=1 "$AAU_DIR/run.sh" v3_work/trainer/tools/build_scale_table.py \
    --view-dir "datasets/$dir/npz" --domain "$dom" --out "$ST/$name.npz" --workers "$W"
done
[[ -f "$C/factorized/gt_shape.npz" ]] || UGT_THREADS="$W" v3_work/unified_gt/run.sh \
  v3_work/trainer/tools/build_factorized_targets.py --ref-gt "$C/gt.npz" --split "$C/split.json" --out-dir "$C/factorized"
echo "[factdata] OK"
