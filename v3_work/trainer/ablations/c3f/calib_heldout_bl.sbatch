#!/usr/bin/env bash
#SBATCH --job-name=wbes-fact-calib-bl
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#
# k delle composizioni sugli held-out sintetici (factorized_protocol emendamento 5, tools/fact_calib.py), solo CPU:
#   1. stage: le stesse mesh grezze di calib_heldout.sbatch, su /tmp (proprio, niente lock: non tocca $OPS);
#   2. bl: ICP + Chamfer (tutte le coppie) e NICP per coppia (6000 coppie) in modo cs -> factorized/calib_heldout_bl;
#   3. calib-bl -> aau/runs/evidence/trainer_v3/factorized_calibration_bl.csv.
#   aau/submit.sh v3_work/trainer/ablations/c3f/calib_heldout_bl.sbatch
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export AAU_NV=
T=v3_work/trainer/tools/fact_calib.py
STAGE="/tmp/wbes_calib_bl_${SLURM_JOB_ID:-manual}"
trap 'rm -rf "$STAGE"' EXIT
echo "[calib-bl] $(date +%T) stage"
"$AAU_DIR/run.sh" "$T" stage --in-dir "$STAGE/in" --table "$STAGE/scale_table.npz"
echo "[calib-bl] $(date +%T) bl"
"$AAU_DIR/outlineB/run_o3d.sh" "$T" bl --in-dir "$STAGE/in" --table "$STAGE/scale_table.npz" \
  --workers "${SLURM_CPUS_PER_TASK:-32}" --steps "${WBES_BL_STEPS:-fast,nicp}"
echo "[calib-bl] $(date +%T) calib-bl"
"$AAU_DIR/outlineB/run_o3d.sh" "$T" calib-bl
