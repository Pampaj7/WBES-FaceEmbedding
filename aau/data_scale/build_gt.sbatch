#!/usr/bin/env bash
#SBATCH --job-name=wbes-ds-gt
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=03:00:00
#
# GT congiunta BFM + ICT esteso (aau/data_scale/build_gt.py): 55.000 identita' ICT, ~1.5e9 coppie x
# 9409 vertici, su GPU. Uscita ~12 GB (float32 denso) in datasets/ICT_SCALE/.
#
#   aau/submit.sh data_scale/build_gt.sbatch --dependency=afterok:<array>
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv
echo "[gt] host=$(hostname) job=${SLURM_JOB_ID:-none}"
"$AAU_DIR/run.sh" aau/data_scale/build_gt.py --shards-dir "$WBES_ROOT/datasets/ICT_SCALE/shards" \
    --out "$WBES_ROOT/datasets/ICT_SCALE/gt_joint_bfm_ict.npz"
