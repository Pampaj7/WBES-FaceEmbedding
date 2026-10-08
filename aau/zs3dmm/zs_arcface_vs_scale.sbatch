#!/usr/bin/env bash
#SBATCH --job-name=wbes-arcface-vs-scale
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#
# ArcFace su render contro i checkpoint del run grande su HIFI3D, nei due protocolli
# (zs_arcface_vs_scale.py) -> aau/runs/data_scale_ood/arcface_vs_scale_hifi3d.md. Vuole gli embedding
# dei checkpoint:
#
#   WBES_ZS_DOMAIN=hifi WBES_ZS_ARMS="scale_e036 scale_e072 scale_e108" WBES_ZS_PART=embed \
#     WBES_ZS_CKPT_SCALE_E036=... aau/submit.sh zs3dmm/zs_zeroshot.sbatch --time=00:40:00
#   aau/submit.sh zs3dmm/zs_arcface_vs_scale.sbatch --dependency=afterok:<embed>
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv

echo "[arc-vs-scale] host=$(hostname) job=${SLURM_JOB_ID:-none}"
"$AAU_DIR/run.sh" aau/zs3dmm/zs_arcface_vs_scale.py --eval-seed "${WBES_EVAL_SEED:-1234}"
