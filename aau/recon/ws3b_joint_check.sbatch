#!/usr/bin/env bash
#SBATCH --job-name=wbes-joint-check
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#
# Il latent di WS3a/WS3b (catena di ws3b_latent.py) riproduce la cella joint__bfm di WS2?
# 10 soggetti BFM held-out x 6 topologie = 60 forward, scenario clean.  Vedi
# ws3b_joint_check.py; esce con 1 se non riproduce, quindi i job sui dati reali si
# accodano con --dependency=afterok:<questo>.
#
#   WBES_CKPT=<congiunto> sbatch aau/recon/ws3b_joint_check.sbatch
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv
aau_require_diffusion_net
aau_require_checkpoint

echo "[joint-check] host=$(hostname) job=${SLURM_JOB_ID:-none} ckpt=$WBES_CKPT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$AAU_DIR/run.sh" aau/recon/ws3b_joint_check.py "$@"

echo "[joint-check] OK"
