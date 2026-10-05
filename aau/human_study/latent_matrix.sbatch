#!/usr/bin/env bash
#SBATCH --job-name=wbes-hs-latent-matrix
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#
# WS4, matrice 100x100 della distanza latente del checkpoint v1 sui soggetti held-out in
# topologia original: e' l'input di `select_triplets.py --latent-matrix`.
#
#   aau/submit.sh human_study/latent_matrix.sbatch
#   WBES_CKPT=... aau/submit.sh human_study/latent_matrix.sbatch      # un altro modello
#
# T4 e non L40S: sono 100 forward, il tempo se ne va nel caricare 100 npz con operatori
# (~1.5 GB su CephFS) e nel costruire il modello, non nella GPU.
#
# Il job e' ripartibile: i latenti gia' calcolati stanno in
# aau/runs/baselines/embeddings/latent_v1_original.npz e la matrice gia' presente viene
# saltata (senza --overwrite).
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv
aau_require_diffusion_net
aau_require_dataset
aau_require_checkpoint

echo "[latent-matrix] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[latent-matrix] ckpt=$WBES_CKPT data=$WBES_DATA_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$AAU_DIR/run.sh" aau/human_study/latent_matrix.py --device cuda "$@"

echo "[latent-matrix] OK"
