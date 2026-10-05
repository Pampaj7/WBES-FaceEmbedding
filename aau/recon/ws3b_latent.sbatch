#!/usr/bin/env bash
#SBATCH --job-name=wbes-ws3b-latent
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#
# WS3b, passo 3b: distanza latente del checkpoint v1 fra ricostruzione e GT, e fra
# ricostruzioni per l'AUC di identita'.
#
#   aau/submit.sh recon/ws3b_latent.sbatch
#
# L'unico pezzo del cantiere che vuole una GPU (12073 forward dell'encoder: 2848 GT piu'
# 3 x 3075 ricostruzioni).  Il grosso del tempo e' pero' l'I/O degli npz con gli operatori
# su CephFS (19 GB, letti uno alla volta da `GTReadyDataset` senza worker), non il calcolo:
# le mesh sono da 2660 e 2700 vertici e la GPU resta quasi ferma.  Per questo la corsa
# vera e' stata fatta su una T4 (`--gres=gpu:t4:1` da riga di comando, che vince sulla
# direttiva qui sopra): le L40S erano tutte occupate dal training e dagli eval, e il tetto
# di 12 GPU per utente era a un passo.
# Gli embedding vanno in cache in `embeddings/`, quindi risottomettere riprende.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv
aau_require_diffusion_net
aau_require_checkpoint

echo "[ws3b-latent] host=$(hostname) job=${SLURM_JOB_ID:-none} ckpt=$WBES_CKPT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$AAU_DIR/run.sh" aau/recon/ws3b_latent.py "$@"

echo "[ws3b-latent] OK"
