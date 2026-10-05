#!/usr/bin/env bash
#SBATCH --job-name=wbes-ws3a-latent
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#
# WS3a, distanza latente del checkpoint v1 sulle 8000 coppie del protocollo Multiface.
#
#   aau/submit.sh multiface/ws3a_latent.sbatch
#   WBES_CKPT=... aau/submit.sh multiface/ws3a_latent.sbatch      # un altro modello
#   WBES_WS3A_PAIRS=hard aau/submit.sh multiface/ws3a_latent.sbatch --time=04:00:00
#
# T4 e non L40S: il collo di bottiglia e' l'I/O degli npz con operatori (21 GB su CephFS),
# non la GPU — misurati 5-7 mesh/s con la T4 mai satura.  Le L40S sono occupate dai
# training areanorm e non c'e' niente da guadagnarci.
#
# --time: stimato dal test su 200 coppie (214 mesh per topologia) su T4, 29+40+39 s,
# cioe' ~6 mesh/s, quindi ~24 min per le 8520 mesh del protocollo intero.  Misurato poi
# sul giro vero (job 1019520): 25m18s, con 3.4-9.5 mesh/s a seconda della contesa sul
# nodo.  Le 2 h restano perche' la velocita' dipende da CephFS, non dalla GPU.
#
# Il job e' ripartibile: i latenti gia' calcolati stanno in embeddings/latent_v1_*.npz e
# i csv gia' presenti vengono saltati (senza --overwrite).
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv
aau_require_diffusion_net
aau_require_checkpoint

echo "[ws3a-latent] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[ws3a-latent] ckpt=$WBES_CKPT ops=${WBES_MF_OPS_SUFFIX:-_withops}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$AAU_DIR/run.sh" aau/multiface/ws3a_latent.py --device cuda "$@"

echo "[ws3a-latent] OK"
