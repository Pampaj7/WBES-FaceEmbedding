#!/usr/bin/env bash
#SBATCH --job-name=wbes-ws3a-perc
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#
# WS3a, render + ArcFace + LPIPS sulle 8000 coppie del protocollo Multiface.
#
#   aau/submit.sh multiface/ws3a_perceptual.sbatch
#   WBES_WS3A_EXTRACTORS=arcface,clip,dinov2,lpips aau/submit.sh multiface/ws3a_perceptual.sbatch
#   WBES_WS3A_PAIRS=hard aau/submit.sh multiface/ws3a_perceptual.sbatch --time=10:00:00
#
# Solo ArcFace, rifacendo calibrazione, embedding e csv (serve dopo un cambio di crop):
#   WBES_WS3A_EXTRACTORS=arcface WBES_WS3A_PERC_ARGS=--overwrite \
#     aau/submit.sh multiface/ws3a_perceptual.sbatch --gres=NONE --time=10:00:00
# Gli argomenti extra passano da WBES_WS3A_PERC_ARGS e non dopo il nome dello script:
# submit.sh mette tutto quello che viene dopo fra le OPZIONI di sbatch, non fra gli
# argomenti del job.
#
# Render e metriche nello stesso job perche' i 25560 png sono un artefatto intermedio da
# 3 GB: separarli vorrebbe dire due job che si passano quella roba via CephFS.
#
# T4 e non L40S: la GPU serve solo a LPIPS, che fa 1000 forward/s anche su T4 e in tutto
# ha 96000 forward da fare.  Il costo vero e' ArcFace, che gira su onnxruntime CPU — per
# quello 32 core e OMP_NUM_THREADS=1 (un thread per processo: senza, ogni worker prova a
# pinnare un thread per core del nodo e onnxruntime riempie il log di
# pthread_setaffinity_np failed, oltre a rallentare tutto).
#
# --time: misurato con --max-pairs-per-class 50 (1926 render) su T4 + 32 core assegnati,
# 24 processi di embedding — render 10 s (200/s), ArcFace 849 s (2.3 render/s in tutto,
# ~10 s per immagine per processo: e' il detector a 512x512 su CPU), LPIPS ~1000
# forward/s.  Il protocollo intero e' 25560 render (x13.3) e 96000 forward LPIPS, cioe'
# ~2 min di render, ~3 h di ArcFace e ~2 min di LPIPS.  6 h danno un fattore 2 di
# margine sul pezzo che conta.
#
# Il job e' ripartibile: i render gia' fatti e gli embedding in embeddings/arcface.npz
# vengono riusati, i csv gia' presenti saltati (senza --overwrite).
set -euo pipefail

# Senza GPU assegnata (p.es. il giro di solo ArcFace, che e' onnxruntime CPU) singularity
# non deve ricevere --nv.
if [[ -z "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}${GPU_DEVICE_ORDINAL:-}" ]]; then
    export AAU_NV=""
else
    export AAU_NV="${AAU_NV---nv}"
fi
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

EXTRACTORS="${WBES_WS3A_EXTRACTORS:-arcface,lpips}"
CORES="${SLURM_CPUS_PER_TASK:-8}"
RENDER_WORKERS="${WBES_WS3A_RENDER_WORKERS:-$CORES}"
EMBED_WORKERS="${WBES_WS3A_EMBED_WORKERS:-$CORES}"

# Un thread per processo: il parallelismo e' sui processi.  Vedi il commento sopra.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[ws3a-perc] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[ws3a-perc] estrattori=$EXTRACTORS render_workers=$RENDER_WORKERS embed_workers=$EMBED_WORKERS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

DEVICE="cuda"
if [[ -z "${AAU_NV}" ]]; then DEVICE="cpu"; fi

"$AAU_DIR/baselines/run_bl.sh" aau/multiface/ws3a_render.py --workers "$RENDER_WORKERS"
"$AAU_DIR/baselines/run_bl.sh" aau/multiface/ws3a_perceptual.py \
    --device "$DEVICE" --extractors "$EXTRACTORS" --embed-workers "$EMBED_WORKERS" \
    ${WBES_WS3A_PERC_ARGS:-} "$@"

echo "[ws3a-perc] OK"
