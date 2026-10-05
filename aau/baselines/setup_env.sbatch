#!/usr/bin/env bash
#SBATCH --job-name=wbes-bl-setup
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#
# Venv separato per le baseline percettive: insightface (ArcFace), open_clip, lpips.
#   aau/submit.sh baselines/setup_env.sbatch
#
# NON si tocca .venv_aau: training ed eval ci girano dentro adesso, e insightface tira
# dietro numpy/opencv/onnxruntime che potrebbero sovrascrivere quelli del container.
# .venv_baselines eredita comunque torch/numpy/scipy del container
# (--system-site-packages) e ci aggiunge solo il resto.
#
# Il probe (aau/baselines/probe_env.py) ha verificato che i nodi vedono pypi.org,
# github.com e huggingface.co, ma NON dl.insightface.net: i pesi buffalo_l vengono presi
# dalle release GitHub di insightface, ed e' per quello che qui si scaldano le cache.
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
export VENV="$WBES_ROOT/.venv_baselines"
cd "$WBES_ROOT"

echo "[bl-setup] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[bl-setup] VENV=$VENV"

if [[ "${WBES_FORCE_VENV:-0}" == "1" && -d "$VENV" ]]; then
    echo "[bl-setup] rimuovo il venv esistente"
    rm -rf "$VENV"
fi

singularity exec "$CONTAINER" bash -c '
set -euo pipefail

if [[ ! -x "$VENV/bin/python3" ]]; then
    echo "[bl-setup] creo il venv (--system-site-packages, --without-pip come .venv_aau)"
    python3 -m venv --system-site-packages --without-pip "$VENV"
fi
source "$VENV/bin/activate"

# Due paletti, imparati sbagliando (job 1019231):
#  - numpy: le dipendenze tirano su numpy 2.x, che nel venv SHADOWA quello del container
#    e fa esplodere torch/torchvision NGC, compilati contro numpy 1.x. Il constraint lo
#    blocca a <2, e cosi restiamo sul 1.24.4 gia presente (--system-site-packages).
#  - opencv: insightface dipende da opencv-python, il cui wheel vuole libxcb.so.1, che
#    nel container non c e (ImportError all import di cv2). Il cv2 4.7.0 del container si
#    importa ma e una build ridotta, senza il modulo dnn che serve ad ArcFace
#    (AttributeError: module cv2 has no attribute dnn). La versione che va bene e
#    opencv-python-headless 4.x: niente libreria X, modulo dnn presente, numpy 1.x.
printf "numpy<2\n" > "$VENV/pip-constraints.txt"
export PIP_CONSTRAINT="$VENV/pip-constraints.txt"
python3 -m pip install --no-input --no-deps insightface
python3 -m pip install --no-input "opencv-python-headless<5" onnxruntime scikit-image open_clip_torch lpips

echo "[bl-setup] --- pacchetti locali al venv ---"
python3 -m pip list --local
'

echo "[bl-setup] --- scaldo le cache dei pesi ---"
"$AAU_DIR/baselines/run_bl.sh" aau/baselines/warm_caches.py

echo "[bl-setup] OK"
