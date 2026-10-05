#!/usr/bin/env bash
#SBATCH --job-name=wbes-recon-setup
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#
# Due venv per i tre metodi di ricostruzione 3D da immagine (WS3b).
#   aau/submit.sh recon/setup_env.sbatch
#
#   external/venvs/ddfa   3DDFA_V2 + SynergyNet: torch del container, opencv headless,
#                         matplotlib (i loro utils lo importano in cima), cython per la
#                         NMS di FaceBoxes.
#   external/venvs/prnet  PRNet: tensorflow-cpu 2.15 + tf_slim (il codice e' TF1 con
#                         tf.contrib, e tf_slim e' contrib.slim estratto dagli autori di
#                         TF).  Sta in un venv suo perche' tensorflow si tira dietro
#                         numpy/protobuf/typing-extensions e non deve avvicinarsi ne' a
#                         .venv_aau ne' al venv torch.
#
# Ne' .venv_aau ne' .venv_baselines vengono toccati.  I due venv nuovi ereditano torch e
# numpy del container (--system-site-packages) come tutti gli altri del repo.
#
# Prerequisito: aau/recon/fetch_assets.sh (frontend, ha git e internet).
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

export EXTERNAL="$WBES_ROOT/external"
export VENV_DDFA="$EXTERNAL/venvs/ddfa"
export VENV_PRNET="$EXTERNAL/venvs/prnet"

echo "[recon-setup] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[recon-setup] CONTAINER=$CONTAINER"

for repo in 3DDFA_V2 SynergyNet PRNet; do
    if [[ ! -d "$EXTERNAL/$repo" ]]; then
        echo "ERRORE: clone mancante: $EXTERNAL/$repo" >&2
        exit 1
    fi
done
if [[ ! -f "$EXTERNAL/SynergyNet/pretrained/best.pth.tar" || \
      ! -f "$EXTERNAL/PRNet/Data/net-data/256_256_resfcn256_weight.data-00000-of-00001" ]]; then
    echo "ERRORE: asset mancanti. Esegui prima: aau/recon/fetch_assets.sh" >&2
    exit 1
fi

if [[ "${RECON_FORCE_VENV:-0}" == "1" ]]; then
    echo "[recon-setup] rimuovo i venv esistenti"
    rm -rf "$VENV_DDFA" "$VENV_PRNET"
fi
mkdir -p "$EXTERNAL/venvs"

singularity exec "$CONTAINER" bash -c '
set -euo pipefail

# numpy<2 come per le baseline: numpy 2.x nel venv shadowa quello del container e fa
# esplodere torch/torchvision NGC.  typing_extensions non deve scendere sotto quello che
# vuole torch 2.5, altrimenti a rompersi e FaceBoxes dentro il venv di PRNet.
printf "numpy<2\ntyping_extensions>=4.8\n" > "$EXTERNAL/venvs/pip-constraints.txt"
export PIP_CONSTRAINT="$EXTERNAL/venvs/pip-constraints.txt"

for v in "$VENV_DDFA" "$VENV_PRNET"; do
    if [[ ! -x "$v/bin/python3" ]]; then
        echo "[recon-setup] creo il venv $v"
        python3 -m venv --system-site-packages --without-pip "$v"
    fi
done

echo "[recon-setup] --- venv ddfa (3DDFA_V2 + SynergyNet) ---"
source "$VENV_DDFA/bin/activate"
python3 -m pip install --no-input "opencv-python-headless<5" matplotlib pyyaml cython
python3 -m pip list --local

# La NMS di FaceBoxes e un .pyx: va compilata in-place dentro ciascun clone (i due repo
# ne hanno una copia a testa, con la stessa sorgente).  Il .so finisce nel clone, non nel
# venv, ed e valido per entrambi i venv perche il python e lo stesso del container.
for fb in "$EXTERNAL/3DDFA_V2/FaceBoxes" "$EXTERNAL/SynergyNet/FaceBoxes"; do
    echo "[recon-setup] compilo cpu_nms in $fb"
    (cd "$fb/utils" && python3 build.py build_ext --inplace)
done
deactivate

echo "[recon-setup] --- venv prnet (PRNet) ---"
source "$VENV_PRNET/bin/activate"
# tensorflow-cpu 2.15 e l ultima serie con keras 2 e numpy<2: tf_slim (2020) non regge
# keras 3, e i wheel con CUDA vogliono cuDNN 8 mentre il container ha cuDNN 9.
# NON aggiungere tf-keras qui (provato nel job 1019443): lo dichiara come dipendenza
# `tensorflow` e non `tensorflow-cpu`, quindi pip tira giu il wheel GPU da 475 MB e lo
# installa SOPRA tensorflow-cpu; e in 2.15.1 non ha nemmeno il legacy_tf_layers che serve.
# Il modulo mancante c e gia dentro keras 2.15: lo aggancia prnet_run.py.
python3 -m pip install --no-input "tensorflow-cpu==2.15.1" tf_slim scikit-image "opencv-python-headless<5"
python3 -m pip list --local
deactivate
'

echo "[recon-setup] --- import test ---"
"$AAU_DIR/recon/run_recon.sh" ddfa -c "
import cv2, matplotlib, yaml, torch
print('cv2', cv2.__version__, 'torch', torch.__version__)
import sys; sys.path.insert(0, '$EXTERNAL/3DDFA_V2')
from FaceBoxes.utils.nms.cpu_nms import cpu_nms
print('cpu_nms 3DDFA_V2 ok')
"
"$AAU_DIR/recon/run_recon.sh" ddfa -c "
import sys; sys.path.insert(0, '$EXTERNAL/SynergyNet')
from FaceBoxes.utils.nms.cpu_nms import cpu_nms
print('cpu_nms SynergyNet ok')
"
"$AAU_DIR/recon/run_recon.sh" prnet -c "
import numpy, torch, tensorflow as tf, tf_slim, skimage
print('numpy', numpy.__version__, 'torch', torch.__version__)
print('tf', tf.__version__, 'tf_slim ok, skimage', skimage.__version__)
import keras.src.legacy_tf_layers.normalization as n
print('legacy BatchNormalization', n.BatchNormalization.__module__)
"

echo "[recon-setup] OK"
