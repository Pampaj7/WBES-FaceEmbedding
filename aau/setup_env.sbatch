#!/usr/bin/env bash
#SBATCH --job-name=wbes-setup
# --chdir e i log sono path ASSOLUTI: le direttive #SBATCH non espandono variabili, e un
# --output relativo viene risolto nella submit dir (se non esiste, Slurm scarta il job in
# un secondo senza scrivere niente). Se sposti il repo, aggiorna queste tre righe oppure
# sottometti con aau/submit.sh, che le ricalcola e le passa da riga di comando.
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#
# Crea il venv aau dentro il container e installa cio' che manca.
# Da sottomettere dalla root del repo:  sbatch aau/setup_env.sbatch
#
# Prerequisito (da fare sul frontend, dove c'e' git e internet):
#   git clone --depth 1 https://github.com/nmwsharp/diffusion-net.git diffusion-net
#
# Gira su partizione cpu senza --gres: l'installazione non usa la GPU.
# Il check finale stampera' quindi "cuda available: False" — normale.
# Per verificare anche la GPU c'e' aau/smoke_test.sbatch.
set -euo pipefail

# Nodo senza GPU: niente --nv per singularity.
export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_diffusion_net

echo "[setup] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[setup] CONTAINER=$CONTAINER"
echo "[setup] VENV=$VENV"

# WBES_FORCE_VENV=1 per rifare il venv da zero.
if [[ "${WBES_FORCE_VENV:-0}" == "1" && -d "$VENV" ]]; then
    echo "[setup] rimuovo il venv esistente"
    rm -rf "$VENV"
fi

singularity exec "$CONTAINER" bash -c '
set -euo pipefail

if [[ ! -x "$VENV/bin/python3" ]]; then
    # --without-pip: il container NGC non ha ensurepip, quindi il venv normale fallisce.
    # Non serve un pip dentro il venv: quello di sistema, invocato col python del venv,
    # installa comunque nel venv perche sys.prefix punta li.
    echo "[setup] creo il venv (--system-site-packages: eredita torch del container)"
    python3 -m venv --system-site-packages --without-pip "$VENV"
fi
source "$VENV/bin/activate"

echo "[setup] python=$(python3 -c "import sys;print(sys.version.split()[0])") torch=$(python3 -c "import torch;print(torch.__version__)")"

# numpy scipy scikit-learn tqdm tensorboard (e pip, setuptools, wheel) sono gia nel
# container NGC 24.10; nel venv finiscono i tre pacchetti geometrici mancanti piu
# zstandard, che serve solo a spacchettare i .tar.zst scaricati da Hugging Face
# (il container non ha ne il binario zstd ne il modulo python).
python3 -m pip install --no-input libigl potpourri3d robust-laplacian zstandard

echo "[setup] --- pacchetti installati nel venv (non ereditati) ---"
python3 -m pip list --local
'

echo "[setup] --- import test diffusion_net ---"
"$AAU_DIR/run.sh" -c "
import diffusion_net, igl, potpourri3d, robust_laplacian
print('diffusion_net', diffusion_net.__file__)
print('igl', getattr(igl, '__version__', '?'))
print('potpourri3d ok, robust_laplacian ok')
"

echo "[setup] --- check_twotower_robust_env.py ---"
"$AAU_DIR/run.sh" scripts/check_twotower_robust_env.py

echo "[setup] OK"
