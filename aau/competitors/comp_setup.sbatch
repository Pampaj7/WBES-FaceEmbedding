#!/usr/bin/env bash
#SBATCH --job-name=wbes-comp-setup
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#
# Venv dei competitori neurali (Uni3D, OpenShape), dentro il container come aau/setup_env.sbatch:
# --system-site-packages (torch del container) e solo i pacchetti che mancano. Niente pointnet2_ops
# ne' dgl (estensioni CUDA da compilare): il loro FPS lo sostituisce comp_embed.py, in torch puro.
#
#   aau/submit.sh competitors/comp_setup.sbatch
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
source "$WBES_ROOT/aau/competitors/comp_env.sh"
cd "$WBES_ROOT"

echo "[comp-setup] host=$(hostname) job=${SLURM_JOB_ID:-none} venv=$COMP_VENV"
singularity exec "$CONTAINER" bash -c '
set -euo pipefail
if [[ ! -x "$COMP_VENV/bin/python3" ]]; then
    python3 -m venv --system-site-packages --without-pip "$COMP_VENV"
fi
source "$COMP_VENV/bin/activate"
# timm 0.9.16: ha eva_giant_patch14_560 con i nomi dei parametri del checkpoint di Uni3D.
# huggingface_hub fissato: le 2.x tirano dipendenze da un indice non raggiungibile dal nodo.
python3 -m pip install --no-cache-dir "timm==0.9.16" "huggingface_hub==0.23.4" easydict torch_redstone einops
# I pacchetti geometrici del venv del repo (igl, potpourri3d: li importa zs_stage.select_subjects) in coda
# al sys.path con un .pth: non coprono nessun pacchetto di questo venv (il venv del repo non ha timm).
echo "$VENV/lib/python3.10/site-packages" > "$COMP_VENV/lib/python3.10/site-packages/wbes_aau_venv.pth"
python3 -c "import timm, easydict, torch_redstone, einops, torch, igl; print(\"timm\", timm.__version__, \"torch\", torch.__version__)"
'
