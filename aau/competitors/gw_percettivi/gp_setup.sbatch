#!/usr/bin/env bash
#SBATCH --job-name=wbes-gp-setup
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#
# Venv di Gromov-Wasserstein e delle feature percettive (gp_env.sh), dentro il container come
# aau/competitors/comp_setup.sbatch: --system-site-packages (torch del container) e solo i pacchetti che mancano,
# torch / torchvision / numpy bloccati sulle versioni del container. Pesi scaldati in $GP_WEIGHTS (fuori dal repo).
#
#   aau/submit.sh competitors/gw_percettivi/gp_setup.sbatch
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
source "$WBES_ROOT/aau/competitors/gw_percettivi/gp_env.sh"
cd "$WBES_ROOT"

echo "[gp-setup] host=$(hostname) job=${SLURM_JOB_ID:-none} venv=$GP_VENV pesi=$GP_WEIGHTS"
singularity exec "$CONTAINER" bash -c '
set -euo pipefail
if [[ ! -x "$GP_VENV/bin/python3" ]]; then
    python3 -m venv --system-site-packages --without-pip "$GP_VENV"
fi
source "$GP_VENV/bin/activate"
python3 -c "import torch, torchvision, numpy; print(f\"torch=={torch.__version__}\ntorchvision=={torchvision.__version__}\nnumpy=={numpy.__version__}\")" \
    > "$GP_VENV/pip-constraints.txt"
cat "$GP_VENV/pip-constraints.txt"
export PIP_CONSTRAINT="$GP_VENV/pip-constraints.txt"
# peft e transformers coevi di DreamSim 0.2.1: le versioni nuove importano transformer_engine del container (che
# vuole una GPU al momento del caricamento) e rifiutano il torch "2.5.0a0" del container
printf "huggingface_hub<0.26\ntransformers==4.44.2\npeft==0.11.1\naccelerate<1.1\n" >> "$GP_VENV/pip-constraints.txt"
python3 -m pip install --no-cache-dir "pot==0.9.5" lpips open_clip_torch "timm<1.1" huggingface_hub transformers peft accelerate
python3 -m pip install --no-cache-dir --no-deps "dreamsim==0.2.1"
# I pacchetti geometrici del venv del repo (igl, potpourri3d, robust_laplacian: blmm, zs_stage) in coda al sys.path
echo "$VENV/lib/python3.10/site-packages" > "$GP_VENV/lib/python3.10/site-packages/wbes_aau_venv.pth"
python3 -c "
import ot, lpips, open_clip, timm, dreamsim, peft, torch, potpourri3d, robust_laplacian, igl
print(\"pot\", ot.__version__, \"timm\", timm.__version__, \"torch\", torch.__version__, \"peft\", peft.__version__)
"
python3 -m pip freeze > "$GP_VENV/freeze.txt"
'
# pesi: DINOv2 e CLIP come v2_work/phase0/perceptual_embed.py, LPIPS alex, DreamSim ensemble
if [[ -f aau/competitors/gw_percettivi/gp_percettivi.py ]]; then
    aau/competitors/gw_percettivi/gp_run.sh aau/competitors/gw_percettivi/gp_percettivi.py warm --device cpu
fi
echo "[gp-setup] OK"
