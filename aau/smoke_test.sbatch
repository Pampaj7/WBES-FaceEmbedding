#!/usr/bin/env bash
#SBATCH --job-name=wbes-smoke
# --chdir e i log sono path ASSOLUTI: le direttive #SBATCH non espandono variabili, e un
# --output relativo viene risolto nella submit dir (se non esiste, Slurm scarta il job in
# un secondo senza scrivere niente). Se sposti il repo, aggiorna queste tre righe oppure
# sottometti con aau/submit.sh, che le ricalcola e le passa da riga di comando.
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#
# Verifica su GPU vera che venv + container + diffusion_net + pacchetti geometrici
# si importino e che torch veda la L40S.  sbatch aau/smoke_test.sbatch
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

"$AAU_DIR/run.sh" -c "import diffusion_net, igl, potpourri3d, robust_laplacian, torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"
"$AAU_DIR/run.sh" scripts/check_twotower_robust_env.py
