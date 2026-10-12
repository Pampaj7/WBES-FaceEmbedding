#!/bin/bash
#SBATCH --job-name=paper-figs
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/create.aau.dk/ga41wf/containers/texbuild/out/figs_%j.log

# Figure del paper CVPR: tabelle (stdlib), figure 3-6 (matplotlib) e teaser (GNM Head, Apache-2.0) nel container
# del progetto (aau/run.sh, venv .venv_aau). Solo CPU. Uscite in paper/figs/ e paper/tables/.
#   sbatch paper/figs/scripts/make_figs.sbatch
set -euo pipefail
REPO=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
cd "$REPO"
export AAU_NV=""   # nessuna GPU
export FIG_PREVIEW="${FIG_PREVIEW:-/home/create.aau.dk/ga41wf/containers/texbuild/out/figs_preview}"
mkdir -p "$FIG_PREVIEW"
export WBES_GNM_NPZ="${WBES_GNM_NPZ:-$HOME/data/gnm_head/gnm_head.npz}"
echo "== host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID}"
python3 paper/figs/scripts/make_tables.py
aau/run.sh paper/figs/scripts/make_figures.py
aau/run.sh paper/figs/scripts/make_teaser.py
echo "== fine $(date -Is)"
