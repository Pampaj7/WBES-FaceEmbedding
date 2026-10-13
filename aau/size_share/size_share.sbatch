#!/usr/bin/env bash
#SBATCH --job-name=wbes-size-share
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#
# Quota della taglia nella GT FR per dominio (aau/size_share/size_share.py; LEGGE LE GT di test) e la figura
# paper/figs/fig_size_share.pdf (paper/figs/scripts/fig_size_share.py). Solo CPU.
#   sbatch aau/size_share/size_share.sbatch
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export AAU_NV=""
export FIG_PREVIEW="${FIG_PREVIEW:-/home/create.aau.dk/ga41wf/containers/texbuild/out/figs_preview}"
mkdir -p "$FIG_PREVIEW"
echo "[size-share] host=$(hostname) job=${SLURM_JOB_ID:-none} $(date +%T)"
v3_work/unified_gt/run.sh -u aau/size_share/size_share.py
aau/run.sh paper/figs/scripts/fig_size_share.py
echo "[size-share] fine $(date +%T)"
