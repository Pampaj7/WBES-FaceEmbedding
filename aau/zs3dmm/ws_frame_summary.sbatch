#!/usr/bin/env bash
#SBATCH --job-name=wbes-ws-frame-summary
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#
# Rigenera aau/runs/ws_frame/summary.md dai csv di zs_summarize.py (nessun calcolo).
#   aau/submit.sh zs3dmm/ws_frame_summary.sbatch --dependency=afterok:<summary delle varianti>
set -euo pipefail
export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
"$AAU_DIR/run.sh" aau/zs3dmm/ws_frame_summary.py
