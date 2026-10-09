#!/usr/bin/env bash
#SBATCH --job-name=wbes-e1-summarize
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:30:00
#
# E1, riepilogo: aau/runs/evidence/e1/summary.md (e1_summarize.py) e la tabella standard di aau/zs3dmm
# sui bracci HIFI3D delle celle (e1_zs_summarize.py -> aau/runs/evidence/e1/hifi/). Niente GPU.
#   aau/submit.sh evidence/e1_factorial/e1_summarize.sbatch [--dependency=afterany:<eval>]
set -euo pipefail
export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv
echo "[e1-sum] host=$(hostname) job=${SLURM_JOB_ID:-none}"
"$AAU_DIR/run.sh" aau/evidence/e1_factorial/e1_summarize.py
bash "$AAU_DIR/evidence/e1_factorial/e1_zs_summarize.sh" || echo "[e1-sum] ATTENZIONE: tabella standard zs3dmm non scritta"
