#!/usr/bin/env bash
#SBATCH --job-name=wbes-comp-summarize
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#
# Tabella dei competitori su HIFI3D (comp_summarize.py): Spearman con CI, delta appaiati contro e108
# e Chamfer eval, riconoscimento. Niente GPU. Scrive aau/runs/competitors_hifi3d/summary.md.
#
#   aau/submit.sh competitors/comp_summarize.sbatch --dependency=afterok:<spettrali>:<embedding>
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
source "$WBES_ROOT/aau/competitors/comp_env.sh"
cd "$WBES_ROOT"
aau_require_venv

echo "[comp-sum] host=$(hostname) job=${SLURM_JOB_ID:-none}"
"$AAU_DIR/run.sh" aau/competitors/comp_summarize.py --out-dir "$COMP_OUT" --zs-runs "$COMP_ZS_RUNS" \
    --view-dir "$COMP_VIEW_DIR" --eval-seed "${WBES_EVAL_SEED:-1234}"
