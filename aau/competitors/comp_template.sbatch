#!/usr/bin/env bash
#SBATCH --job-name=wbes-comp-template
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#
# NICP su template (baseline di indomain_recog) sulle 600 mesh HIFI3D valutate: comp_template.py, nel
# venv con open3d. TEMPI (indomain): 1.2-2.6 s per mesh a worker -> 600 mesh, ~1 min a 32 core.
#
#   aau/submit.sh competitors/comp_template.sbatch
#   WBES_COMP_LIMIT=12 aau/submit.sh competitors/comp_template.sbatch --time=00:10:00 --cpus-per-task=12  (prova)
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
source "$WBES_ROOT/aau/competitors/comp_env.sh"
cd "$WBES_ROOT"

limit="${WBES_COMP_LIMIT:-0}"
out="$COMP_OUT/template_hifi3d.npz"
[[ "$limit" != 0 ]] && out="$COMP_OUT/template_hifi3d_limit${limit}.npz"
echo "[comp-tpl] host=$(hostname) job=${SLURM_JOB_ID:-none} limit=$limit"
"$AAU_DIR/outlineB/run_o3d.sh" aau/competitors/comp_template.py --view-dir "$COMP_VIEW_DIR" --out "$out" \
    --eval-seed "${WBES_EVAL_SEED:-1234}" --workers "${SLURM_CPUS_PER_TASK:-8}" --limit "$limit"
