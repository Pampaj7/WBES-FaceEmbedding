#!/usr/bin/env bash
#SBATCH --job-name=wbes-comp-spectral
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#
# ShapeDNA, HKS e WKS globali delle 600 mesh HIFI3D valutate (comp_spectral.py). Niente GPU.
#
#   aau/submit.sh competitors/comp_spectral.sbatch
#   WBES_COMP_LIMIT=12 aau/submit.sh competitors/comp_spectral.sbatch --time=00:10:00   (prova)
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
source "$WBES_ROOT/aau/competitors/comp_env.sh"
cd "$WBES_ROOT"
aau_require_venv
# Un thread BLAS per processo: il parallelismo e' sulle mesh.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

limit="${WBES_COMP_LIMIT:-0}"
out="$COMP_OUT/desc_spectral.npz"
[[ "$limit" != 0 ]] && out="$COMP_OUT/desc_spectral_limit${limit}.npz"
echo "[comp-spec] host=$(hostname) job=${SLURM_JOB_ID:-none} limit=$limit"
"$AAU_DIR/run.sh" aau/competitors/comp_spectral.py --view-dir "$COMP_VIEW_DIR" --out "$out" \
    --eval-seed "${WBES_EVAL_SEED:-1234}" --limit "$limit"
