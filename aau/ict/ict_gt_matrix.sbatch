#!/usr/bin/env bash
#SBATCH --job-name=wbes-ict-gt
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#
# ICT-5000, passo 3: matrice GT vertex-mean-L2 normalizzata fra tutte le coppie sulla
# topologia canonica (`original`), stesse chiavi della matrice BFM
# (`normalized_matrix_distances.npz`: D_orig float32 + names).
#
#   aau/submit.sh ict/ict_gt_matrix.sbatch --dependency=afterok:<job topologie>
#
# Subito dopo scrive la vista train-ready (passo 5) SENZA symlink: a questo punto gli
# operatori non ci sono ancora, ma lo split 4500/500 serve al passo 6 (espressioni), che
# non deve aspettare l'ora di precompute. Lo stesso script va rilanciato dopo gli
# operatori per creare i 30.000 symlink.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv

TOPO_DIR="${WBES_ICT_TOPO_DIR:-$WBES_ROOT/datasets/ICT/topo}"
GT_DIR="${WBES_ICT_GT_DIR:-$WBES_ROOT/datasets/ICT/gt}"
WITHOPS_DIR="${WBES_ICT_WITHOPS_DIR:-$WBES_ROOT/datasets/ICT/topo_withops}"
READY_DIR="${WBES_ICT_READY_DIR:-$WBES_ROOT/datasets/ICT/train_ready}"

echo "[ict-gt] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[ict-gt] topo=$TOPO_DIR out=$GT_DIR"

"$AAU_DIR/run.sh" v2_work/genict/build_ict_gt_matrix.py \
    --topo-dir "$TOPO_DIR" --out-dir "$GT_DIR"

echo "[ict-gt] vista train-ready + split 4500/500 (symlink solo se gli operatori esistono)"
mkdir -p "$WITHOPS_DIR"
"$AAU_DIR/run.sh" v2_work/genict/make_train_ready.py \
    --withops-dir "$WITHOPS_DIR" \
    --gt-npz "$GT_DIR/ict_matrix_distances_maxabs.npz" \
    --out-dir "$READY_DIR"
