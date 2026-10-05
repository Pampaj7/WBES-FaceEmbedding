#!/usr/bin/env bash
#SBATCH --job-name=wbes-ict-train-ready
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#
# ICT-5000, passo 5: vista `idNNNN_GTready_<topo>.npz` (symlink, offset id 10000) sugli
# operatori, piu' la matrice GT rinominata e lo split 4500/500.
#
#   aau/submit.sh ict/ict_train_ready.sbatch --dependency=afterok:<job operatori>
#
# Lo stesso script gira gia' dentro ict_gt_matrix.sbatch per pubblicare lo split presto;
# qui viene ripetuto quando gli operatori esistono, cosi' i 30.000 symlink si creano.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

# `ls dir/*.npz` esplode con "Argument list too long" a 30.000 file (misurato: job 1019283
# e' fallito cosi' DOPO aver scritto correttamente tutte le mesh). find non espande niente.
count_npz() { find "$1" -maxdepth 1 -name "$2" -type f -o -maxdepth 1 -name "$2" -type l | wc -l; }
aau_require_venv

GT_DIR="${WBES_ICT_GT_DIR:-$WBES_ROOT/datasets/ICT/gt}"
WITHOPS_DIR="${WBES_ICT_WITHOPS_DIR:-$WBES_ROOT/datasets/ICT/topo_withops}"
READY_DIR="${WBES_ICT_READY_DIR:-$WBES_ROOT/datasets/ICT/train_ready}"

echo "[ict-ready] host=$(hostname) job=${SLURM_JOB_ID:-none}"

"$AAU_DIR/run.sh" v2_work/genict/make_train_ready.py \
    --withops-dir "$WITHOPS_DIR" \
    --gt-npz "$GT_DIR/ict_matrix_distances_maxabs.npz" \
    --out-dir "$READY_DIR"

echo "[ict-ready] symlink: $(count_npz "$READY_DIR/npz_withops" "*.npz")"
echo "[ict-ready] split: $(wc -l < "$READY_DIR/split_train.txt") train, $(wc -l < "$READY_DIR/split_heldout.txt") held-out"
