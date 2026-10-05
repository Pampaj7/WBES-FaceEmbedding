#!/usr/bin/env bash
#SBATCH --job-name=wbes-x3dmm-view
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#
# Vista congiunta BFM(areanorm) + ICT per la tabella cross-3DMM di WS2: 33.000 symlink in
# una sola cartella piatta piu' una GT a blocchi diagonali con NaN fuori blocco.
# Lo script e' quello gia' usato per BFM+FLAME (v2_work/train_v2/make_joint_view.py): qui
# gli si passa solo il dominio ict e l'override della cartella BFM su npz_data_topo_500_withops_areanorm,
# perche' entrambi i domini devono avere operatori nella stessa convenzione (area unitaria).
#
#   aau/submit.sh cross3dmm/joint_view.sbatch
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv

BFM_DIR="${WBES_JOINT_BFM_DIR:-$WBES_ROOT/datasets/REMESH/npz_data_topo_500_withops_areanorm}"
ICT_DIR="${WBES_JOINT_ICT_DIR:-$WBES_ROOT/datasets/ICT/train_ready/npz_withops}"
OUT_DIR="${WBES_JOINT_DIR:-$WBES_ROOT/datasets/JOINT_BFM_ICT}"

echo "[x3dmm-view] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[x3dmm-view] bfm=$BFM_DIR"
echo "[x3dmm-view] ict=$ICT_DIR"
echo "[x3dmm-view] out=$OUT_DIR"

"$AAU_DIR/run.sh" v2_work/train_v2/make_joint_view.py \
    --domains bfm,ict \
    --data-dir "bfm=$BFM_DIR" \
    --data-dir "ict=$ICT_DIR" \
    --support-bank "" \
    --out-dir "$OUT_DIR"

# `ls dir/*.npz` esplode con "Argument list too long" a 33.000 file (job 1019283), find no.
echo "[x3dmm-view] symlink: $(find "$OUT_DIR/npz_withops" -maxdepth 1 -name '*.npz' | wc -l)"
