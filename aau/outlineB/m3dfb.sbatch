#!/usr/bin/env bash
#SBATCH --job-name=wbes-oB-m3dfb
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=30
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#
# M3DFB RLR + Chamfer (E9, aau/outlineB/m3dfb_matrix.py) su un set di soggetti:
# original->original + le 20 coppie cross no-crop. Solo CPU, venv con open3d (M3DFB lo
# importa anche se E9 non lo usa).
#
#   OB_SET=heldout aau/submit.sh outlineB/m3dfb.sbatch --partition=prioritized --qos=unprivileged
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

OB_SET="${OB_SET:?OB_SET=heldout|facebench_first100|ict_heldout}"
case "$OB_SET" in
    heldout)            OUT="$AAU_RUNS/outlineB/bfm_heldout" ;;
    facebench_first100) OUT="$AAU_RUNS/outlineB/bfm_fb100" ;;
    ict_heldout)        OUT="$AAU_RUNS/outlineB/ict_heldout"
                        export WBES_REMESH_NOOPS_DIR="$WBES_ROOT/datasets/ICT/eval_view_heldout"
                        export WBES_DIST_NPZ="$WBES_ROOT/datasets/ICT/train_ready/gt_matrix.npz"
                        export WBES_EVAL_SEED="${WBES_EVAL_SEED:-1234}" ;;
    *) echo "OB_SET sconosciuto: $OB_SET" >&2; exit 2 ;;
esac
echo "[oB-m3dfb] host=$(hostname) job=${SLURM_JOB_ID:-none} set=$OB_SET out=$OUT"

"$AAU_DIR/outlineB/run_o3d.sh" aau/outlineB/m3dfb_matrix.py \
    --subject-set "$OB_SET" --out-root "$OUT" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" ${OB_ARGS:-}
echo "[oB-m3dfb] OK"
