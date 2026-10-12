#!/usr/bin/env bash
#SBATCH --job-name=wbes-g0
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#
# Gate G0 (aau/runs/evidence/diagnostics/PROTOCOL_G0.md), solo CPU; un passo per job:
#   G0_STEP=test    sbatch --cpus-per-task=64 --mem=128G aau/diagnostics/g0.sbatch          (K5, prima dei numeri)
#   G0_STEP=teacher sbatch --cpus-per-task=48 --mem=160G --time=03:00:00 aau/diagnostics/g0.sbatch
#                   (mesh rigenerate e K3, frame check di tutti gli insiemi, poi render + crop + ArcFace per insieme)
#   G0_STEP=cost    sbatch --cpus-per-task=2 --mem=8G --time=00:30:00 aau/diagnostics/g0.sbatch   (un thread, sez. 7)
#   G0_STEP=stats   sbatch --cpus-per-task=64 --mem=128G aau/diagnostics/g0.sbatch          (dopo teacher)
# Mesh rigenerate e render in /tmp del job (contano contro --mem), cancellati in uscita. Un thread per processo:
# onnxruntime come arcface_fixed, BLAS come le statistiche.
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
STEP="${G0_STEP:?G0_STEP: test | teacher | cost | stats}"
W="${SLURM_CPUS_PER_TASK:-8}"
export AAU_NV=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 UGT_THREADS=1
D=aau/diagnostics
SETS="${G0_SETS:-hifi3d facescape faceverse_neutral flame2023_s1 famos bfm ict gnm}"
echo "[g0] host=$(hostname) job=${SLURM_JOB_ID:-none} passo=$STEP cpu=$W $(date +%T)"
t0=$(date +%s)
case "$STEP" in
    test)
        v3_work/unified_gt/run.sh "$D/g0_test.py" --workers "$W" ;;
    teacher)
        STAGE="/tmp/wbes_g0_${SLURM_JOB_ID:-manual}"
        rm -rf "$STAGE"
        trap 'rm -rf "$STAGE"' EXIT
        for s in flame2023_s1 famos heldout; do
            aau/run.sh "$D/g0_meshes.py" --set "$s" --out "$STAGE/meshes" --workers "$W"
        done
        echo "[g0] mesh rigenerate: $(find "$STAGE/meshes" -name 'id*.npz' | wc -l), $(du -sh "$STAGE/meshes" | cut -f1)"
        for s in $SETS; do        # tutti i frame check PRIMA di ogni embedding (sez. 2)
            aau/baselines/run_bl.sh "$D/g0_teacher.py" frame --set "$s" --mesh-root "$STAGE/meshes"
        done
        for s in $SETS; do
            aau/baselines/run_bl.sh "$D/g0_teacher.py" run --set "$s" --stage "$STAGE/render" \
                --mesh-root "$STAGE/meshes" --workers "$W"
            echo "[g0] $s: /tmp $(du -sh "$STAGE/render/$s" | cut -f1), $(( $(date +%s) - t0 ))s"
            rm -rf "$STAGE/render/$s"
        done ;;
    cost)
        aau/baselines/run_bl.sh "$D/g0_teacher.py" cost ;;
    stats)
        v3_work/unified_gt/run.sh "$D/g0_stats.py" --workers "$W" ;;
    *) echo "passo sconosciuto: $STEP" >&2; exit 2 ;;
esac
echo "[g0] OK passo=$STEP in $(( $(date +%s) - t0 ))s"
