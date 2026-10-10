#!/usr/bin/env bash
#SBATCH --job-name=wbes-diag
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#
# Diagnostica D (aau/runs/evidence/diagnostics/PROTOCOL_D.md), passi CPU; un passo per job:
#   DIAG_STEP=test                                   sbatch aau/diagnostics/diag.sbatch   (prove di diag.py)
#   DIAG_STEP=gen                                    sbatch aau/diagnostics/diag.sbatch   (mesh, tabella, GT, operatori)
#   DIAG_STEP=bp_static                              sbatch aau/diagnostics/diag.sbatch   (distanze B sui visti)
#   DIAG_STEP=bp_fit  sbatch --cpus-per-task=64 --mem=96G aau/diagnostics/diag.sbatch    (fit B sugli insiemi FLAME)
#   DIAG_STEP=d1                                     sbatch aau/diagnostics/diag.sbatch   (dopo embed.sbatch e bp_*)
#   DIAG_STEP=d2                                     sbatch aau/diagnostics/diag.sbatch   (sonda di D2, indipendente)
#   DIAG_STEP=summary                                sbatch aau/diagnostics/diag.sbatch   (results.md)
# Gli embedding dei bracci sugli insiemi nuovi: aau/diagnostics/embed.sbatch (A100). Partizione prioritized senza GPU
# (la cpu e' in fail). gen: mesh grezze in datasets/DIAG_D1/in, operatori areanorm k_eig 128 in
# datasets/V3_OPS_CACHE/diag_d1/ops (fuori da git, ~25 GB, da cancellare a fine lavoro), come calib_heldout.sbatch.
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
STEP="${DIAG_STEP:?DIAG_STEP: test | gen | bp_static | bp_fit | d1 | d2 | summary}"
W="${SLURM_CPUS_PER_TASK:-8}"
export AAU_NV=""
D=aau/diagnostics
OPS="$WBES_ROOT/datasets/V3_OPS_CACHE/diag_d1"
echo "[diag] host=$(hostname) job=${SLURM_JOB_ID:-none} passo=$STEP $(date +%T)"
t0=$(date +%s)
case "$STEP" in
    test)      aau/run.sh $D/test_diag.py ;;
    gen)
        aau/run.sh $D/d1_gen.py --workers "$W"
        N=$(find datasets/DIAG_D1/in -name "*.npz" | wc -l)
        mkdir -p "$OPS/ops"
        pids=()
        for (( i = 0; i < W - 1; i++ )); do
          OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 aau/run.sh v2_work/potential/areanorm_operators.py \
            --input-dir datasets/DIAG_D1/in --output-dir "$OPS/ops" --k-eig 128 --shard "$i/$((W - 1))" \
            > "/tmp/wbes_diag_ops_${SLURM_JOB_ID:-x}.$i.log" 2>&1 & pids+=($!)
        done
        for i in "${!pids[@]}"; do
          wait "${pids[$i]}" || { echo "ERRORE: shard $i"; cat "/tmp/wbes_diag_ops_${SLURM_JOB_ID:-x}.$i.log"; exit 1; }
        done
        rm -f /tmp/wbes_diag_ops_"${SLURM_JOB_ID:-x}".*.log
        M=$(find "$OPS/ops" -name "*.npz" ! -name ".*" | wc -l)
        echo "[diag] $(date +%T) operatori $M / $N, $(du -sh "$OPS/ops" | cut -f1)"
        (( M == N )) || { echo "ERRORE: operatori incompleti"; exit 1; } ;;
    bp_static) aau/outlineB/run_o3d.sh $D/d1_bp.py static ;;
    bp_fit)    aau/outlineB/run_o3d.sh $D/d1_bp.py fit --workers "$W" ;;
    d1)        aau/run.sh $D/d1_stats.py --workers "$W" ;;
    d2)        UGT_THREADS=1 v3_work/unified_gt/run.sh $D/d2_probe.py --workers "$W" ;;
    summary)   aau/run.sh $D/summarize.py ;;
    *)         echo "passo sconosciuto: $STEP"; exit 2 ;;
esac
echo "[diag] passo $STEP finito in $(( $(date +%s) - t0 ))s"
