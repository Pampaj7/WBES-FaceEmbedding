#!/usr/bin/env bash
#SBATCH --job-name=wbes-d3
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#
# Diagnostica D3 (aau/runs/evidence/diagnostics/PROTOCOL_D3.md), passi CPU; un passo per job:
#   D3_STEP=test                                     sbatch aau/diagnostics/d3.sbatch   (prove di d3_head.py)
#   D3_STEP=famos                                    sbatch aau/diagnostics/d3.sbatch   (mesh, tabella, GT, operatori)
#   D3_STEP=pools sbatch --cpus-per-task=64 --mem=128G aau/diagnostics/d3.sbatch          (pool non valutati, operatori)
#   D3_STEP=stats sbatch --cpus-per-task=64 --mem=128G aau/diagnostics/d3.sbatch          (dopo d3_embed.sbatch)
#   D3_STEP=summary                                  sbatch aau/diagnostics/d3.sbatch   (results_d3.md)
# Gli embedding (FaMoS TRAIN, C3M e123 su FaceVerse neutra): aau/diagnostics/d3_embed.sbatch (A100). famos: mesh grezze in
# datasets/DIAG_D3/famos/in, operatori areanorm k_eig 128 in datasets/V3_OPS_CACHE/diag_d3/famos/ops (fuori da git, da
# cancellare a fine lavoro), come diag.sbatch. pools (emendamento 1): messa in scena in datasets/DIAG_D3/pools/<pool>/in,
# operatori in datasets/V3_OPS_CACHE/diag_d3/pools/<pool>/ops (circa 50 GB, da cancellare a fine lavoro).
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
STEP="${D3_STEP:?D3_STEP: test | famos | pools | stats | summary}"
W="${SLURM_CPUS_PER_TASK:-8}"
export AAU_NV=""
D=aau/diagnostics
IN="$WBES_ROOT/datasets/DIAG_D3/famos/in"
OPS="$WBES_ROOT/datasets/V3_OPS_CACHE/diag_d3/famos/ops"
echo "[d3] host=$(hostname) job=${SLURM_JOB_ID:-none} passo=$STEP $(date +%T)"
t0=$(date +%s)
ops() {  # <cartella delle mesh> <cartella degli operatori>: areanorm k_eig 128 in W-1 shard, controllo del conteggio
  local src="$1" out="$2" i N M pids=()
  N=$(find "$src" -name "*.npz" | wc -l)
  mkdir -p "$out"
  for (( i = 0; i < W - 1; i++ )); do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 aau/run.sh v2_work/potential/areanorm_operators.py \
      --input-dir "$src" --output-dir "$out" --k-eig 128 --shard "$i/$((W - 1))" \
      > "/tmp/wbes_d3_ops_${SLURM_JOB_ID:-x}.$i.log" 2>&1 & pids+=($!)
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "ERRORE: shard $i"; cat "/tmp/wbes_d3_ops_${SLURM_JOB_ID:-x}.$i.log"; return 1; }
  done
  rm -f /tmp/wbes_d3_ops_"${SLURM_JOB_ID:-x}".*.log
  M=$(find "$out" -name "*.npz" ! -name ".*" | wc -l)
  echo "[d3] $(date +%T) operatori $M / $N in $out, $(du -sh "$out" | cut -f1)"
  (( M == N )) || { echo "ERRORE: operatori incompleti"; return 1; }
}
case "$STEP" in
    test)      aau/run.sh $D/test_d3.py ;;
    famos)
        aau/run.sh $D/d3_famos.py --workers "$W"
        ops "$IN" "$OPS" ;;
    pools)
        aau/run.sh $D/d3_pools.py
        for p in facescape hifi3d faceverse; do
          ops "$WBES_ROOT/datasets/DIAG_D3/pools/$p/in" "$WBES_ROOT/datasets/V3_OPS_CACHE/diag_d3/pools/$p/ops"
        done ;;
    stats)     UGT_THREADS=1 v3_work/unified_gt/run.sh $D/d3_stats.py --workers "$W" ;;
    summary)   aau/run.sh $D/d3_summarize.py ;;
    *)         echo "passo sconosciuto: $STEP"; exit 2 ;;
esac
echo "[d3] passo $STEP finito in $(( $(date +%s) - t0 ))s"
