#!/usr/bin/env bash
#SBATCH --job-name=wbes-blmm
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# Baseline geometriche con la normalizzazione coerente con la GT (aau/baselines_mm/blmm.py). Un passo per job:
#
#   BLMM_STEP=scalars                                         sbatch aau/baselines_mm/blmm.sbatch
#   BLMM_STEP=pairs BLMM_ARGS="hifi3d mm fast --shard 0/4"    sbatch aau/baselines_mm/blmm.sbatch
#   BLMM_STEP=pairs BLMM_ARGS="hifi3d maxabs nicp --check 200" sbatch aau/baselines_mm/blmm.sbatch
#   BLMM_STEP=template BLMM_ARGS="--views hifi3d --modes mm,cs" sbatch aau/baselines_mm/blmm.sbatch
#   BLMM_STEP=eval BLMM_ARGS="--views hifi3d"                 sbatch aau/baselines_mm/blmm.sbatch
#
# Calcolo nel venv con open3d (aau/outlineB/run_o3d.sh, un thread BLAS per processo, un worker per core); eval nel
# venv della GT unificata (cgt e i summarizer di E12). Partizione prioritized senza GPU: il nodo cpu e' in fail.
# TEMPI (misurati il 10 ottobre): NICP 2-4 s per coppia a un core (fino a 9 s su nodi carichi), Chamfer + ICP
# 0.1-0.2 s; fase veloce 5-12 min per (vista, modo) a 32 core. Il NICP va con blmm_multi.sbatch (launch.sh nicp).
set -euo pipefail

export AAU_NV=""
STEP="${BLMM_STEP:?BLMM_STEP: scalars | pairs | template | eval}"
W="${SLURM_CPUS_PER_TASK:-8}"
echo "[blmm] host=$(hostname) job=${SLURM_JOB_ID:-none} passo=$STEP args=${BLMM_ARGS:-}"
t0=$(date +%s)
case "$STEP" in
    scalars)  aau/outlineB/run_o3d.sh aau/baselines_mm/blmm_scalars.py --workers "$W" ${BLMM_ARGS:-} ;;
    pairs)    aau/outlineB/run_o3d.sh aau/baselines_mm/blmm_pairs.py ${BLMM_ARGS} --workers "$W" ;;
    template) aau/outlineB/run_o3d.sh aau/baselines_mm/blmm_template.py --workers "$W" ${BLMM_ARGS:-} ;;
    eval)     UGT_THREADS=1 v3_work/unified_gt/run.sh aau/baselines_mm/blmm_eval.py ${BLMM_ARGS:-} ;;
    *) echo "passo sconosciuto: $STEP" >&2; exit 2 ;;
esac
echo "[blmm] $STEP: $(( $(date +%s) - t0 )) s"
