#!/usr/bin/env bash
#SBATCH --job-name=wbes-now-metrics
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# NoW, baseline senza GPU sulle patch di now_prepare_meshes.py:
#   1. now_geometric.py   chamfer_raw e icp_chamfer_mm, contro la scansione e fra ricostruzioni
#   2. now_arcface.py     render ombreggiati (calibrazione del crop), poi normal map
#
#   aau/submit.sh recon/now_metrics.sbatch
#   WBES_NOW_ARCFACE_OVERWRITE=1 aau/submit.sh recon/now_metrics.sbatch   # metodo aggiunto dopo:
#       camera unica e crop ArcFace vanno ricalcolati su TUTTE le mesh, render compresi
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

W="${SLURM_CPUS_PER_TASK:-8}"
OVERWRITE=(); [[ "${WBES_NOW_ARCFACE_OVERWRITE:-0}" == "1" ]] && OVERWRITE=(--overwrite)
WORK="${WBES_NOW_WORK:-$HOME/data/now_eval_work}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "[now-metrics] host=$(hostname) job=${SLURM_JOB_ID:-none} workers=$W"

"$AAU_DIR/run.sh" aau/recon/now_geometric.py --workers "$W"

"$AAU_DIR/baselines/run_bl.sh" aau/recon/now_arcface.py --mode shaded --workers "$W" "${OVERWRITE[@]}"
"$AAU_DIR/baselines/run_bl.sh" aau/recon/now_arcface.py --mode normals --workers "$W" "${OVERWRITE[@]}" \
    --calibration-from "$WORK/arcface/shaded/renders/arcface_align.json"

echo "[now-metrics] OK"
