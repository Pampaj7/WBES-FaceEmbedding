#!/usr/bin/env bash
#SBATCH --job-name=wbes-bp
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#
# Concorrenti parametrici e varifold (aau/baselines_param/bp.py; protocollo
# aau/runs/evidence/baselines_param/PROTOCOL.md). Un passo per job:
#
#   BP_STEP=fit BP_ARGS="--models gnm"                                   sbatch aau/baselines_param/bp.sbatch
#   BP_STEP=varifold BP_ARGS="hifi3d --device cuda" sbatch --gres=gpu:l40s:1 aau/baselines_param/bp.sbatch
#   BP_STEP=paired                                                       sbatch aau/baselines_param/bp.sbatch
#
# fit e varifold nel venv con open3d (aau/outlineB/run_o3d.sh, un thread BLAS per processo); paired nel venv della
# GT unificata (importa fact_paired e blmm_eval). Partizione prioritized senza GPU (il nodo cpu e' in fail), tranne il
# varifold delle viste a 500 mesh: una L40S (QoS normal).
# TEMPI (pilota del 10 ottobre, soggetti NON valutati): NICP 1.6-2.5 s per mesh (GNM, 4096 punti) e 0.4-0.9 s (FLAME),
# regressione 0.4 s (GNM) e 7-9 s (FLAME, ricerca della mandibola); varifold 7.8 ms per prodotto interno su L40S
# (blocco 8192), ~17 min per vista a 500 mesh; 1.2 s su un core.
set -euo pipefail

STEP="${BP_STEP:?BP_STEP: fit | varifold | paired}"
W="${SLURM_CPUS_PER_TASK:-8}"
export AAU_NV=""
[[ "${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}" != "" ]] && export AAU_NV="--nv"
echo "[bp] host=$(hostname) job=${SLURM_JOB_ID:-none} passo=$STEP args=${BP_ARGS:-}"
t0=$(date +%s)
case "$STEP" in
    fit)      aau/outlineB/run_o3d.sh aau/baselines_param/bp_fit.py --workers "$W" ${BP_ARGS:-} ;;
    varifold) aau/outlineB/run_o3d.sh aau/baselines_param/bp_varifold.py --block 8192 ${BP_ARGS} ;;
    paired)   UGT_THREADS=1 v3_work/unified_gt/run.sh aau/baselines_param/bp_paired.py --workers "$W" ${BP_ARGS:-} ;;
    *) echo "passo sconosciuto: $STEP" >&2; exit 2 ;;
esac
echo "[bp] $STEP: $(( $(date +%s) - t0 )) s"
