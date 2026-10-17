#!/usr/bin/env bash
#SBATCH --job-name=wbes-fm
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#
# Matrice completa e confrontabile (aau/full_matrix/): un passo per job.
#
#   FM_STEP=varifold sbatch --gres=gpu:l40s:1 --time=01:00:00 aau/full_matrix/fm.sbatch   (crop, 4 viste)
#   FM_STEP=fit_crop    sbatch aau/full_matrix/fm.sbatch            (fit preregistrato sulle crop, venv open3d)
#   FM_STEP=spectral    sbatch aau/full_matrix/fm.sbatch            (ShapeDNA / HKS / WKS, FaceScape, FaceVerse, FaMoS)
#   FM_STEP=famos_tpl   sbatch aau/full_matrix/fm.sbatch            (NICP su template in mm sulle scansioni FaMoS TEST;
#                       FM_ARGS="--mode cs": la stessa in cs)
#   FM_STEP=famos_arc   sbatch aau/full_matrix/fm.sbatch            (ArcFace-render di G0 sulle scansioni FaMoS TEST)
#   FM_STEP=icp_same    sbatch --cpus-per-task=64 --mem=128G aau/full_matrix/fm.sbatch    (ICP fra soggetti, stessa topologia)
#   FM_STEP=eval        sbatch --cpus-per-task=64 --mem=192G --time=02:00:00 aau/full_matrix/fm.sbatch
#                       (righe, maschera comune, repliche, delta, CSV e results.md; LEGGE LE GT di test)
# Uscite in aau/runs/evidence/full_matrix/ (npz fuori da git). Ogni passo salta le uscite gia' su disco.
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
STEP="${FM_STEP:?FM_STEP: varifold | fit_crop | spectral | famos_tpl | famos_arc | icp_same | eval}"
W="${SLURM_CPUS_PER_TASK:-8}"
export AAU_NV=""
[[ "${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}" != "" ]] && export AAU_NV="--nv"
echo "[fm] host=$(hostname) job=${SLURM_JOB_ID:-none} passo=$STEP cpu=$W gpu=${CUDA_VISIBLE_DEVICES:-} $(date +%T)"
t0=$(date +%s)
case "$STEP" in
    varifold)
        for v in ${FM_VIEWS:-hifi3d facescape faceverse faceverse_neutral}; do
            aau/outlineB/run_o3d.sh -u aau/full_matrix/fm_varifold.py "$v" --device cuda --workers "$W"
        done ;;
    fit_crop)  aau/outlineB/run_o3d.sh -u aau/full_matrix/fm_compute.py fit-crop --workers "$W" ;;
    spectral)  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
                   aau/run.sh -u aau/full_matrix/fm_compute.py spectral --workers "$W" ;;
    famos_tpl) aau/outlineB/run_o3d.sh -u aau/full_matrix/fm_compute.py famos-template --workers "$W" ${FM_ARGS:-} ;;
    icp_same)  aau/outlineB/run_o3d.sh -u aau/full_matrix/fm_compute.py icp-same --workers "$W" ;;
    famos_arc)
        STAGE="/tmp/wbes_fm_arc_${SLURM_JOB_ID:-manual}"
        rm -rf "$STAGE"
        trap 'rm -rf "$STAGE"' EXIT
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
        aau/baselines/run_bl.sh -u aau/full_matrix/fm_arcface.py --stage "$STAGE" --workers "$W" ;;
    eval)      UGT_THREADS=1 v3_work/unified_gt/run.sh -u aau/full_matrix/fm_eval.py --workers "$W" ${FM_ARGS:-} ;;
    *) echo "passo sconosciuto: $STEP" >&2; exit 2 ;;
esac
echo "[fm] OK passo=$STEP in $(( $(date +%s) - t0 ))s"
