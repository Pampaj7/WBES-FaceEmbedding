#!/usr/bin/env bash
#SBATCH --job-name=wbes-ulr
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#
# ULRSSM riaddestrato sui generatori di training (aau/competitors/ulrssm/): un passo per job.
#
#   ULR_STEP=data   sbatch aau/competitors/ulrssm/ulr.sbatch                                  (mesh di training, CPU)
#   ULR_STEP=train  ULR_INPUT=xyz sbatch --gres=gpu:l40s:1 --mem=160G --time=24:00:00 aau/competitors/ulrssm/ulr.sbatch
#   ULR_STEP=smoke  ULR_CKPT=<pth> sbatch --gres=gpu:l40s:1 --mem=96G --time=00:40:00 aau/competitors/ulrssm/ulr.sbatch
#   ULR_STEP=eval   ULR_VIEW=hifi3d ULR_TAG=faces_xyz ULR_CKPT=<pth> [ULR_SHARD=0/1] [ULR_SPEC=<json>] \
#                   sbatch --gres=gpu:l40s:1 --mem=160G --time=06:00:00 aau/competitors/ulrssm/ulr.sbatch
#   ULR_STEP=merge  ULR_VIEW=hifi3d ULR_TAG=faces_xyz sbatch aau/competitors/ulrssm/ulr.sbatch       (venv della GT)
#   ULR_STEP=select sbatch aau/competitors/ulrssm/ulr.sbatch
#   ULR_STEP=paired ULR_TAG=faces_xyz sbatch --mem=192G --time=02:00:00 aau/competitors/ulrssm/ulr.sbatch   (LEGGE LE GT)
# Operatori di ULRSSM nella cache su /tmp del nodo (conta contro --mem: FaceVerse ~20 GB), cancellata a fine job.
# ULRSSM clonato in ULRSSM_DIR (default ~/ext/ulrssm), mai nel repo.
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
STEP="${ULR_STEP:?ULR_STEP: data | train | smoke | eval | merge | select | paired}"
W="${SLURM_CPUS_PER_TASK:-8}"
D=aau/competitors/ulrssm
export AAU_NV=""
[[ "${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}" != "" ]] && export AAU_NV="--nv"
CACHE="/tmp/wbes_ulr_${SLURM_JOB_ID:-manual}"
trap 'rm -rf "$CACHE"' EXIT
echo "[ulr] host=$(hostname) job=${SLURM_JOB_ID:-none} passo=$STEP cpu=$W gpu=${CUDA_VISIBLE_DEVICES:-} $(date +%T)"
t0=$(date +%s)
SPEC=()
[[ -n "${ULR_SPEC:-}" ]] && SPEC=(--spec "$ULR_SPEC")
case "$STEP" in
    data)   aau/run.sh -u $D/ulr_data.py --workers "$W" ${ULR_ARGS:-} ;;
    train)  aau/run.sh -u $D/ulr_train.py --input "${ULR_INPUT:?ULR_INPUT: xyz | wks}" --cache "$CACHE" \
                --workers "$W" ${ULR_ARGS:-} ;;
    smoke)  aau/run.sh -u $D/ulr_eval.py smoke --ckpt "${ULR_CKPT:?}" --cache "$CACHE" ;;
    eval)   aau/run.sh -u $D/ulr_eval.py run --view "${ULR_VIEW:?}" --tag "${ULR_TAG:?}" --ckpt "${ULR_CKPT:?}" \
                --cache "$CACHE" --workers "$W" --shard "${ULR_SHARD:-0/1}" "${SPEC[@]}" ${ULR_ARGS:-} ;;
    merge)  v3_work/unified_gt/run.sh -u $D/ulr_eval.py merge --view "${ULR_VIEW:?}" --tag "${ULR_TAG:?}" "${SPEC[@]}" ;;
    select) aau/run.sh -u $D/ulr_eval.py select ;;
    paired) UGT_THREADS=1 v3_work/unified_gt/run.sh -u $D/ulr_paired.py --tag "${ULR_TAG:?}" --workers "$W" ${ULR_ARGS:-} ;;
    *) echo "passo sconosciuto: $STEP" >&2; exit 2 ;;
esac
echo "[ulr] OK passo=$STEP in $(( $(date +%s) - t0 ))s"
