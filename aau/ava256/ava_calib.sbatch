#!/usr/bin/env bash
#SBATCH --job-name=wbes-ava256-calib
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#
# Ava-256, test CONFERMATIVO: GT dei 56 di calibrazione e controllo delle viste di calibrazione (ava_calib.py), su CPU.
# Nessun metodo, nessuna distanza di un metodo, nessuno Spearman.
#
#   AVA_CALIB_STEPS="dry views" sbatch aau/ava256/ava_calib.sbatch    # controlli, nulla di scritto nei dati
#   AVA_CALIB_STEPS="gt verify" sbatch aau/ava256/ava_calib.sbatch    # scrive e congela la GT di calibrazione
#
#   dry     ava_calib.py gt --dry-run   procedura contro le GT congelate, impronte della GT di calibrazione
#   views   ava_calib.py views          calib_view completa e coerente con eval_view -> calib_view_check.json
#   gt      ava_calib.py gt             datasets/AVA256/calib_gt, calib_manifest.json, sola lettura
#   verify  ava_calib.py verify         verify_frozen() + verify_calib()
set -euo pipefail

if [[ -z "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}${GPU_DEVICE_ORDINAL:-}" ]]; then
    export AAU_NV=""
fi
STEPS="${AVA_CALIB_STEPS:-dry views}"
UGT="v3_work/unified_gt/run.sh"
export UGT_THREADS="${UGT_THREADS:-8}"
echo "[ava256-calib] host=$(hostname) job=${SLURM_JOB_ID:-none} steps: $STEPS"
for step in $STEPS; do
    t0=$(date +%s)
    case "$step" in
        dry)    $UGT aau/ava256/ava_calib.py gt --dry-run ;;
        views)  $UGT aau/ava256/ava_calib.py views ;;
        gt)     $UGT aau/ava256/ava_calib.py gt ;;
        verify) $UGT aau/ava256/ava_calib.py verify ;;
        *) echo "passo sconosciuto: $step" >&2; exit 2 ;;
    esac
    echo "[ava256-calib] $step: $(( $(date +%s) - t0 )) s"
done
echo "[ava256-calib] OK"
