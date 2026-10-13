#!/usr/bin/env bash
#SBATCH --job-name=wbes-prn
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#
# Protocolli con landmark stimati (aau/competitors/protocolli/pr_noise.py): passi in sequenza nello stesso job, solo CPU.
#
#   PRN_STEPS="render:hifi3d detect:hifi3d prep:hifi3d@det check:hifi3d" sbatch pr_noise.sbatch
#   PRN_STEPS="prep:hifi3d@s2 run:hifi3d@s2 merge:hifi3d@s2" sbatch pr_noise.sbatch
#   PRN_STEPS="eval" sbatch pr_noise.sbatch                     (Spearman con IC; LEGGE LE GT di test)
# PRN_REVERSE=1: run dalle ultime mesh A (job di riserva sulla stessa condizione).
set -euo pipefail
export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
W="${SLURM_CPUS_PER_TASK:-8}"
D=aau/competitors/protocolli
RUN=(v3_work/unified_gt/run.sh -u)
echo "[prn] host=$(hostname) job=${SLURM_JOB_ID:-none} passi=${PRN_STEPS:?PRN_STEPS} cpu=$W $(date +%T)"
for tok in $PRN_STEPS; do
    step="${tok%%:*}"; view=""; cond=""; [[ "$tok" == *:* ]] && view="${tok#*:}"
    [[ "$view" == *@* ]] && { cond="${view#*@}"; view="${view%%@*}"; }
    t0=$(date +%s)
    echo "[prn] passo $step $view $cond $(date +%T)"
    case "$step" in
        render) "${RUN[@]}" $D/pr_noise.py render --view "$view" --workers "$W" ;;
        detect) OMP_NUM_THREADS="$W" aau/recon/run_recon.sh mica -u $D/pr_noise_detect.py --view "$view" ;;
        prep)   "${RUN[@]}" $D/pr_noise.py prep --view "$view" --cond "$cond" --workers "$W" ;;
        run)    "${RUN[@]}" $D/pr_noise.py run --view "$view" --cond "$cond" --workers "$W" ${PRN_REVERSE:+--reverse} ;;
        merge)  "${RUN[@]}" $D/pr_noise.py merge --view "$view" --cond "$cond" ;;
        check)  "${RUN[@]}" $D/pr_noise.py check --view "$view" --workers "$W" ;;
        eval)   UGT_THREADS=1 "${RUN[@]}" $D/pr_noise.py eval --workers "$W" ;;
        *) echo "passo sconosciuto: $step" >&2; exit 2 ;;
    esac
    echo "[prn] OK $step $view $cond in $(( $(date +%s) - t0 ))s"
done
echo "[prn] OK tutti i passi $(date +%T)"
