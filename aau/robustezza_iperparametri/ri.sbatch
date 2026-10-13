#!/usr/bin/env bash
#SBATCH --job-name=wbes-ri
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#
# Robustezza agli iperparametri (aau/robustezza_iperparametri/ri.py): passi in sequenza nello stesso job, solo CPU.
#
#   RI_STEPS="run:hifi3d@icp merge:hifi3d@icp" sbatch ri.sbatch
#   RI_STEPS="run:famos@now merge:famos@now" sbatch ri.sbatch
#   RI_LIMIT=64 RI_STEPS="run:hifi3d@nicp" sbatch ri.sbatch            (pilota dei tempi, <WORK>/pilot)
#   RI_STEPS="nownode:hifi3d nownode:facescape" sbatch --exclude=<tutti tranne a256-t4-0[1-4]> -c 12 ri.sbatch
#                                                 (NoW di default sul tipo di nodo della matrice: controllo)
#   RI_STEPS="check:hifi3d check:facescape check:famos eval" sbatch ri.sbatch   (eval LEGGE LE GT di test)
# ICP e NICP nel venv della matrice (run_o3d.sh, come blmm_pairs); NoW, merge, check ed eval in quello dei protocolli.
set -euo pipefail
export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 UGT_THREADS=1
W="${SLURM_CPUS_PER_TASK:-8}"
D=aau/robustezza_iperparametri
UG=(v3_work/unified_gt/run.sh -u)
O3=(aau/outlineB/run_o3d.sh -u)
echo "[ri] host=$(hostname) job=${SLURM_JOB_ID:-none} passi=${RI_STEPS:?RI_STEPS} cpu=$W limite=${RI_LIMIT:-0} $(date +%T)"
for tok in $RI_STEPS; do
    step="${tok%%:*}"; view=""; meth=""; [[ "$tok" == *:* ]] && view="${tok#*:}"
    [[ "$view" == *@* ]] && { meth="${view#*@}"; view="${view%%@*}"; }
    t0=$(date +%s)
    echo "[ri] passo $step $view $meth $(date +%T)"
    case "$step" in
        run)
            if [[ "$meth" == now ]]; then R=("${UG[@]}"); else R=("${O3[@]}"); fi
            "${R[@]}" $D/ri.py run --view "$view" --method "$meth" --workers "$W" --limit "${RI_LIMIT:-0}" ;;
        merge)   "${UG[@]}" $D/ri.py merge --view "$view" --method "$meth" ;;
        check)   "${UG[@]}" $D/ri.py check --view "$view" ;;
        nownode) "${UG[@]}" $D/ri.py check-now-node --view "$view" --workers "$W" ;;
        eval)    "${UG[@]}" $D/ri.py eval --workers "$W" ;;
        results) "${UG[@]}" $D/ri.py results ;;
        *) echo "passo sconosciuto: $step" >&2; exit 2 ;;
    esac
    echo "[ri] OK $step $view $meth in $(( $(date +%s) - t0 ))s"
done
echo "[ri] OK tutti i passi $(date +%T)"
