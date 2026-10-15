#!/usr/bin/env bash
#SBATCH --job-name=wbes-nc
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# NICP per le corrispondenze (aau/nicp_corr/nc.py): passi in sequenza nello stesso job, solo CPU.
#
#   NC_STEPS="run:hifi3d@mm merge:hifi3d@mm" sbatch nc.sbatch
#   NC_LIMIT=64 NC_STEPS="run:hifi3d@mm run:facescape@cs" sbatch nc.sbatch      (prova, <WORK>/pilot)
#   NC_STEPS="node:hifi3d@mm node:facescape@cs" sbatch --nodelist=<nodo> -c 32 nc.sbatch
#                                                 (nicp_p2tri della matrice sulla CPU d'origine: controllo; coppie di
#                                                 topologie fuori tolleranza, o NC_TOPO="noisy>up60k,remesh>up60k")
#   NC_STEPS="check:hifi3d check:facescape eval" sbatch nc.sbatch          (eval LEGGE LE GT di test)
# run e node nel venv della matrice (run_o3d.sh, come blmm_pairs); merge, check ed eval in quello dei protocolli.
set -euo pipefail
export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 UGT_THREADS=1
W="${SLURM_CPUS_PER_TASK:-8}"
D=aau/nicp_corr
UG=(v3_work/unified_gt/run.sh -u)
O3=(aau/outlineB/run_o3d.sh -u)
echo "[nc] host=$(hostname) job=${SLURM_JOB_ID:-none} passi=${NC_STEPS:?NC_STEPS} cpu=$W limite=${NC_LIMIT:-0} $(date +%T)"
for tok in $NC_STEPS; do
    step="${tok%%:*}"; view=""; mode=""; [[ "$tok" == *:* ]] && view="${tok#*:}"
    [[ "$view" == *@* ]] && { mode="${view#*@}"; view="${view%%@*}"; }
    t0=$(date +%s)
    echo "[nc] passo $step $view $mode $(date +%T)"
    case "$step" in
        run)     "${O3[@]}" $D/nc.py run --view "$view" --mode "$mode" --workers "$W" --limit "${NC_LIMIT:-0}" ;;
        merge)   "${UG[@]}" $D/nc.py merge --view "$view" --mode "$mode" ;;
        node)    "${O3[@]}" $D/nc.py check-node --view "$view" --mode "$mode" --topo "${NC_TOPO:-auto}" --workers "$W" ;;
        check)   "${UG[@]}" $D/nc.py check --view "$view" ;;
        eval)    "${UG[@]}" $D/nc.py eval --workers "$W" ;;
        results) "${UG[@]}" $D/nc.py results ;;
        *) echo "passo sconosciuto: $step" >&2; exit 2 ;;
    esac
    echo "[nc] OK $step $view $mode in $(( $(date +%s) - t0 ))s"
done
echo "[nc] OK tutti i passi $(date +%T)"
