#!/usr/bin/env bash
#SBATCH --job-name=wbes-fs
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#
# Protocollo adattabile (aau/few_shot/, REGOLE.md in aau/runs/evidence/few_shot/): passi in sequenza nello stesso job,
# solo CPU (al massimo 4 job contemporanei: il tetto dei job e' condiviso).
#
#   FS_STEPS="now-prep:faceverse now-prep:faceverse_neutral" sbatch --cpus-per-task=32 --time=01:30:00 fs.sbatch
#   FS_STEPS="now-run:hifi3d@0/2" sbatch fs.sbatch              (parte 0 di 2; ripartibile, salta le mesh A salvate)
#   FS_STEPS="now-merge:hifi3d now-merge:facescape" sbatch --cpus-per-task=2 --mem=48G --time=00:30:00 fs.sbatch
#   FS_STEPS="selftest" sbatch --cpus-per-task=8 --mem=64G --time=00:30:00 fs.sbatch          (nessuna GT di test)
#   FS_STEPS="fit" sbatch --nodelist=i256-a10-10 fs.sbatch      (teste, nodo di D3 e2 / D5 / R6; ripartibile)
#   FS_STEPS="eval famos" sbatch --mem=192G fs.sbatch           (LEGGE LE GT di test)
#   FS_STEPS="fit wait-now eval famos" ...                      (wait-now: aspetta le matrici NoW delle 4 viste)
set -euo pipefail
export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
W="${SLURM_CPUS_PER_TASK:-8}"
D=aau/few_shot
RUN=(v3_work/unified_gt/run.sh -u)
echo "[fs] host=$(hostname) job=${SLURM_JOB_ID:-none} passi=${FS_STEPS:?FS_STEPS} cpu=$W $(date +%T)"
for tok in $FS_STEPS; do
    step="${tok%%:*}"; view=""; [[ "$tok" == *:* ]] && view="${tok#*:}"
    part="0/1"; [[ "$view" == *@* ]] && { part="${view#*@}"; view="${view%%@*}"; }
    t0=$(date +%s)
    echo "[fs] passo $step $view $part $(date +%T)"
    case "$step" in
        now-prep)
            "${RUN[@]}" $D/fs_now.py render --view "$view" --workers "$W"
            aau/recon/run_recon.sh mica aau/competitors/protocolli/pr_noise_detect.py --view "$view"
            "${RUN[@]}" $D/fs_now.py prep --view "$view" --workers "$W" ;;
        now-run)
            UGT_THREADS=1 "${RUN[@]}" $D/fs_now.py run --view "$view" --workers "$W" --shard "${part%/*}" \
                --nshards "${part#*/}" ${FS_REVERSE:+--reverse} ;;
        now-merge) "${RUN[@]}" $D/fs_now.py merge --view "$view" ;;
        wait-now)                                   # attesa delle matrici NoW di tutte le viste (job di continuazione)
            for v in hifi3d facescape faceverse faceverse_neutral; do
                until [[ -f aau/runs/few_shot/now_det/$v.npz ]]; do sleep 60; done
                echo "[fs] $v.npz presente $(date +%T)"
            done ;;
        selftest)  UGT_THREADS=1 "${RUN[@]}" $D/fs_eval.py selftest --workers "$W" ;;
        fit)       UGT_THREADS=1 "${RUN[@]}" $D/fs_head.py fit --workers "$W" ;;
        eval)      UGT_THREADS=1 "${RUN[@]}" $D/fs_eval.py eval --workers "$W" ;;
        famos)     UGT_THREADS=1 "${RUN[@]}" $D/fs_eval.py famos --workers "$W" ;;
        results)   "${RUN[@]}" $D/fs_eval.py results ;;
        *) echo "passo sconosciuto: $step"; exit 2 ;;
    esac
    echo "[fs] passo $step $view finito in $(( $(date +%s) - t0 ))s"
done
