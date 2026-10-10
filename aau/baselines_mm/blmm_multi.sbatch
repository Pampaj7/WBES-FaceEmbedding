#!/usr/bin/env bash
#SBATCH --job-name=wbes-blmm-multi
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --ntasks=15
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3G
#SBATCH --time=03:00:00
#
# NICP per coppia (blmm_pairs.py, passo nicp) di piu' (vista, modo) in UN job: ogni task e' un processo
# con 16 worker, i task si spargono sui nodi liberi. Perche' un job solo: la QoS normal
# ha MaxJobsPU = 12 per utente, condiviso con gli altri lavori; 15 task x 16 core = 240 core con un posto.
# Viste a 6 topologie: coda condivisa delle 30 coppie ordinate (blmm_pairs.py --claim); FaMoS: 1/15 delle righe per task.
#
#   BLMM_JOBS="hifi3d:cs facescape:mm" sbatch aau/baselines_mm/blmm_multi.sbatch
#
# Ripartibile: le coppie di topologie (o gli shard FaMoS) gia' su disco si saltano.
set -euo pipefail

export AAU_NV=""
JOBS="${BLMM_JOBS:?BLMM_JOBS: vista:modo ...}"
echo "[blmm-multi] job=${SLURM_JOB_ID:-none} task=${SLURM_NTASKS} nodi=${SLURM_JOB_NODELIST} jobs=$JOBS"
for vm in $JOBS; do
    t0=$(date +%s)
    export BLMM_V="${vm%%:*}" BLMM_M="${vm##*:}"
    # viste a 6 topologie: coda condivisa (--claim, ogni task prende la prossima coppia di topologie libera);
    # FaMoS: shard fissi delle righe
    if [[ "$BLMM_V" == famos ]]; then export BLMM_SPLIT='--shard $SLURM_PROCID/$SLURM_NTASKS'; else export BLMM_SPLIT=--claim; fi
    srun --ntasks="$SLURM_NTASKS" --cpus-per-task="$SLURM_CPUS_PER_TASK" --kill-on-bad-exit=0 bash -c \
        'aau/outlineB/run_o3d.sh aau/baselines_mm/blmm_pairs.py "$BLMM_V" "$BLMM_M" nicp \
             $(eval echo $BLMM_SPLIT) --workers "$SLURM_CPUS_PER_TASK" --chunk 4' || \
        echo "[blmm-multi] $vm: almeno un task e' fallito (vedi .err)"
    echo "[blmm-multi] $vm: $(( $(date +%s) - t0 )) s"
done
