#!/usr/bin/env bash
#SBATCH --job-name=wbes-v3-keepstore
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --partition=aicentre-a100
#SBATCH --qos=unprivileged
#SBATCH --requeue
#SBATCH --nodelist=nv-ai-04
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=4-00:00:00
#
# Custode dello store in /tmp di nv-ai-04. L'epilogo Slurm del nodo (epilog.d/42-lastuserjob-cleanup) cancella
# TUTTI i file dell'utente in /tmp e /dev/shm quando finisce il suo ultimo job RUNNING sul nodo: senza questo
# job lo store sparirebbe fra la fine di build_store e l'avvio dei bracci (afterok), o se tutti i bracci
# venissero prelazionati insieme. Resta RUNNING finche' uno dei job in WBES_V3_KEEP_JOBS e' in coda o in
# esecuzione, poi esce e l'epilogo libera il disco. (8 ottobre: e' cosi' che si sono persi i primi operatori.)
set -uo pipefail
JOBS="${WBES_V3_KEEP_JOBS:?id dei job da attendere, separati da virgola}"
echo "[keepstore] $(date +%F_%T) host=$(hostname) restart=${SLURM_RESTART_COUNT:-0} attende: $JOBS"
while :; do
  # squeue puo' fallire per un attimo (controller occupato): si esce solo su una lettura riuscita e senza job
  if out=$(squeue -h -u "$USER" -o %i 2>/dev/null); then
    left=$(grep -cwF -f <(tr ',' '\n' <<< "$JOBS") <<< "$out")
    [ "$left" -eq 0 ] && break
  fi
  sleep 300
done
echo "[keepstore] $(date +%F_%T) nessun job da attendere: esco, l'epilogo libera /tmp"
