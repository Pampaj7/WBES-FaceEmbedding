#!/usr/bin/env bash
#SBATCH --job-name=wbes-v3-keepstore-fact
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
# Custode dello store C3F ricostruito per i bracci fattorizzati (come keep_store.sbatch): l'epilogo di nv-ai-04
# cancella /tmp dell'utente quando finisce il suo ultimo job sul nodo. Resta RUNNING finche' esiste il file
# ${WBES_V3_HOLD:-aau/runs/evidence/trainer_v3/factorized/HOLD_STORE} (lo store e' pronto prima delle GT di E12)
# o un job dell'utente con nome wbes-v3-store-c3f, wbes-v3-c3f-fact* o wbes-v3-c3f-ctrlfr* e' in coda o in esecuzione.
set -uo pipefail
HOLD="${WBES_V3_HOLD:-/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/trainer_v3/factorized/HOLD_STORE}"
echo "[keepstore-fact] $(date +%F_%T) host=$(hostname) restart=${SLURM_RESTART_COUNT:-0} hold=$HOLD"
while :; do
  if out=$(squeue -h -u "$USER" -o %j 2>/dev/null); then
    left=$(grep -cE '^(wbes-v3-store-c3f|wbes-v3-c3f-fact|wbes-v3-c3f-ctrlfr)' <<< "$out")
    [[ "$left" -eq 0 && ! -e "$HOLD" ]] && break
  fi
  sleep 300
done
echo "[keepstore-fact] $(date +%F_%T) nessun job da attendere e nessun HOLD: esco, l'epilogo libera /tmp"
