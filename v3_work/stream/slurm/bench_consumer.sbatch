#!/usr/bin/env bash
#SBATCH --job-name=wbes-stream-cons
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --no-requeue
#
# Tetto del lato dati per rank (bench_consumer.py, --fast-data come il trainer) con un budget di CPU da nodo pieno:
# l'anello si riempie (produttori su tutte le CPU), poi i produttori si fermano e il consumatore gira ristretto a
# 2, 4 e 8 CPU logiche con 4 e 8 thread di prefetch. Uscite: aau/runs/evidence/stream/consumer/c<CPU>_t<thread>.json
set -euo pipefail
export AAU_NV=
RING="/tmp/${SLURM_JOB_ID:-manual}/stream"
E=aau/runs/evidence/stream/consumer
mkdir -p "$E"
trap 'kill $(jobs -p) 2>/dev/null || true; rm -rf "/tmp/${SLURM_JOB_ID:-manual}"' EXIT
aau/run.sh v3_work/stream/producer.py --ring "$RING" --ring-gb 16 --n-proc 60 --k-eig 128 --stats-every 60 \
    > "$E/producers.log" 2>&1 &
until [[ $(ls "$RING/shards" 2>/dev/null | wc -l) -ge 600 ]]; do sleep 5; done
kill %1; wait %1 || true
echo "[cons] host=$(hostname) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) anello $(ls "$RING/shards" | wc -l) shard"
for C in 2 4 8; do
  for T in 4 8; do
    CPUS=$(python3 -c "import os; print(','.join(map(str, sorted(os.sched_getaffinity(0))[:$C])))")
    taskset -c "$CPUS" aau/run.sh v3_work/stream/bench_consumer.py --ring "$RING" --seconds 45 --batch 16 \
        --views 4 --reuse 1000000 --prefetch "$T" --fast-data --out "$E/c${C}_t${T}.json"
  done
done
