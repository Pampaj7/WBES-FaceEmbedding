#!/usr/bin/env bash
#SBATCH --job-name=wbes-stream-prod
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=64
#SBATCH --mem=150G
#SBATCH --time=01:10:00
#SBATCH --no-requeue
#
# Viste fresche/s dei produttori al variare di P (processi) e k (autovettori), solo CPU, su un nodo della
# famiglia del training (sottomettere con --nodelist=a768-l40s-0N). Ogni configurazione: anello vuoto,
# producer.py per STREAM_SECONDS, riepilogo in aau/runs/evidence/stream/producers/k<K>_P<P>.json (regime =
# dopo il primo shard di ogni processo).
#   STREAM_P="16 32 64" STREAM_K="64 128" sbatch --nodelist=a768-l40s-03 v3_work/stream/slurm/bench_producers.sbatch
set -euo pipefail
export AAU_NV=
PS="${STREAM_P:-16 32 64}"
KS="${STREAM_K:-64 128}"
SEC="${STREAM_SECONDS:-240}"
RING="/tmp/${SLURM_JOB_ID:-manual}/stream"
E=aau/runs/evidence/stream/producers
mkdir -p "$E"
echo "[bench] host=$(hostname) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) nproc=$(nproc) P=($PS) k=($KS) ${SEC}s"
for K in $KS; do
  for P in $PS; do
    rm -rf "$RING"
    aau/run.sh v3_work/stream/producer.py --ring "$RING" --ring-gb 24 --n-proc "$P" --k-eig "$K" \
        --duration "$SEC" --stats-every 60 --summary "$E/k${K}_P${P}.json"
    du -sh "$RING" || true
  done
done
rm -rf "$RING"
echo "[bench] OK"
