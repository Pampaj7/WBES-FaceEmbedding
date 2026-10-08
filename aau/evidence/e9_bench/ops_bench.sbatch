#!/usr/bin/env bash
#SBATCH --job-name=wbes-e9-ops
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --exclude=a256-a40-[04-07],a256-t4-[01-04],a512-l4-06,i256-a10-[06-10],nv-ai-[02-03]
#SBATCH --cpus-per-task=100
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#
# E9 (a) e (d) sulle CPU di un nodo L40S, SENZA GPU (la partizione non la richiede: nessun
# job_submit plugin). --exclude lascia solo a768-l40s-[01-06]. 100 CPU logiche = 50 core con
# hyperthreading: e' la quota di produttori di P1 (PLAN_MASSIVE.md sez. 8, ~100 CPU logiche).
#
#   1. grad_vec.py: build_grad vettorizzato contro l'originale su 20 mesh + embedding e108 (~3 min)
#   2. ops_bench.py: le configurazioni di (a) e (d), una dopo l'altra (25 min misurati, job 1061626)
# Follow-up su poche configurazioni, senza il punto 1:
#   E9_SKIP_GRADCHECK=1 E9_OPS_ARGS="--only nome,nome --tag _f2" sbatch --time=00:30:00 ops_bench.sbatch
#
# Memoria [smoke 1061620]: worker 0.6-0.84 GB di RSS ciascuno -> ~85 GB a P=100, piu' le uscite
# tenute per le dimensioni (~8 GB su /tmp, che conta contro --mem).
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export AAU_NV=""
OUT="$AAU_RUNS/evidence/e9"
WORK="/tmp/${SLURM_JOB_ID:-manual}/e9"
mkdir -p "$OUT/ops" "$WORK"
trap 'rm -rf "/tmp/${SLURM_JOB_ID:-manual}"' EXIT

{ echo "host=$(hostname) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?} date=$(date -Is)"
  lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket|MHz"
  echo "squeue del nodo all'avvio:"; squeue -w "$(hostname -s)" -h -o "%i %u %C %b" || true
} > "$OUT/ops/node_${SLURM_JOB_ID:-manual}.txt"

# memoria del cgroup del job (dove Slurm applica --mem), come data_scale/train_scale.sbatch
JOBCG="/sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_${SLURM_JOB_ID:-none}"
( while [[ -r "$JOBCG/memory.usage_in_bytes" ]]; do
    echo "$(date +%T) usage=$(( $(cat $JOBCG/memory.usage_in_bytes) >> 30 )) peak=$(( $(cat $JOBCG/memory.max_usage_in_bytes) >> 30 )) GiB" \
      >> "$OUT/ops/mem_${SLURM_JOB_ID:-manual}.log"
    sleep 20
  done ) &

if [[ -z "${E9_SKIP_GRADCHECK:-}" ]]; then
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$AAU_DIR/run.sh" aau/evidence/e9_bench/grad_vec.py \
        --work-dir "$WORK/gradcheck" --out-json "$OUT/grad_check.json"
    rm -rf "$WORK/gradcheck"
fi

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$AAU_DIR/run.sh" \
    aau/evidence/e9_bench/ops_bench.py --work-dir "$WORK/ops" --out-dir "$OUT/ops" ${E9_OPS_ARGS:-}
echo "[e9-ops] OK"
