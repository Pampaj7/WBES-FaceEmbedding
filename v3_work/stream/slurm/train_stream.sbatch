#!/usr/bin/env bash
#SBATCH --job-name=wbes-stream-train
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#
# P1 su un nodo, 1 GPU: produttori e trainer insieme (PLAN_MASSIVE sez. 8), misure in
# aau/runs/evidence/stream/train_<job>/:
#   1. producer.py in background (STREAM_P processi, k STREAM_K, anello su /tmp di STREAM_RING_GB GiB);
#   2. nvidia-smi ogni 2 s (gpu.csv) e memoria del job (mem.csv: cgroup e /tmp) ogni 10 s;
#   3. bench_consumer.py: tetto del lato dati per rank, senza GPU (consumer.json);
#   4. train_stream.py --stream per STREAM_STEPS passi: 16 identita' x 4 viste per passo, forward a gruppi, loss v2,
#      GT unificata al volo; eval online solo alla fine, sul BFM REMESH (serve solo a far girare il trainer intero).
# Altre GPU: sbatch --gres=gpu:a10:1 ... (stesso script).
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
source "$AAU_DIR/data_scale/recipe_v1.sh"
P="${STREAM_P:-48}"
K="${STREAM_K:-128}"
GB="${STREAM_RING_GB:-24}"
T="${STREAM_STEPS:-2000}"
S="${STREAM_STEPS_PER_EPOCH:-200}"
R="${STREAM_REUSE:-4}"
RING="/tmp/${SLURM_JOB_ID:-manual}/stream"
E="$AAU_RUNS/evidence/stream/train_${SLURM_JOB_ID:-manual}"
mkdir -p "$E" "$RING"
trap 'kill $(jobs -p) 2>/dev/null || true; rm -rf "/tmp/${SLURM_JOB_ID:-manual}"' EXIT
echo "[stream] host=$(hostname) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) nproc=$(nproc) P=$P k=$K anello=${GB}GiB T=$T R=$R"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# STREAM_TRAIN_CORES=N: i primi N core fisici (con i loro thread gemelli) al trainer, le altre CPU ai produttori
TRAIN_CPUS="" PROD_ARGS=()
if [[ -n "${STREAM_TRAIN_CORES:-}" ]]; then
  read -r TRAIN_CPUS PROD_CPUS < <(python3 - "$STREAM_TRAIN_CORES" <<'PY'
import os, sys
n, allowed = int(sys.argv[1]), sorted(os.sched_getaffinity(0))
def parse(s):
    out = set()
    for it in s.split(","):
        a, _, b = it.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out
cores, seen = [], set()
for c in allowed:
    if c in seen:
        continue
    try:
        g = parse(open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read().strip())
    except OSError:
        g = {c}
    g &= set(allowed)
    seen |= g
    cores.append(sorted(g))
train = sorted(x for g in cores[:n] for x in g)
print(",".join(map(str, train)), ",".join(map(str, sorted(set(allowed) - set(train)))))
PY
)
  PROD_ARGS=(--cpus "$PROD_CPUS")
  echo "[stream] CPU del trainer: $TRAIN_CPUS; dei produttori: $PROD_CPUS"
fi
TS=(); [[ -n "$TRAIN_CPUS" ]] && TS=(taskset -c "$TRAIN_CPUS")

AAU_NV= "$AAU_DIR/run.sh" v3_work/stream/producer.py --ring "$RING" --ring-gb "$GB" --n-proc "$P" --k-eig "$K" \
    --stats-every 60 --summary "$E/producers.json" "${PROD_ARGS[@]}" > "$E/producers.log" 2>&1 &
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 2 > "$E/gpu.csv" 2>/dev/null &
(
  cg=$(awk -F: '$2=="memory"||$2==""{print $3}' /proc/self/cgroup | head -1)
  while true; do
    cur=$(cat /sys/fs/cgroup/memory${cg%/step_*}/memory.usage_in_bytes 2>/dev/null \
          || cat /sys/fs/cgroup${cg%/step_*}/memory.current 2>/dev/null || echo nan)
    echo "$(date +%s),$cur,$(du -sb "$RING" 2>/dev/null | cut -f1)"
    sleep 10
  done
) > "$E/mem.csv" &
# l'anello deve contenere almeno 2 batch di identita' prima di partire (STREAM_MIN_SHARDS)
until [[ $(ls "$RING/shards" 2>/dev/null | wc -l) -ge ${STREAM_MIN_SHARDS:-16} ]]; do sleep 5; done
echo "[stream] anello pronto $(date +%T): $(ls "$RING/shards" | wc -l) shard"
"${TS[@]}" "$AAU_DIR/run.sh" v3_work/stream/bench_consumer.py --ring "$RING" --seconds 60 --batch 16 --views 4 \
    --reuse "$R" --out "$E/consumer.json"
# STREAM_FREEZE=1: produttori fermati prima del training (anello congelato, riuso senza tetto con STREAM_REUSE
# grande): il passo senza la contesa delle CPU dei produttori, per separare GPU e CPU.
if [[ "${STREAM_FREEZE:-0}" == 1 ]]; then
  kill %1 2>/dev/null || true; wait %1 2>/dev/null || true
  echo "[stream] produttori fermati $(date +%T): anello congelato, $(ls "$RING/shards" | wc -l) shard"
fi

CMD=(v3_work/stream/train_stream.py --stream "$RING" --stream-reuse "$R"
  --total-steps "$T" --steps-per-epoch "$S" --epochs $(( (T + S - 1) / S ))
  --data_dir datasets/REMESH/npz_data_topo_500_withops_areanorm --no-cache
  --dist_npz face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz
  "${RECIPE_V1[@]}" --batch_subjects 16 --max_meshes_per_subject_train 4 --max_subjects_eval_train 8
  --forward groups --fast-data --ema-decay 0.999 --train-threads 8 --eval_every 1000 --save_every 1000
  --seed 1234 --runs_root "$E/runs" --log-every 50 --ckpt-minutes 10
  --plateau-patience 8 --stream-batch-domains mixed)      # il comportamento di prima, ora esplicito (train_stream.py)
printf '%q ' "${CMD[@]}" > "$E/launch.txt"; echo >> "$E/launch.txt"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "${TS[@]}" "$AAU_DIR/run.sh" "${CMD[@]}" 2>&1 \
  | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v '%|' > "$E/train.log"
grep -E "passi eseguiti|^Best|epoca" "$E/train.log" | tail -20 || true
kill %1 2>/dev/null || true
wait %1 2>/dev/null || true
echo "[stream] OK"
