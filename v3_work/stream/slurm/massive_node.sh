#!/usr/bin/env bash
# Corpo PER NODO di massive.sbatch (uno srun per nodo; variabili STREAM_* documentate li').
#   1. CPU del nodo: i primi STREAM_TRAIN_CPUS_PER_RANK x GPU core (coi gemelli SMT) ai rank, le altre ai produttori;
#   2. producer.py --provenance --canonical-gt --expr-frac sulle fonti STREAM_SOURCES, anello in /tmp (RAM, conta
#      contro la memoria del job) da STREAM_RING_GB_PER_GPU x GPU GiB, semi distinti per nodo e per riavvio;
#   3. nvidia-smi (gpu.csv) e memoria del job (mem.csv), in $STREAM_OUT/node<N>/;
#   4. torchrun (rendezvous c10d su $STREAM_MASTER:$STREAM_PORT, GPU del nodo come rank locali) di train_stream.py con la
#      testa di STREAM_ARM, GT di E12 al volo, ingresso globale, registro delle viste usate; eval online sul BFM
#      REMESH (solo per far girare il trainer intero e scegliere i checkpoint come sempre).
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
source "$AAU_DIR/data_scale/recipe_v1.sh"
read -r NODE NN < <(python3 -c "import socket, sys; n = sys.argv[1].split(); h = socket.gethostname().split('.')[0]; \
print(next(i for i, x in enumerate(n) if x.split('.')[0] == h), len(n))" "${STREAM_NODES:-$(hostname -s)}")
RST="${SLURM_RESTART_COUNT:-0}"
G=$(nvidia-smi -L | wc -l)
ARM="${STREAM_ARM:-factorized}"
SOURCES="${STREAM_SOURCES:-massive}"
K="${STREAM_K:-128}"
T="${STREAM_STEPS:-2000}"
S="${STREAM_SPE:-200}"
R="${STREAM_REUSE:-4}"
EF="${STREAM_EXPR_FRAC:-0.5}"
MMAUG="${STREAM_MM_AUG-hybrid=0.3,expr_transfer=0.15,rbf=0.15}"    # vuoto = solo identita' pure
CPR="${STREAM_TRAIN_CPUS_PER_RANK:-4}"
RGB=$(( ${STREAM_RING_GB_PER_GPU:-10} * G ))
SEED="${STREAM_SEED:-1234}"
E="$STREAM_OUT/node$NODE"
mkdir -p "$E"
ln -sfn ../runs "$E/runs"
JOBTMP="/tmp/${SLURM_JOB_ID:-manual}_stream"
RING="$JOBTMP/r$RST"
rm -rf "$JOBTMP"
mkdir -p "$RING"
trap 'kill $(jobs -p) 2>/dev/null || true; rm -rf "$JOBTMP"' EXIT
log() { echo "[node$NODE] $(date +%F_%T) $*" | tee -a "$E/node.log"; }
log "host=$(hostname) GPU=$G ($(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | xargs)) CPU=$(nproc)" \
    "riavvio=$RST arm=$ARM fonti=$SOURCES k=$K T=$T S=$S R=$R expr=$EF mm_aug=${MMAUG:-no} anello=${RGB}GiB"

# --- CPU: rank e produttori ------------------------------------------------------------------------------
read -r TRAIN_CPUS PROD_CPUS < <(python3 - $(( G * CPR )) <<'PY'
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
train, k = [], 0
while len(train) < n and k < len(cores):
    train += cores[k]
    k += 1
print(",".join(map(str, sorted(train))), ",".join(map(str, sorted(set(allowed) - set(train)))))
PY
)
NPROD=$(python3 -c "import sys; print(len([x for it in sys.argv[1].split(',') for x in range(int(it.split('-')[0]), int(it.split('-')[-1]) + 1)]))" "$PROD_CPUS")
log "CPU dei rank: $TRAIN_CPUS; produttori: $NPROD processi"

# --- produttori ------------------------------------------------------------------------------------------
PSEED=$(( 20261009 + 1000 * NODE + 100000 * RST ))
AAU_NV= "$AAU_DIR/run.sh" v3_work/stream/producer.py --ring "$RING" --ring-gb "$RGB" --n-proc "$NPROD" --k-eig "$K" \
    --evecs-dtype fp32 --sources "$SOURCES" --provenance --canonical-gt --expr-frac "$EF" --seed "$PSEED" \
    ${MMAUG:+--mm-aug "$MMAUG"} \
    --cpus "$PROD_CPUS" --stats-every 60 --summary "$E/producers.json" > "$E/producers.log" 2>&1 &
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 5 >> "$E/gpu.csv" 2>/dev/null &
(
  cg=$(awk -F: '$2=="memory"||$2==""{print $3}' /proc/self/cgroup | head -1)
  while true; do
    cur=$(cat /sys/fs/cgroup/memory${cg%/step_*}/memory.usage_in_bytes 2>/dev/null \
          || cat /sys/fs/cgroup${cg%/step_*}/memory.current 2>/dev/null || echo nan)
    echo "$(date +%s),$cur,$(du -sb "$RING" 2>/dev/null | cut -f1)"
    sleep 30
  done
) >> "$E/mem.csv" &
MIN=$(( G * 8 > 16 ? G * 8 : 16 ))
until [[ $(ls "$RING/shards" 2>/dev/null | wc -l) -ge $MIN ]]; do
  sleep 5
  kill -0 %1 2>/dev/null || { log "ERRORE: produttori morti"; tail -20 "$E/producers.log"; exit 3; }
done
log "anello pronto: $(ls "$RING/shards" | wc -l) shard"

# --- trainer ---------------------------------------------------------------------------------------------
C3F="$AAU_RUNS/evidence/trainer_v3/ablations/c3f"
CGT="$WBES_ROOT/datasets/CANONICAL_GT/train"
ARMF=(--area robust --area-robust smooth --dropout 0.0 --gt-keep-scale --input-norm global
      --scale-table "$AAU_RUNS/evidence/stream/scale_remesh500.npz" --stream-scale 0)
case "$ARM" in
  factorized|factorized2)
    KAPPA=$(python3 -c "import json; print(repr(json.load(open('$C3F/fr_params.json'))['kappa']))")
    ARMF+=(--head "$ARM" --size-table "$CGT/centroid_size_bfm_ict_gnm.npz" --scale-aug 0.8,1.25
           --size-mask-domains bfm --gt-scale "$KAPPA" --stream-gt sr --dist_npz "$C3F/gt_sr.npz") ;;
  ctrlfr)
    ARMF+=(--stream-gt fr --dist_npz "$C3F/gt_frcal.npz") ;;
  unified)
    ARMF=(--area robust --area-robust smooth --dropout 0.0 --stream-gt unified
          --dist_npz face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz) ;;
  *) log "ERRORE: STREAM_ARM=$ARM"; exit 2 ;;
esac
EXTRA=()
[[ -n "${STREAM_EXTRA:-}" ]] && EXTRA+=(--stream-extra "$STREAM_EXTRA" --stream-extra-mirror "$JOBTMP/mirror")
EP=$(( (T + S - 1) / S ))
CMD=(v3_work/stream/train_stream.py --stream "$RING" --stream-reuse "$R" --stream-sources "$SOURCES" --stream-log-views
  --stream-wait-s 1800 --total-steps "$T" --steps-per-epoch "$S" --epochs "$EP"
  --data_dir datasets/REMESH/npz_data_topo_500_withops_areanorm --no-cache
  "${RECIPE_V1[@]}" --batch_subjects 16 --max_meshes_per_subject_train 4 --max_subjects_eval_train 8
  --forward sequential --fast-data --ema-decay 0.999 --train-threads "$CPR"
  --eval_every "${STREAM_EVAL_EVERY:-$(( EP > 10 ? EP / 10 : 1 ))}" --save_every "${STREAM_SAVE_EVERY:-$EP}"
  --seed "$SEED" --runs_root "$STREAM_OUT/runs" --log-every 50 --ckpt-minutes "${STREAM_CKPT_MIN:-20}"
  "${ARMF[@]}" "${EXTRA[@]}")
printf '%q ' "${CMD[@]}" > "$E/launch.txt"; echo >> "$E/launch.txt"
export WBES_STACK_DUMP_S="${WBES_STACK_DUMP_S:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
[[ -n "${STREAM_NCCL_IB_HCA:-}" ]] && export NCCL_IB_HCA="$STREAM_NCCL_IB_HCA"
# Ripresa dopo un crash: un torchrun NUOVO per tentativo (--max-restarts 0, rendezvous e porta nuovi), che riparte
# da last.pth (--resume auto). Il riavvio elastico DENTRO lo stesso torchrun su piu' nodi si blocca nella prima
# collettiva di NCCL (misurato il 9 ottobre, 1 + 1 GPU): per questo non si usa.
ATT=0
while :; do
  XC=()
  [[ -n "${STREAM_TEST_CRASH:-}" && "$ATT" == 0 && "$RST" == 0 ]] && XC=(--test-crash-at-step "$STREAM_TEST_CRASH")
  log "torchrun tentativo $ATT: $NN nodi, $G rank qui, rendezvous $STREAM_MASTER:$(( STREAM_PORT + ATT ))"
  set +e
  OMP_NUM_THREADS="$CPR" MKL_NUM_THREADS="$CPR" taskset -c "$TRAIN_CPUS" "$AAU_DIR/run.sh" -m torch.distributed.run \
    --nnodes "$NN" --nproc-per-node "$G" --rdzv-backend c10d --rdzv-endpoint "$STREAM_MASTER:$(( STREAM_PORT + ATT ))" \
    --rdzv-id "${SLURM_JOB_ID:-manual}_${RST}_$ATT" --max-restarts 0 "${CMD[@]}" "${XC[@]}" 2>&1 \
    | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v '%|\|FloatTensor\|SparseTensor' >> "$E/train.log"
  rc=${PIPESTATUS[0]}
  set -e
  log "trainer finito rc=$rc (tentativo $ATT)"
  [[ "$rc" == 0 ]] && break
  ATT=$(( ATT + 1 ))
  [[ "$ATT" -gt "${STREAM_ATTEMPTS:-3}" ]] && break
  sleep 30
done
kill %1 2>/dev/null || true
wait %1 2>/dev/null || true
[[ "$NODE" == 0 ]] && python3 v3_work/stream/summarize_run.py "$E" | tee -a "$E/node.log" || true
exit "$rc"
