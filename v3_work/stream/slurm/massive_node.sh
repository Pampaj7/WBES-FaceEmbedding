#!/usr/bin/env bash
# Corpo PER NODO di massive.sbatch (uno srun per nodo; variabili STREAM_* documentate li').
#   1. CPU del nodo: i primi STREAM_TRAIN_CPUS_PER_RANK x GPU core (coi gemelli SMT) ai rank, le altre ai produttori;
#   2. producer.py --provenance --canonical-gt --expr-frac --label-draw sulle fonti STREAM_SOURCES, anello in /tmp
#      (RAM, conta contro la memoria del job) da STREAM_RING_GB_PER_GPU x GPU GiB, semi distinti per nodo e segmento
#      (STREAM_SEGMENT: partenze precedenti del run, riavvii e continuazioni, massive.sbatch);
#   3. nvidia-smi (gpu.csv) e memoria del job (mem.csv), in $STREAM_OUT/node<N>/;
#   4. torchrun (rendezvous c10d su $STREAM_MASTER:$STREAM_PORT, GPU del nodo come rank locali) di train_stream.py con la
#      testa di STREAM_ARM, GT di E12 al volo, ingresso globale, registro delle viste usate; eval online sul BFM
#      REMESH (solo per far girare il trainer intero e scegliere i checkpoint come sempre).
# Ricetta dei bracci decisivi C3F/C3M (factorized/c3m/launch.txt), cambiato solo cio' che serve per scalare (dati dallo
# stream, rank, passi); le differenze residue sono variabili esplicite, scritte in node.log:
#   lr 1e-4 costante (--lr-constant; il C3M ha --lr-steps 81747:5e-5, oltre la fine); batch a dominio singolo
#   (--stream-batch-domains single = --sampler balanced --domain-alpha 0); STREAM_IDS_PER_RANK x STREAM_MESHES_PER_ID
#   = 5 x <= 6 per rank come il C3M (a 12 rank il batch globale e' 60 identita', il doppio dei 6 rank del C3M), con
#   STREAM_VIEWS (= STREAM_MESHES_PER_ID) viste per identita' dai produttori.
# STREAM_RECIPE sceglie i default dei dati (ognuno sovrascrivibile dalla sua variabile):
#   c3m (default con STREAM_SOURCES=validated) = il C3M scalato: quota d'espressioni del C3M per dominio
#       (STREAM_EXPR_FRAC bfm2019=0,ict=0.2315,gnm=0.1968: ICT 50.000/54.008 x 2/8, GNM (5.038 x 2/8 + 4.962 x 1/7)
#       / 10.000, BFM 0; sono le etichette rexprA/B di factorized/c3m/spec.json col campionatore v1, 0.233 e 0.199
#       simulati), nessun moltiplicatore (STREAM_MM_AUG vuoto), nessuna rotazione a ogni uso (STREAM_ROT 0: il C3M
#       ruota solo nel rumore della ricetta v1), discretizzazioni senza reinserimento nel gruppo (STREAM_LABEL_DRAW
#       perm, come _sample_subject_mesh_entries);
#   max (default per ogni altro STREAM_SOURCES) = le deviazioni, da proporre all'utente (PLAN_MASSIVE sez. 22):
#       STREAM_EXPR_FRAC 0.5, STREAM_MM_AUG hybrid=0.3,expr_transfer=0.15,rbf=0.15, STREAM_ROT 30,15,10,
#       STREAM_LABEL_DRAW replace (coi pesi di views.LABEL_WEIGHTS).
#   Differenze dichiarate anche con c3m: BFM 2019 al posto di BFM REMESH; STREAM_SIZE_MASK vuoto (il C3M toglie bfm
#   dalla MSE di s perche' le taglie di BFM REMESH sono normalizzate una per una; quelle di BFM 2019 sono vere, CV
#   0.049, massive_ready.md sez. 3; STREAM_SIZE_MASK=bfm2019 per riprodurre la maschera); forward groups (C3M
#   sequential); identita' fresche dai produttori invece di 64.400 fisse; espressioni su qualunque discretizzazione
#   (nel C3M rexpr* sono sulla topologia original); k_eig STREAM_K 128.
# Opt-in del run SECONDARIO (secondary_partial_a100.sbatch), spento di default (riga di lancio e anello di prima):
#   STREAM_PARTIAL p (STREAM_PARTIAL_AREA lo,hi): parzialita' variabile per vista, producer.py --partial-p (partial_aug.py).
# Opt-in della cella B' dell'ablazione 2x2 (emendamento 1), spento di default: STREAM_FLAME_SUBDIV n, le viste di
#   flame2023 suddivise 1-a-4 n volte coi punti medi (producer.py --subdiv flame2023=n, come gli insiemi FLAME di D1).
# Ripresa, anche del run principale:
#   STREAM_PREEMPTIBLE (default 1; 0 = trainer in primo piano, come prima): WBES_PREEMPT_SAVE=1 al trainer (su SIGTERM
#   salva last.pth al passo in corso ed esce: train_stream.py). Prelazione, scontrol requeue e scancel mandano SIGTERM
#   SOLO al capo del task, cioe' a questo bash, non ai figli (proctrack/cgroup, Slurm 21.08: misurato l'11 ottobre,
#   aau/runs/evidence/stream/partial_aug/sigtest), e SIGKILL a tutti dopo KillWait (30 s): qui torchrun gira in
#   background, la trap inoltra SIGTERM (run.sh -> singularity -> torchrun -> rank, inoltro misurato) e non fa partire
#   altri tentativi; l'anello resta finche' il trainer ha salvato. Con 0 il bash muore sul SIGTERM e la trap EXIT
#   termina i rank senza salvataggio: si riprende dall'ultimo checkpoint periodico (STREAM_CKPT_MIN).
#   --stream (l'anello) e --ckpt-minutes entrano nell'hash della run dir (train_v3.make_run_dir): l'anello e'
#   /tmp/<STREAM_RUN_TAG>_stream/r0 a ogni partenza (STREAM_RUN_TAG = il job della prima partenza, anche in una
#   continuazione con un job nuovo, massive.sbatch) e STREAM_CKPT_MIN non va cambiato fra le partenze.
#   Codice: la copia in STREAM_CODE (massive.sbatch, prima partenza; vuota = il repo); dati e percorsi della riga di
#   lancio dal repo vivo (WBES_ROOT, AAU_RUNS), come prima. STREAM_EXPECT_RESUME=1 (riavvio o continuazione): la run dir
#   di questa riga di lancio deve avere last.pth (controllo prima del primo tentativo), altrimenti errore e uscita 3.
set -euo pipefail
source "${STREAM_CODE:-${WBES_ROOT:-$PWD}}/aau/env.sh"
cd "$WBES_ROOT"
CODE="${STREAM_CODE:-$WBES_ROOT}"
source "$AAU_DIR/data_scale/recipe_v1.sh"
read -r NODE NN < <(python3 -c "import socket, sys; n = sys.argv[1].split(); h = socket.gethostname().split('.')[0]; \
print(next(i for i, x in enumerate(n) if x.split('.')[0] == h), len(n))" "${STREAM_NODES:-$(hostname -s)}")
RST="${SLURM_RESTART_COUNT:-0}"
SEG="${STREAM_SEGMENT:-$RST}"              # partenze precedenti del run (= RST se non ci sono continuazioni)
G=$(nvidia-smi -L | wc -l)
ARM="${STREAM_ARM:-factorized}"
SOURCES="${STREAM_SOURCES:?STREAM_SOURCES obbligatorio: validated | max | open_core | lista di domini (massive.sbatch)}"
IDS="${STREAM_IDS_PER_RANK:-5}"            # identita' per rank e per passo (C3M: --batch_subjects 5)
MPI="${STREAM_MESHES_PER_ID:-6}"           # viste massime per identita' (C3M: --max_meshes_per_subject_train 6)
VIEWS="${STREAM_VIEWS:-$MPI}"              # viste per identita' scritte dai produttori (producer.py --views)
BDOM="${STREAM_BATCH_DOMAINS:-single}"     # single (C3M) | mixed
SMASK="${STREAM_SIZE_MASK-}"               # domini fuori dalla MSE di s (vuoto = nessuno; vedi sopra)
K="${STREAM_K:-128}"
T="${STREAM_STEPS:-2000}"
S="${STREAM_SPE:-200}"
R="${STREAM_REUSE:-4}"
RCP="${STREAM_RECIPE:-$([[ "$SOURCES" == validated ]] && echo c3m || echo max)}"
case "$RCP" in
  c3m) D_EF="bfm2019=0,ict=0.2315,gnm=0.1968"; D_MMAUG=""; D_ROT="0"; D_LD="perm" ;;
  max) D_EF="0.5"; D_MMAUG="hybrid=0.3,expr_transfer=0.15,rbf=0.15"; D_ROT="30,15,10"; D_LD="replace" ;;
  *) echo "[node] ERRORE: STREAM_RECIPE=$RCP (c3m | max)" >&2; exit 2 ;;
esac
EF="${STREAM_EXPR_FRAC:-$D_EF}"
MMAUG="${STREAM_MM_AUG-$D_MMAUG}"          # vuoto = solo identita' pure
ROT="${STREAM_ROT:-$D_ROT}"                # yaw,pitch,roll massimi a ogni uso (0 = nessuna)
LD="${STREAM_LABEL_DRAW:-$D_LD}"           # perm (senza reinserimento, C3M) | replace
PART="${STREAM_PARTIAL:-}"                 # probabilita' per vista della parzialita' (vuoto = spenta)
PAREA="${STREAM_PARTIAL_AREA:-}"           # lo,hi della frazione d'area tolta (vuoto = default di partial_aug)
FSUB="${STREAM_FLAME_SUBDIV:-}"            # suddivisioni 1-a-4 delle viste di flame2023 (vuoto = nessuna)
CPR="${STREAM_TRAIN_CPUS_PER_RANK:-8}"     # ottimo misurato il 10 ottobre: groups 8 / 12 / 16 CPU = 108 / 112 / 110 mesh/s
RGB=$(( ${STREAM_RING_GB_PER_GPU:-10} * G ))
SEED="${STREAM_SEED:-1234}"
E="$STREAM_OUT/node$NODE"
mkdir -p "$E"
ln -sfn ../runs "$E/runs"
JOBTMP="/tmp/${STREAM_RUN_TAG:-${SLURM_JOB_ID:-manual}}_stream"
# --stream e' nell'hash della run dir (train_v3.make_run_dir): con r$RST un requeue cambierebbe run dir e ripartirebbe
# da zero invece di riprendere da last.pth; r0 a ogni partenza (JOBTMP si svuota comunque qui sotto); alla prima
# partenza RST = 0, quindi riga di lancio e run dir sono quelle di prima
RING="$JOBTMP/r0"
rm -rf "$JOBTMP"
mkdir -p "$RING"
trap 'kill $(jobs -p) 2>/dev/null || true; rm -rf "$JOBTMP"' EXIT
log() { echo "[node$NODE] $(date +%F_%T) $*" | tee -a "$E/node.log"; }
log "host=$(hostname) GPU=$G ($(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | xargs)) CPU=$(nproc)" \
    "riavvio=$RST segmento=$SEG arm=$ARM fonti=$SOURCES k=$K T=$T S=$S R=$R anello=$RING (${RGB}GiB) codice=$CODE"
log "dati: ricetta $RCP, quota d'espressioni $EF, mm_aug ${MMAUG:-no}, rotazione a ogni uso $ROT gradi," \
    "discretizzazioni $LD"
[[ -n "$PART" ]] && log "parzialita' variabile: p=$PART per vista, area ${PAREA:-default} (producer.py --partial-p)"
[[ -n "$FSUB" ]] && log "viste di flame2023 suddivise 1-a-4 $FSUB volte (producer.py --subdiv)"
log "ricetta: batch ${IDS} identita' x <= ${MPI} viste per rank (produttori: ${VIEWS} viste per identita'), batch" \
    "$BDOM, lr costante, maschera di taglia: ${SMASK:-nessuna}, forward ${STREAM_FORWARD:-groups}"

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
PSEED=$(( 20261009 + 1000 * NODE + 100000 * SEG ))
AAU_NV= "$AAU_DIR/run.sh" "$CODE/v3_work/stream/producer.py" --ring "$RING" --ring-gb "$RGB" --n-proc "$NPROD" --k-eig "$K" \
    --evecs-dtype fp32 --sources "$SOURCES" --provenance --canonical-gt --expr-frac "$EF" --seed "$PSEED" \
    --views "$VIEWS" --label-draw "$LD" \
    ${MMAUG:+--mm-aug "$MMAUG"} ${PART:+--partial-p "$PART"} ${PAREA:+--partial-area "$PAREA"} \
    ${FSUB:+--subdiv "flame2023=$FSUB"} \
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
           ${SMASK:+--size-mask-domains "$SMASK"} --gt-scale "$KAPPA" --stream-gt sr --dist_npz "$C3F/gt_sr.npz") ;;
  ctrlfr)
    ARMF+=(--stream-gt fr --dist_npz "$C3F/gt_frcal.npz") ;;
  unified)
    ARMF=(--area robust --area-robust smooth --dropout 0.0 --stream-gt unified
          --dist_npz face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz) ;;
  *) log "ERRORE: STREAM_ARM=$ARM"; exit 2 ;;
esac
EXTRA=()
[[ -n "${STREAM_EXTRA:-}" ]] && EXTRA+=(--stream-extra "$STREAM_EXTRA" --stream-extra-mirror "$JOBTMP/mirror" --stream-extra-mirror-threads "${STREAM_MIRROR_THREADS:-2}")
EP=$(( (T + S - 1) / S ))
# checkpoint (pesi + EMA, epochNNN.pth 11.7 MB + epochNNN_ema.pth 2.9 MB nel C3M) ogni ~10% del run e alla fine
CMD=("$CODE/v3_work/stream/train_stream.py" --stream "$RING" --stream-reuse "$R" --stream-rot "$ROT" --stream-sources "$SOURCES" --stream-log-views
  --stream-wait-s 1800 --total-steps "$T" --steps-per-epoch "$S" --epochs "$EP"
  --data_dir datasets/REMESH/npz_data_topo_500_withops_areanorm --no-cache
  "${RECIPE_V1[@]}" --batch_subjects "$IDS" --max_meshes_per_subject_train "$MPI" --lr-constant
  --stream-batch-domains "$BDOM" --stream-log-batches "${STREAM_LOG_BATCHES:-200}"
  --forward "${STREAM_FORWARD:-groups}" --fast-data --ema-decay 0.999 --train-threads "$CPR"
  --eval_every "${STREAM_EVAL_EVERY:-$(( EP > 10 ? EP / 10 : 1 ))}"
  --save_every "${STREAM_SAVE_EVERY:-$(( EP > 10 ? EP / 10 : 1 ))}"
  --seed "$SEED" --runs_root "$STREAM_OUT/runs" --log-every 50 --ckpt-minutes "${STREAM_CKPT_MIN:-20}"
  "${ARMF[@]}" "${EXTRA[@]}")
printf '%q ' "${CMD[@]}" > "$E/launch.txt"; echo >> "$E/launch.txt"
if [[ "${STREAM_EXPECT_RESUME:-0}" == 1 ]]; then
  # riavvio o continuazione: la run dir di QUESTA riga di lancio deve avere last.pth (train_stream.py, solo il controllo)
  export WBES_EXPECT_RESUME=1
  if ! WBES_CHECK_RESUME_ONLY=1 "$AAU_DIR/run.sh" "${CMD[@]}" > "$E/resume_check.log" 2>&1; then
    log "ERRORE: $(grep -m1 '^\[resume\]' "$E/resume_check.log" || tail -3 "$E/resume_check.log")"
    exit 3
  fi
  log "$(grep -m1 '^\[resume\]' "$E/resume_check.log")"
fi
export WBES_STACK_DUMP_S="${WBES_STACK_DUMP_S:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
[[ -n "${STREAM_NCCL_IB_HCA:-}" ]] && export NCCL_IB_HCA="$STREAM_NCCL_IB_HCA"
# Ripresa dopo un crash: un torchrun NUOVO per tentativo (--max-restarts 0, rendezvous e porta nuovi), che riparte
# da last.pth (--resume auto). Il riavvio elastico DENTRO lo stesso torchrun su piu' nodi si blocca nella prima
# collettiva di NCCL (misurato il 9 ottobre, 1 + 1 GPU): per questo non si usa.
ATT=0
TERMED=""
TPID=""
if [[ "${STREAM_PREEMPTIBLE:-1}" == 1 ]]; then
  export WBES_PREEMPT_SAVE=1
  trap 'TERMED=1; [[ -n "$TPID" ]] && kill -TERM "$TPID" 2>/dev/null' TERM
  log "prelazionabile: su SIGTERM il trainer salva al passo in corso; checkpoint ogni ${STREAM_CKPT_MIN:-20} min"
fi
while :; do
  XC=()
  [[ -n "${STREAM_TEST_CRASH:-}" && "$ATT" == 0 && "$RST" == 0 ]] && XC=(--test-crash-at-step "$STREAM_TEST_CRASH")
  log "torchrun tentativo $ATT: $NN nodi, $G rank qui, rendezvous $STREAM_MASTER:$(( STREAM_PORT + ATT ))"
  TR=(taskset -c "$TRAIN_CPUS" "$AAU_DIR/run.sh" -m torch.distributed.run
    --nnodes "$NN" --nproc-per-node "$G" --rdzv-backend c10d --rdzv-endpoint "$STREAM_MASTER:$(( STREAM_PORT + ATT ))"
    --rdzv-id "${SLURM_JOB_ID:-manual}_${RST}_$ATT" --max-restarts 0 "${CMD[@]}" "${XC[@]}")
  set +e
  if [[ "${STREAM_PREEMPTIBLE:-1}" == 1 ]]; then
    # in background: wait si interrompe per la trap (un comando in primo piano la rimanderebbe alla sua fine)
    OMP_NUM_THREADS="$CPR" MKL_NUM_THREADS="$CPR" "${TR[@]}" \
      > >(stdbuf -oL tr '\r' '\n' | grep --line-buffered -v '%|\|FloatTensor\|SparseTensor' >> "$E/train.log") 2>&1 &
    TPID=$!
    [[ -n "$TERMED" ]] && kill -TERM "$TPID" 2>/dev/null    # SIGTERM arrivato prima di questo tentativo
    while :; do
      wait "$TPID"
      rc=$?
      kill -0 "$TPID" 2>/dev/null || break
    done
    TPID=""
  else
    OMP_NUM_THREADS="$CPR" MKL_NUM_THREADS="$CPR" "${TR[@]}" 2>&1 \
      | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v '%|\|FloatTensor\|SparseTensor' >> "$E/train.log"
    rc=${PIPESTATUS[0]}
  fi
  set -e
  log "trainer finito rc=$rc (tentativo $ATT)"
  [[ "$rc" == 0 ]] && break
  [[ -n "$TERMED" ]] && { log "SIGTERM (prelazione o requeue): nessun nuovo tentativo"; break; }
  ATT=$(( ATT + 1 ))
  [[ "$ATT" -gt "${STREAM_ATTEMPTS:-3}" ]] && break
  sleep 30
done
kill %1 2>/dev/null || true
wait %1 2>/dev/null || true
[[ "$NODE" == 0 ]] && python3 "$CODE/v3_work/stream/summarize_run.py" "$E" | tee -a "$E/node.log" || true
exit "$rc"
