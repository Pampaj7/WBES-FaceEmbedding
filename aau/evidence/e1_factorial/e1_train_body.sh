#!/usr/bin/env bash
# Corpo di e1_train.sbatch (letto all'avvio del job). Una cella di E1 con la ricetta, il trainer e la
# politica dei semi del run su scala 1060130 (aau/data_scale/train_scale.sbatch, che resta invariato):
# stessi flag, stessa spec dei dati (viste BFM + ICT-5000, i 241 tar di SCALE_ALL, gruppi di espressioni,
# BFM residente con il 8.87% dei passi), stessa GT, stesso held-out ed eval online. Cambiano SOLO:
#   - lo split (--split-json aau/evidence/e1_factorial/split_<cella>.json, make_e1_splits.py);
#   - il numero di blocchi K (un blocco deve contenere i soggetti di un'epoca per dominio: e1_design.py);
#   - T nominale ed epoca di arresto:
#       c2m  K=40, T=91.709 (E=313): cambia blocco alle STESSE epoche di c3m (K=46, E=360) e il job si
#            ferma dopo il checkpoint dell'epoca 72 (21.096 passi), come i checkpoint e036/e072 di c3m,
#            che sono istantanee di un run piu' lungo
#       c2f, c3f  K=4, T=21.096 (E=72)          g1  K=6, T=21.096 (E=72)
#   - --save_every 36 esplicito (train_scale.sbatch lo ricava da E/10): checkpoint a 10.548 e 21.096 passi.
# Smoke (WBES_E1_SMOKE=1, nodo cpu): split ridotto, S e T piccoli, arresto dopo l'epoca 2, device cpu.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv
aau_require_diffusion_net
source "$AAU_DIR/data_scale/recipe_v1.sh"

CELL="${WBES_E1_CELL:?WBES_E1_CELL=c2m|c2f|c3f|g1}"
E1="$AAU_DIR/evidence/e1_factorial"
SEED=1234
S=293
SAVE_EVERY=36
case "$CELL" in
    c2m) NB=40; T=91709; STOP=72 ;;
    c2f|c3f) NB=4; T=21096; STOP=0 ;;
    g1) NB=6; T=21096; STOP=0 ;;
    *) echo "ERRORE: cella '$CELL' (c2m|c2f|c3f|g1)" >&2; exit 2 ;;
esac
SPLIT="$E1/split_$CELL.json"
DEVICE=()
PREPASS_PROC=22
TRAIN_THREADS=8
if [[ "${WBES_E1_SMOKE:-0}" == 1 ]]; then
    # plumbing: stesso percorso (spec, split esplicito, blocchi, arresto, sync), dimensioni minime
    SPLIT="$E1/smoke/split_${CELL}_smoke.json"
    S=2; NB=2; SAVE_EVERY=1; T=20; STOP=2
    DEVICE=(--device cpu)
    PREPASS_PROC=8
    TRAIN_THREADS=4
    export AAU_NV=""
fi
[[ -f "$SPLIT" ]] || { echo "ERRORE: split assente $SPLIT (make_e1_splits.py)" >&2; exit 1; }
EPOCHS=$(( (T + S - 1) / S ))
D="$WBES_ROOT/datasets"
RUN_NAME="train_${CELL}_s${SEED}$( [[ "${WBES_E1_SMOKE:-0}" == 1 ]] && echo _smoke )"
RUNS_ROOT="${WBES_RUNS_ROOT:-$AAU_RUNS/evidence/e1/${RUN_NAME}_${SLURM_JOB_ID:-manual}}"
if [[ -e "$RUNS_ROOT/launch.txt" || -e "$RUNS_ROOT/train.log" ]]; then
    n=1; while [[ -e "${RUNS_ROOT}_rerun$n" ]]; do n=$(( n + 1 )); done
    echo "[e1] $RUNS_ROOT contiene gia' un run: uso ${RUNS_ROOT}_rerun$n" >&2
    RUNS_ROOT="${RUNS_ROOT}_rerun$n"
fi
# puntatore alla run dir vera, per la catena di eval (e1_eval_body.sh)
echo "$RUNS_ROOT" > "$AAU_RUNS/evidence/e1/${RUN_NAME}.runs_root"
STAGE="/tmp/${SLURM_JOB_ID:-manual}"
RUN_TMP="$STAGE/run"
LOG_TMP="$STAGE/train.log"
mkdir -p "$RUNS_ROOT" "$RUN_TMP"

# come train_scale.sbatch: run dir su /tmp, copiata nella home ogni 10 minuti e a fine job
sync_home() {
    [[ -f "$LOG_TMP" ]] && { tr '\r' '\n' < "$LOG_TMP" | sed 's/\(Epoch [0-9]\{3\} |\)/\n\1/g' \
        | grep -v '%|.*\(it/s\|s/it\)\|^[[:space:]]*$' > "$RUNS_ROOT/train.log.part" \
        && mv -f "$RUNS_ROOT/train.log.part" "$RUNS_ROOT/train.log"; } 2> /dev/null
    rsync -a --exclude 'stage/' "$RUN_TMP/" "$RUNS_ROOT/" 2> /dev/null
    mkdir -p "$RUNS_ROOT/prepass_logs" && cp -f "$STAGE"/stage/*.prepass.log "$STAGE"/stage/*.subjects.txt \
        "$RUNS_ROOT/prepass_logs/" 2> /dev/null
    return 0
}
final_sync() {
    local i
    for (( i = 0; i < 30; i++ )); do
        sync_home && { echo "[e1] run dir copiata in $RUNS_ROOT" || true; return 0; }
        sleep 60
    done
    echo "ERRORE: sync finale verso $RUNS_ROOT fallito per 30 minuti" >&2 || true
    return 1
}
SYNC_PID=""
STOP_PID=""
TRAIN_PID=""
cleanup() {
    [[ -n "$SYNC_PID" ]] && kill "$SYNC_PID" 2> /dev/null
    [[ -n "$STOP_PID" ]] && kill "$STOP_PID" 2> /dev/null
    [[ -n "${MEM_PID:-}" ]] && kill "$MEM_PID" 2> /dev/null
    [[ -n "$TRAIN_PID" ]] && kill -TERM -- -"$TRAIN_PID" 2> /dev/null
    final_sync || true
    rm -rf "$STAGE"
}
trap cleanup EXIT

# spec: quella di train_scale.sbatch con frame spento, carattere per carattere; cambia solo n_blocks
SH="$D/SCALE_ALL/shards"
TARS=$(find "$SH" -maxdepth 1 -name '*shard_*.tar' | sort | sed 's/.*/"&"/' | paste -sd, -)
N_TARS=$(find "$SH" -maxdepth 1 -name '*shard_*.tar' | wc -l)
[[ "$N_TARS" == 241 ]] || { echo "ERRORE: $N_TARS shard in $SH, attesi 241 (200 ICT + 41 GNM)" >&2; exit 1; }
GT="$D/SCALE_ALL/gt_joint_bfm_ict_gnm.npz"
VB="$AAU_DIR/data_scale/view_bytes_all.json"
cat > "$RUNS_ROOT/spec.json" <<JSON
{"views": ["$D/REMESH/npz_data_topo_500_withops_areanorm", "$D/ICT/train_ready/npz_withops"], "geom_dirs": [], "tars": [$TARS], "tar_index": "$SH/index.npz", "view_bytes_json": "$VB",
 "labels": null,
 "label_groups": {"by_domain": {
   "ict": {"rexprA": ["rexpr1","rexpr2","rexpr3","rexpr4"], "rexprB": ["rexpr5","rexpr6","rexpr7","rexpr8"]},
   "gnm": {"rexprA": ["rexpr1"], "rexprB": ["rexpr2"]}}},
 "canon": null, "aug": null,
 "pin_domains": ["bfm"], "domain_step_share": {"bfm": 0.0887372}, "stratify_blocks": true,
 "convention": "areanorm", "n_blocks": $NB, "block_seed": $SEED}
JSON
# controllo: a parte n_blocks, la spec e' quella del run su scala
REF_SPEC="$AAU_RUNS/data_scale_runs/scale_bfm_ict_gnm_s1234_nocanon_noaug_20261007_1411/spec.json"
AAU_NV= "$AAU_DIR/run.sh" -c "
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
a.pop('n_blocks'); b.pop('n_blocks')
sys.exit(int(a != b))
" "$REF_SPEC" "$RUNS_ROOT/spec.json" 2> /dev/null || { echo "ERRORE: spec diversa da $REF_SPEC oltre a n_blocks" >&2; exit 1; }

# memoria del cgroup del JOB ogni 30 s, come train_scale.sbatch: totale, picco, rss, /tmp (shmem), page cache
JOBCG="/sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_${SLURM_JOB_ID:-none}"
( while [[ -r "$JOBCG/memory.usage_in_bytes" ]]; do
    awk -v t="$(date +%F_%T)" -v u="$(cat $JOBCG/memory.usage_in_bytes)" -v p="$(cat $JOBCG/memory.max_usage_in_bytes)" '
      $1=="total_rss"{r=$2} $1=="total_shmem"{s=$2} $1=="total_cache"{c=$2}
      END{printf "%s usage=%.1f peak=%.1f rss=%.1f shmem=%.1f cache=%.1f GiB\n", t, u/2^30, p/2^30, r/2^30, s/2^30, c/2^30}' \
      "$JOBCG/memory.stat" >> "$RUNS_ROOT/mem_job.log" 2> /dev/null
    sleep 30
  done ) &
MEM_PID=$!

echo "[e1] cella=$CELL host=$(hostname) job=${SLURM_JOB_ID:-none} T=$T S=$S epoche=$EPOCHS blocchi=$NB arresto=${STOP:-0} split=$SPLIT runs_root=$RUNS_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2> /dev/null || true

CMD=(
  v2_work/fastio/train_steps.py --total-steps "$T" --steps-per-epoch "$S"
  --split-json "$SPLIT"
  --frozen-heldout "$AAU_DIR/data_scale/heldout_frozen.json"
  --data-spec "$RUNS_ROOT/spec.json" --stage-root "$STAGE/stage" --prepass-proc "$PREPASS_PROC"
  --train-threads "$TRAIN_THREADS"
  --domain-blocked --eval_domain bfm --gt-keep-scale --lr-steps 81747:5e-5
  --cache-residency ram --cache-workers 16 --cache-max-gb 240 --frame current
  "${RECIPE_V1[@]}" "${DEVICE[@]}"
  --epochs "$EPOCHS" --eval_every 6 --save_every "$SAVE_EVERY" --seed "$SEED"
  --data_dir "$STAGE/stage" --dist_npz "$GT" --runs_root "$RUN_TMP"
)
printf '%q ' "${CMD[@]}" > "$RUNS_ROOT/launch.txt"; echo >> "$RUNS_ROOT/launch.txt"
cat > "$RUNS_ROOT/e1_cell.json" <<JSON
{"cell": "$CELL", "split": "$SPLIT", "n_blocks": $NB, "total_steps_nominal": $T, "steps_per_epoch": $S,
 "epochs_nominal": $EPOCHS, "stop_after_epoch": $STOP, "save_every": $SAVE_EVERY, "seed": $SEED,
 "job": "${SLURM_JOB_ID:-manual}", "host": "$(hostname)", "mem_request": "${SLURM_MEM_PER_NODE:-?}M",
 "smoke": ${WBES_E1_SMOKE:-0}}
JSON

( while sleep 600; do sync_home || true; done ) &
SYNC_PID=$!
# processo di training in un gruppo suo (setsid): l'arresto all'epoca STOP ferma trainer e pre-pass insieme
OMP_NUM_THREADS="$TRAIN_THREADS" MKL_NUM_THREADS="$TRAIN_THREADS" setsid "$AAU_DIR/run.sh" "${CMD[@]}" > "$LOG_TMP" 2>&1 &
TRAIN_PID=$!
if (( STOP > 0 )); then
    # train_runner salva epochNNN.pth a FINE epoca, prima che la barra tqdm dell'epoca successiva
    # ("Epoch <STOP+1>/<E>") parta: quando la barra compare, il checkpoint dell'epoca STOP e' completo
    CK_NAME="epoch$(printf %03d "$STOP").pth"
    ( until grep -q "Epoch $(( STOP + 1 ))/" "$LOG_TMP" 2> /dev/null; do
          kill -0 "$TRAIN_PID" 2> /dev/null || exit 0
          sleep 20
      done
      ck=$(ls "$RUN_TMP"/*/checkpoints/"$CK_NAME" 2> /dev/null | head -1)
      if [[ -n "$ck" ]]; then
          echo "[e1] epoca $(( STOP + 1 )) partita, $ck scritto: training fermato dopo $(( STOP * S )) passi ($(date +%F_%T))" \
              | tee "$RUNS_ROOT/stop.txt"
          # gruppo di setsid; in piu' per riga di comando (--stage-root e' propria di questo job)
          kill -TERM -- -"$TRAIN_PID" 2> /dev/null
          pkill -TERM -f -- "--stage-root $STAGE/stage" 2> /dev/null
          sleep 60
          kill -KILL -- -"$TRAIN_PID" 2> /dev/null
          pkill -KILL -f -- "--stage-root $STAGE/stage" 2> /dev/null
      fi ) &
    STOP_PID=$!
fi
RC=0
wait "$TRAIN_PID" || RC=$?
TRAIN_PID=""
[[ -n "$STOP_PID" ]] && { wait "$STOP_PID" 2> /dev/null || true; STOP_PID=""; }
kill "$SYNC_PID" 2> /dev/null || true
SYNC_PID=""
final_sync
if (( STOP > 0 )) && [[ -f "$RUNS_ROOT/stop.txt" ]]; then
    echo "[e1] arresto voluto dopo l'epoca $STOP (rc del trainer $RC)"
    RC=0
fi
grep -E "^\[steps\] passi eseguiti|^\[steps\] AVVISO|^Best" "$RUNS_ROOT/train.log" || true
for e in 036 072; do
    [[ "${WBES_E1_SMOKE:-0}" == 1 ]] && e=$(printf %03d "$STOP")
    ls "$RUNS_ROOT"/*/checkpoints/epoch$e.pth > /dev/null 2>&1 || { echo "ERRORE: manca epoch$e.pth" >&2; RC=1; }
done
(( RC == 0 )) && echo "[e1] OK cella=$CELL runs_root=$RUNS_ROOT"
exit "$RC"
