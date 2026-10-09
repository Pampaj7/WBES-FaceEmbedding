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
#       c2f40, c3f40  K=1, T=21.096: ICT a ~1/40 (un blocco = le identita' viste da c2m in ~10 epoche)
#       c3mv  il run su scala RIFATTO su V100 (decisione del PI, 8 ottobre: tutte le celle sulla stessa GPU):
#            stesso split, K=46, T=105.480 (E=360), fermato dopo il checkpoint dell'epoca 72
#   - --save_every 36 esplicito (train_scale.sbatch lo ricava da E/10): checkpoint a 10.548 e 21.096 passi.
# GPU: V100 (decisione del PI): container PyTorch <= 24.11 (sm_70), controllato qui. Sui nodi V100 /tmp e /raid
# sono dischi: lo staging del pre-pass va su /raid/$USER (14 TB), fuori da --mem; i dati sono gli stessi.
# Smoke: WBES_E1_SMOKE=1 nodo cpu (split ridotto, S e T piccoli, arresto dopo l'epoca 2, device cpu);
#        WBES_E1_SMOKE=v100 la configurazione di c3mv fermata dopo l'epoca 1, da confrontare con la loss
#        dell'epoca 1 del run su scala su L40S (stessi soggetti, stessi semi).
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv
aau_require_diffusion_net
source "$AAU_DIR/data_scale/recipe_v1.sh"

CELL="${WBES_E1_CELL:?WBES_E1_CELL=c2m|c2f|c3f|c2fgnm|c3fugt|c3fugtraw|c2fs2|c3fs2|c2f40|c3f40|g1|c3mv}"
E1="$AAU_DIR/evidence/e1_factorial"
SEED=1234
S=293
SAVE_EVERY=36
SPLIT_CELL="$CELL"
GT_FILE="SCALE_ALL/gt_joint_bfm_ict_gnm.npz"
case "$CELL" in
    c2m) NB=40; T=91709; STOP=72 ;;
    c2f|c3f|c2fgnm) NB=4; T=21096; STOP=0 ;;
    # secondo seme (critic, 8 ottobre): stesso split, --seed e block_seed 2345 (la politica del run su scala
    # lega i due semi); nient'altro cambia
    c2fs2|c3fs2) NB=4; T=21096; STOP=0; SEED=2345; SPLIT_CELL="${CELL%s2}" ;;
    # C3F con la GT UNIFICATA delle identita' di training (E8/E11, datasets/UNIFIED_GT/train): stesso split,
    # passi, seme e hardware di c3f; cambia solo --dist_npz. c3fugt usa la GT unificata TARATA sulle mediane
    # per dominio della maxabs del run (emendamento 2: i margini della loss sono in unita' GT); c3fugtraw quella non
    # tarata, a bassa priorita', per misurare l'effetto della scala
    c3fugt) NB=4; T=21096; STOP=0; SPLIT_CELL=c3f; GT_FILE="UNIFIED_GT/train/gt_unified_bfm_ict_gnm_calib.npz" ;;
    c3fugtraw) NB=4; T=21096; STOP=0; SPLIT_CELL=c3f; GT_FILE="UNIFIED_GT/train/gt_unified_bfm_ict_gnm.npz" ;;
    g1) NB=6; T=21096; STOP=0 ;;
    c2f40|c3f40) NB=1; T=21096; STOP=0 ;;
    c3mv) NB=46; T=105480; STOP=72 ;;
    *) echo "ERRORE: cella '$CELL'" >&2; exit 2 ;;
esac
SPLIT="$E1/split_$SPLIT_CELL.json"
[[ "$CELL" == c3mv ]] && SPLIT="$AAU_DIR/data_scale/split_scale_all.json"   # lo split del run su scala
DEVICE=()
PREPASS_PROC="${WBES_E1_PREPASS_PROC:-22}"   # solo il tempo del pre-pass, non i dati
TRAIN_THREADS=8
if [[ "${WBES_E1_SMOKE:-0}" == 1 ]]; then
    # plumbing: stesso percorso (spec, split esplicito, blocchi, arresto, sync), dimensioni minime
    SPLIT="$E1/smoke/split_${CELL}_smoke.json"
    S=2; NB=2; SAVE_EVERY=1; T=20; STOP=2
    DEVICE=(--device cpu)
    PREPASS_PROC=8
    TRAIN_THREADS=4
    export AAU_NV=""
elif [[ "${WBES_E1_SMOKE:-0}" == v100 ]]; then
    [[ "$CELL" == c3mv ]] || { echo "ERRORE: lo smoke v100 e' sulla configurazione di c3mv" >&2; exit 2; }
    STOP=1; SAVE_EVERY=1
fi
[[ -f "$SPLIT" ]] || { echo "ERRORE: split assente $SPLIT (make_e1_splits.py)" >&2; exit 1; }
# build_grad vettorizzato di E9 nel pre-pass, in TUTTE le celle (decisione del PI, 8 ottobre; nota tecnica in
# aau/runs/evidence/e1/nota_tecnica_gradvec.md): gradX/gradY diversi dall'originale al piu' di 1.2e-10 assoluto,
# 1.64x sul pre-pass. Aggancio: gradvec_site/sitecustomize.py, attivo in ogni processo Python del job
GRADVEC=0
if [[ "${WBES_E1_SMOKE:-0}" == 0 ]]; then
    GRADVEC=1
    export WBES_E1_GRADVEC=1 PYTHONPATH="$E1/gradvec_site${PYTHONPATH:+:$PYTHONPATH}"
fi
EPOCHS=$(( (T + S - 1) / S ))
D="$WBES_ROOT/datasets"
case "${WBES_E1_SMOKE:-0}" in
    1) RUN_NAME="train_${CELL}_s${SEED}_smoke" ;;
    v100) RUN_NAME="train_${CELL}_s${SEED}_smokev100" ;;
    *) RUN_NAME="train_${CELL}_s${SEED}" ;;
esac
RUNS_ROOT="${WBES_RUNS_ROOT:-$AAU_RUNS/evidence/e1/${RUN_NAME}_${SLURM_JOB_ID:-manual}}"
if [[ -e "$RUNS_ROOT/launch.txt" || -e "$RUNS_ROOT/train.log" ]]; then
    n=1; while [[ -e "${RUNS_ROOT}_rerun$n" ]]; do n=$(( n + 1 )); done
    echo "[e1] $RUNS_ROOT contiene gia' un run: uso ${RUNS_ROOT}_rerun$n" >&2
    RUNS_ROOT="${RUNS_ROOT}_rerun$n"
fi
# puntatore alla run dir vera, per la catena di eval (e1_eval_body.sh) e il riepilogo: uno per cella
PTR_NAME="train_${CELL}"; [[ "${WBES_E1_SMOKE:-0}" != 0 ]] && PTR_NAME="$RUN_NAME"
echo "$RUNS_ROOT" > "$AAU_RUNS/evidence/e1/${PTR_NAME}.runs_root"
# staging: /raid sui nodi V100 (disco), altrimenti /tmp (RAM sui nodi L40S, conta contro --mem)
if [[ -d /raid && -w /raid ]]; then
    STAGE="/raid/$USER/e1_${SLURM_JOB_ID:-manual}"
    # il container non monta /raid da solo (smoke 1061790/1061791: "Read-only file system: '/raid'")
    export SINGULARITY_BIND="/raid${SINGULARITY_BIND:+,$SINGULARITY_BIND}" APPTAINER_BIND="/raid${APPTAINER_BIND:+,$APPTAINER_BIND}"
else
    STAGE="/tmp/${SLURM_JOB_ID:-manual}"
fi
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
    [[ -n "${GPU_PID:-}" ]] && kill "$GPU_PID" 2> /dev/null
    [[ -n "$TRAIN_PID" ]] && kill -TERM -- -"$TRAIN_PID" 2> /dev/null
    final_sync || true
    rm -rf "$STAGE"
    [[ "$0" == /tmp/e1_train_body_* ]] && rm -f "$0"
}
trap cleanup EXIT

# spec: quella di train_scale.sbatch con frame spento, carattere per carattere; cambia solo n_blocks
SH="$D/SCALE_ALL/shards"
TARS=$(find "$SH" -maxdepth 1 -name '*shard_*.tar' | sort | sed 's/.*/"&"/' | paste -sd, -)
N_TARS=$(find "$SH" -maxdepth 1 -name '*shard_*.tar' | wc -l)
[[ "$N_TARS" == 241 ]] || { echo "ERRORE: $N_TARS shard in $SH, attesi 241 (200 ICT + 41 GNM)" >&2; exit 1; }
GT="$D/$GT_FILE"
if [[ "$CELL" == c3fugtraw ]]; then
    # solo una GT gia' verificata col loader del trainer (v3_work/unified_gt/check_train_gt.py -> check.json)
    CHK="$D/UNIFIED_GT/train/check.json"
    AAU_NV= "$AAU_DIR/run.sh" -c "
import json, sys
c = json.load(open(sys.argv[1]))
bad = (not c['name_to_idx_identical_to_run_gt'] or c['n_nonfinite'] or c['diag_max_abs'] > 0
       or c['symmetry_max_abs_sample'] > 0 or not all(v['all_same_index'] for v in c['index_match'].values())
       or c['values_vs_s_train_max_abs_mm'] > 1e-3)
sys.exit(int(bad))
" "$CHK" 2> /dev/null || { echo "ERRORE: GT unificata non verificata ($CHK)" >&2; exit 1; }
fi
if [[ "$CELL" == c3fugt ]]; then
    # la tarata: verificata da e1_calib_ugt.py check (check_calib.json, "ok") e piu' recente del file
    CHK="$D/UNIFIED_GT/train/check_calib.json"
    [[ -f "$CHK" && "$CHK" -nt "$GT" ]] || { echo "ERRORE: GT tarata assente o non verificata ($CHK)" >&2; exit 1; }
    AAU_NV= "$AAU_DIR/run.sh" -c "
import json, sys
sys.exit(0 if json.load(open(sys.argv[1]))['ok'] is True else 1)
" "$CHK" 2> /dev/null || { echo "ERRORE: GT tarata non verificata ($CHK)" >&2; exit 1; }
fi
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
# controllo: a parte n_blocks e block_seed (= --seed, la politica del run su scala; 2345 nelle celle s2), la spec e'
# quella del run su scala. Prima versione senza block_seed: c2fs2/c3fs2 (1061852/1061854) fermati qui
REF_SPEC="$AAU_RUNS/data_scale_runs/scale_bfm_ict_gnm_s1234_nocanon_noaug_20261007_1411/spec.json"
AAU_NV= "$AAU_DIR/run.sh" -c "
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
ok = b['block_seed'] == int(sys.argv[3])
for k in ('n_blocks', 'block_seed'):
    a.pop(k); b.pop(k)
sys.exit(int(a != b or not ok))
" "$REF_SPEC" "$RUNS_ROOT/spec.json" "$SEED" 2> /dev/null || { echo "ERRORE: spec diversa da $REF_SPEC oltre a n_blocks e block_seed" >&2; exit 1; }

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

echo "[e1] cella=$CELL host=$(hostname) job=${SLURM_JOB_ID:-none} T=$T S=$S epoche=$EPOCHS blocchi=$NB arresto=${STOP:-0} split=$SPLIT runs_root=$RUNS_ROOT stage=$STAGE"
GPU_NAME="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2> /dev/null | head -1 || true)"
CVER="$(basename "$CONTAINER" .sif)"; CVER="${CVER#pytorch_}"
echo "[e1] GPU: ${GPU_NAME:-nessuna}; container $CONTAINER"
# V100 = sm_70: dal 25.02 i container PyTorch non lo supportano piu'
if [[ "$GPU_NAME" == *V100* && "${CVER//./}" > 2411 ]]; then
    echo "ERRORE: V100 con il container $CVER (serve <= 24.11)" >&2
    exit 1
fi
if [[ -n "$GPU_NAME" ]]; then
    nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader -l 60 \
        >> "$RUNS_ROOT/gpu.log" 2> /dev/null &
    GPU_PID=$!
fi

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
{"cell": "$CELL", "split": "$SPLIT", "gpu": "$GPU_NAME", "container": "$CONTAINER", "stage": "$STAGE", "n_blocks": $NB, "total_steps_nominal": $T, "steps_per_epoch": $S,
 "epochs_nominal": $EPOCHS, "stop_after_epoch": $STOP, "save_every": $SAVE_EVERY, "seed": $SEED,
 "gradvec": $GRADVEC, "job": "${SLURM_JOB_ID:-manual}", "host": "$(hostname)", "mem_request": "${SLURM_MEM_PER_NODE:-?}M",
 "smoke": "${WBES_E1_SMOKE:-0}"}
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
# prova che l'aggancio era attivo nei lavoratori del pre-pass (una riga per processo)
echo "[e1] righe [e1-gradvec] nei log del pre-pass: $(cat "$RUNS_ROOT"/prepass_logs/*.prepass.log 2> /dev/null | grep -c '\[e1-gradvec\]' || true) (attese > 0 con gradvec=$GRADVEC)"
for e in 036 072; do
    [[ "${WBES_E1_SMOKE:-0}" != 0 ]] && e=$(printf %03d "$STOP")
    ls "$RUNS_ROOT"/*/checkpoints/epoch$e.pth > /dev/null 2>&1 || { echo "ERRORE: manca epoch$e.pth" >&2; RC=1; }
done
(( RC == 0 )) && echo "[e1] OK cella=$CELL runs_root=$RUNS_ROOT"
exit "$RC"
