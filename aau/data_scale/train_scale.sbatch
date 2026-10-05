#!/usr/bin/env bash
#SBATCH --job-name=wbes-ds-train-scale
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=430G
#SBATCH --time=72:00:00
#SBATCH --no-requeue
#
# Run grande BFM + ICT esteso (aau/data_scale/PLAN.md): ricetta v1 del congiunto (recipe_v1.sh = flag
# di aau/cross3dmm/train_joint_E.sbatch, braccio ctrl) con v2_work/fastio/train_steps.py. NON lanciarlo
# a mano: lo lancia aau/data_scale/launch_scale.sh, che accoda anche la catena di eval.
#
# Dati: BFM (500) e ICT-5000 dalle viste con operatori ad area unitaria in uso; 50.000 ICT nuove dai 200
# shard di sola geometria (pre-pass su /tmp dentro il trainer). Split esplicito split_scale.json
# (held-out = heldout_frozen.json, politica joint_compare: congiunto 1019532 + 100 BFM standard),
# guardia attiva. GT estesa alla scala di quella in uso (--gt-keep-scale).
#
# Passi e lr: ESATTAMENTE quelli del congiunto 1019532, misurati sul suo log: 120 epoche da 879 passi
# (tqdm /879; batch a dominio singolo: 392/5 + 4008/5 = 78 + 801) = 105.480 passi; lr 1e-4 fino al
# passo 81.747 (= 93 x 879: la riga "Epoch 093" stampa gia' 5e-5, ma il lr e' letto DOPO
# scheduler.step, train_runner.py:1735-1736, quindi l'epoca 93 e' a 1e-4) e 5e-5 dopo. Lo schedule e'
# fissato in passi (--lr-steps): ReduceLROnPlateau sulla loss di epoche corte e col dente di sega dei
# cambi di blocco dimezzerebbe a caso. Epoca di S=293 = 879/3 passi: eval online ogni 6 epoche = 1.758
# passi come il congiunto (eval_every 2 da 879), 360 epoche esatte; un blocco deve contenere almeno i
# 267 x 5 = 1.335 soggetti ICT di un'epoca (nessuno due volte), quindi 40 blocchi da 1.350 (9 epoche
# ciascuno). BFM (392 soggetti, gli stessi del congiunto) residente in ogni blocco con 26 passi su 293 =
# 78/879, la quota del congiunto. Eval online sui 16 soggetti del congiunto (split_scale.json), quindi
# i best_by_* sono scelti come i suoi. Espressioni: rexpr1-4 -> rexprA, rexpr5-8 -> rexprB = 25%.
#
# Frame (verdetto del PI in corso): WBES_DS_CANON e WBES_DS_AUG, JSON, spenti di default ("null").
#   WBES_DS_CANON='{"bfm": {"R": [[1,0,0],[0,-1,0],[0,0,-1]], "flip_faces": true}}'
#       Rx(180) e inversione delle facce: BFM ha naso verso -z, alto verso -y e normali verso l'interno
#       (misurato, aau/data_scale/check_canon.py). Applicato PRIMA degli operatori, quindi BFM passa
#       dalla geometria (datasets/REMESH/npz_data_topo_500) e non dalla vista.
#   WBES_DS_AUG='{"rot_deg": 180, "reflect_p": 0.5}'     rotazioni/riflessioni dei vertici di training.
#
# CPU: 8 thread di torch al training e 22 processi al pre-pass del blocco successivo (32 core): senza
# limite il pre-pass portava il training da 0.75 a ~35 s/passo (smoke S1). Requeue: --no-requeue, e una
# run dir che contiene gia' risultati non viene mai riusata (nuova cartella _restartN / _rerunN).
#
# Memoria, dal cgroup (non da MaxRSS: la memoria pinned non vi compare, e qui e' spenta) [misure: smoke
# S4 job 1056277, proiezione esatta cache_budget.py blocks job 1056278]:
#   cache del blocco, ESATTA dagli header/indice: max 220.4 GiB sui 40 blocchi (blocco 30), min 218.0
#   anon fisso (S4: anon 85.1 GiB = cache 41.4 + GT float64 22.9 + resto 20.8)         ~ 46 GiB
#   /tmp (shmem) del blocco successivo durante il suo pre-pass, 5.67 MiB/mesh x 17.500  ~ 97 GiB
#   -> ~363 GiB non recuperabili + page cache (recuperabile, 21.5 GiB in S4); --mem 430G = +18%.
# Tempo [stima]: pre-pass di un blocco ~44 min con 22 processi; S4 (cache non pinned): 1.26-1.91
# s/passo col pre-pass in corso, 0.73-1.11 senza (S3 pinned: 0.70-0.79); ~65 min per blocco, ~44 h.
# train_runner non ha resume: 72 h.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv
aau_require_diffusion_net
source "$AAU_DIR/data_scale/recipe_v1.sh"

SEED="${WBES_DS_SEED:-1234}"
T="${WBES_DS_STEPS:-105480}"
S="${WBES_DS_STEPS_PER_EPOCH:-293}"
EVAL_EVERY="${WBES_DS_EVAL_EVERY:-6}"
LR_STEPS="${WBES_DS_LR_STEPS:-81747:5e-5}"
NB="${WBES_DS_BLOCKS:-40}"
BFM_SHARE="${WBES_DS_BFM_SHARE:-0.0887372}"   # 78/879
TRAIN_THREADS="${WBES_DS_TRAIN_THREADS:-8}"
PREPASS_PROC="${WBES_DS_PREPASS_PROC:-22}"
CACHE_MAX_GB="${WBES_DS_CACHE_MAX_GB:-240}"   # GiB; cache esatta massima 220.4 (blocco 30)
CANON="${WBES_DS_CANON:-null}"
AUG="${WBES_DS_AUG:-null}"
EPOCHS=$(( (T + S - 1) / S ))
D="$WBES_ROOT/datasets"
RUN_NAME="${WBES_RUN_NAME:-scale_bfm_ict55k_s${SEED}}"
RUNS_ROOT="${WBES_RUNS_ROOT:-$AAU_RUNS/data_scale_runs/${RUN_NAME}_${SLURM_JOB_ID:-manual}}"
# un nuovo tentativo (requeue o rilancio a mano con la stessa WBES_RUNS_ROOT) non deve sovrascrivere
if (( ${SLURM_RESTART_COUNT:-0} > 0 )); then
    RUNS_ROOT="${RUNS_ROOT}_restart${SLURM_RESTART_COUNT}"
fi
if [[ -e "$RUNS_ROOT/launch.txt" || -e "$RUNS_ROOT/train.log" ]]; then
    n=1; while [[ -e "${RUNS_ROOT}_rerun$n" ]]; do n=$(( n + 1 )); done
    echo "[scale] $RUNS_ROOT contiene gia' un run: uso ${RUNS_ROOT}_rerun$n" >&2
    RUNS_ROOT="${RUNS_ROOT}_rerun$n"
fi
echo "$RUNS_ROOT" > "$AAU_LOGS/${SLURM_JOB_NAME:-wbes-ds-train-scale}-${SLURM_JOB_ID:-manual}.runs_root"
STAGE="/tmp/${SLURM_JOB_ID:-manual}"
RUN_TMP="$STAGE/run"
LOG_TMP="$STAGE/train.log"
mkdir -p "$RUNS_ROOT" "$RUN_TMP"

# come train_joint_E.sbatch: la run dir vive su /tmp, ogni 10 minuti si copia nella home con un sync che
# non puo' far fallire il job; a fine job il sync si ritenta per 30 minuti
sync_home() {
    [[ -f "$LOG_TMP" ]] && { tr '\r' '\n' < "$LOG_TMP" | sed 's/\(Epoch [0-9]\{3\} |\)/\n\1/g' \
        | grep -v '%|.*\(it/s\|s/it\)\|^[[:space:]]*$' > "$RUNS_ROOT/train.log.part" \
        && mv -f "$RUNS_ROOT/train.log.part" "$RUNS_ROOT/train.log"; } 2> /dev/null
    rsync -a --exclude 'stage/' "$RUN_TMP/" "$RUNS_ROOT/" 2> /dev/null
    # i log del pre-pass stanno nello stage, escluso sopra e cancellato in uscita: copiati a parte,
    # anche quando il job fallisce (cleanup -> final_sync -> sync_home prima di rm)
    mkdir -p "$RUNS_ROOT/prepass_logs" && cp -f "$STAGE"/stage/*.prepass.log "$STAGE"/stage/*.subjects.txt \
        "$RUNS_ROOT/prepass_logs/" 2> /dev/null
    return 0
}
final_sync() {
    local i
    for (( i = 0; i < 30; i++ )); do
        sync_home && { echo "[scale] run dir copiata in $RUNS_ROOT" || true; return 0; }
        sleep 60
    done
    echo "ERRORE: sync finale verso $RUNS_ROOT fallito per 30 minuti" >&2 || true
    return 1
}
SYNC_PID=""
cleanup() {
    [[ -n "$SYNC_PID" ]] && kill "$SYNC_PID" 2> /dev/null
    final_sync || true
    rm -rf "$STAGE"
}
trap cleanup EXIT

if [[ "$CANON" == "null" ]]; then
    BFM_SRC="\"views\": [\"$D/REMESH/npz_data_topo_500_withops_areanorm\", \"$D/ICT/train_ready/npz_withops\"], \"geom_dirs\": []"
else
    BFM_SRC="\"views\": [\"$D/ICT/train_ready/npz_withops\"], \"geom_dirs\": [\"$D/REMESH/npz_data_topo_500\"]"
fi
TARS=$(find "$D/ICT_SCALE/shards" -maxdepth 1 -name 'shard_*.tar' | sort | sed 's/.*/"&"/' | paste -sd, -)
N_TARS=$(find "$D/ICT_SCALE/shards" -maxdepth 1 -name 'shard_*.tar' | wc -l)
[[ "$N_TARS" == 200 ]] || { echo "ERRORE: $N_TARS shard, attesi 200" >&2; exit 1; }
[[ -f "$D/ICT_SCALE/shards/index.npz" ]] || { echo "ERRORE: indice dei tar assente (cache_budget.py index)" >&2; exit 1; }
cat > "$RUNS_ROOT/spec.json" <<JSON
{$BFM_SRC, "tars": [$TARS], "tar_index": "$D/ICT_SCALE/shards/index.npz",
 "labels": null,
 "label_groups": {"rexprA": ["rexpr1","rexpr2","rexpr3","rexpr4"], "rexprB": ["rexpr5","rexpr6","rexpr7","rexpr8"]},
 "canon": $CANON, "aug": $AUG,
 "pin_domains": ["bfm"], "domain_step_share": {"bfm": $BFM_SHARE},
 "convention": "areanorm", "n_blocks": $NB, "block_seed": $SEED}
JSON

echo "[scale] host=$(hostname) job=${SLURM_JOB_ID:-none} T=$T S=$S epoche=$EPOCHS blocchi=$NB lr=$LR_STEPS canon=$CANON aug=$AUG runs_root=$RUNS_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

CMD=(
  v2_work/fastio/train_steps.py --total-steps "$T" --steps-per-epoch "$S"
  --split-json "$AAU_DIR/data_scale/split_scale.json"
  --frozen-heldout "$AAU_DIR/data_scale/heldout_frozen.json"
  --data-spec "$RUNS_ROOT/spec.json" --stage-root "$STAGE/stage" --prepass-proc "$PREPASS_PROC"
  --train-threads "$TRAIN_THREADS"
  --domain-blocked --eval_domain bfm --gt-keep-scale --lr-steps "$LR_STEPS"
  --cache-residency ram --cache-workers 16 --cache-max-gb "$CACHE_MAX_GB" --frame current
  "${RECIPE_V1[@]}"
  --epochs "$EPOCHS" --eval_every "$EVAL_EVERY" --save_every "$EPOCHS" --seed "$SEED"
  --data_dir "$STAGE/stage" --dist_npz "$D/ICT_SCALE/gt_joint_bfm_ict.npz" --runs_root "$RUN_TMP"
)
printf '%q ' "${CMD[@]}" > "$RUNS_ROOT/launch.txt"; echo >> "$RUNS_ROOT/launch.txt"

( while sleep 600; do sync_home || true; done ) &
SYNC_PID=$!
OMP_NUM_THREADS="$TRAIN_THREADS" MKL_NUM_THREADS="$TRAIN_THREADS" "$AAU_DIR/run.sh" "${CMD[@]}" > "$LOG_TMP" 2>&1
kill "$SYNC_PID" 2> /dev/null || true
SYNC_PID=""
final_sync
grep -E "^\[steps\] passi eseguiti|^Best" "$RUNS_ROOT/train.log" || true
echo "[scale] OK"
