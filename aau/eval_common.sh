#!/usr/bin/env bash
# Parte comune dei tre stage di robustness eval (porting di
# scripts/run_mixed_xtopo_robustness_eval.sh). Sourcealo da eval_<stage>.sbatch.
#
# Perche' tre job invece di uno: il log originale della stessa pipeline
# (.../dn_mixed_topology_v1/robustness_queue.log) misura 28h10m in tutto, spesi molto
# male in un unico blocco — ranking 5h13m, topology breakdown 1h03m, sigma sweep 21h54m.
# Un solo job avrebbe dovuto chiedere 36 h anche per lo stage da un'ora, e una morte per
# walltime avrebbe buttato via anche gli stage gia' finiti. I tre stage sono indipendenti
# fra loro e possono girare in parallelo.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv
aau_require_diffusion_net
aau_require_checkpoint
aau_require_dataset

PERT="$WBES_ROOT/face_embedding/gt_encdec/remeshing/intrinsic/perturbated"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

# Output dir STABILE (niente job id): due sottomissioni dello stesso stage riscrivono lo
# stesso posto, ed e' quello che vogliamo — vedi lo skip-if-exists piu' sotto.
#
# La chiave NON puo' essere il basename del checkpoint: train_runner.py scrive sempre
# <run_dir>/checkpoints/best_by_xtopo_mesh_clean.pth, cioe' lo stesso nome del checkpoint
# v1 pubblicato. Valutare un modello nuovo troverebbe la sentinella del v1 e uscirebbe 0,
# spacciando i numeri del v1 per nuovi. Chiave = nome della run dir che contiene
# checkpoints/, piu' un hash dei tre input (checkpoint, dataset, matrice GT), cosi'
# cambiare dataset o matrice non riusa i risultati vecchi.
CKPT_REAL="$(realpath "$WBES_CKPT")"
DATA_REAL="$(realpath "$WBES_DATA_DIR")"
DIST_REAL="$(realpath "$WBES_DIST_NPZ")"

_ckpt_parent="$(dirname "$CKPT_REAL")"
if [[ "$(basename "$_ckpt_parent")" == "checkpoints" ]]; then
    EVAL_RUN_NAME="$(basename "$(dirname "$_ckpt_parent")")"
else
    EVAL_RUN_NAME="$(basename "$_ckpt_parent")"
fi
unset _ckpt_parent

EVAL_HASH_INPUT="$(printf '%s|%s|%s' "$CKPT_REAL" "$DATA_REAL" "$DIST_REAL")"
# WBES_EVAL_SEED entra nell'hash: senza override il seed dello split e' una funzione del
# checkpoint (sta nel suo config.json) e l'hash lo copre gia', con l'override no, e due
# split diversi dello stesso checkpoint finirebbero nella stessa out dir. Quando la
# variabile non c'e' l'hash resta identico a prima, cioe' le out dir esistenti non si
# spostano.
if [[ -n "${WBES_EVAL_SEED:-}" ]]; then
    EVAL_HASH_INPUT="$EVAL_HASH_INPUT|seed=$WBES_EVAL_SEED"
fi
EVAL_HASH="$(printf '%s' "$EVAL_HASH_INPUT" | sha1sum | cut -c1-8)"
OUT_ROOT="${WBES_EVAL_OUT:-$AAU_RUNS/eval_${EVAL_RUN_NAME}_${EVAL_HASH}}"

# aau_eval_check_key
# Scrive (o verifica) OUT_ROOT/eval_key.txt con i tre input in chiaro. Con WBES_EVAL_OUT
# forzata a mano l'hash non protegge piu' niente: qui si scopre il riuso sbagliato.
aau_eval_check_key() {
    local key_file="$OUT_ROOT/eval_key.txt"
    local expected
    expected="$(printf 'ckpt=%s\ndata_dir=%s\ndist_npz=%s\n' "$CKPT_REAL" "$DATA_REAL" "$DIST_REAL")"
    # Riga in piu' solo con l'override, per non invalidare gli eval_key.txt gia' scritti.
    if [[ -n "${WBES_EVAL_SEED:-}" ]]; then
        expected="$expected"$'\n'"eval_seed=$WBES_EVAL_SEED"
    fi

    if [[ -f "$key_file" ]]; then
        if [[ "$(cat "$key_file")" != "$expected" ]]; then
            echo "ERRORE: $OUT_ROOT contiene gia' risultati di un'altra combinazione." >&2
            echo "--- eval_key.txt esistente ---" >&2
            cat "$key_file" >&2
            echo "--- richiesto ora ---" >&2
            printf '%s\n' "$expected" >&2
            echo "  Usa una WBES_EVAL_OUT diversa, o cancella la dir se i vecchi numeri non servono." >&2
            return 1
        fi
    else
        printf '%s\n' "$expected" > "$key_file"
    fi
}

# Argomenti comuni, da run_mixed_xtopo_robustness_eval.sh. UNICA differenza: --seed.
#
# Lo script originale passava --seed 1234 perche' valutava solo il modello v1, che e'
# addestrato con seed 1234. Qui i checkpoint sono tre (seed 1234/2345/3456) e il seed NON e'
# un dettaglio di riproducibilita': rebuild_subject_split (robustness/data_utils.py:76)
# estrae i 100 soggetti held-out con np.random.default_rng(seed), lo stesso seed che il
# training ha usato per tenerli fuori. Con --seed 1234 fisso i modelli 2345 e 3456 venivano
# valutati sullo split del 1234, cioe' su soggetti che avevano visto in training: 79-86% di
# sovrapposizione, misurato (job 1019693). Senza --seed gli script posthoc prendono il seed
# dal config.json del checkpoint (`if cli_args.seed >= 0` in posthoc_runner.py:253 e in
# compare_model_vs_chamfer_rankings.py:216), che e' quello giusto per costruzione.
# WBES_EVAL_SEED forza un seed a mano quando serve davvero, p.es. per valutare piu' modelli
# sulla stessa selezione di soggetti su un dataset che nessuno dei due ha visto (zero-shot).
common_args=(
  --model_path "$WBES_CKPT"
  --checkpoint_selector best_by_clean
  --data_dir "$WBES_DATA_DIR"
  --dist_npz "$WBES_DIST_NPZ"
  --subject_split eval
  --eval_fraction 0.2
  --max_subjects 500
  --max_meshes_per_subject_eval 10
  --preload_workers 8
  --pair_mode cross_topology
  --rigid_rot_deg 20
  --rigid_trans_scale 0.05
  --rigid_rot_deg_min 1
  --rigid_trans_scale_min 0.002
  --chamfer_batch_pairs 256
  --chamfer_cache_verts force
  --chamfer_cache_verts_max_mb 4096
)

if [[ -n "${WBES_EVAL_SEED:-}" ]]; then
    common_args+=(--seed "$WBES_EVAL_SEED")
fi

# aau_eval_begin <stage>
# Prepara la dir dello stage ed esce 0 se lo stage e' gia' andato a buon fine: con job da
# 5-22 ore capita di risottomettere per sbaglio, e ricalcolare 22 ore non serve a nessuno.
# STAGE_OUT resta definita per il chiamante.
#
# La sentinella e' <stage>/.done e la scrive aau_eval_done DOPO che python e' uscito con 0,
# non un artefatto dello script python: ranking_summary.json e' il PRIMO dei tre file che
# lo script scrive (compare_model_vs_chamfer_rankings.py:790, poi csv e md), quindi un kill
# a meta' scrittura lascerebbe uno stage "completo" ma monco.
aau_eval_begin() {
    local stage="$1"
    STAGE_OUT="$OUT_ROOT/$stage"

    mkdir -p "$OUT_ROOT"
    aau_eval_check_key

    if [[ -f "$STAGE_OUT/.done" ]]; then
        echo "[$(timestamp)] stage '$stage' gia' completo:"
        sed 's/^/    /' "$STAGE_OUT/.done"
        echo "[$(timestamp)] niente da fare (cancella $STAGE_OUT per rifarlo)"
        exit 0
    fi

    mkdir -p "$STAGE_OUT"
    echo "[$(timestamp)] host=$(hostname) job=${SLURM_JOB_ID:-none} stage=$stage"
    echo "[$(timestamp)] out=$STAGE_OUT"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

    # Seed e soggetti valutati, scritti nel log PRIMA delle ore di GPU: e' l'unico posto
    # dove si vede che il seed e' quello del checkpoint e non quello di un altro modello.
    # Non e' bloccante: lo split vero lo ricalcola lo script python, questo e' solo il
    # referto, e una eval buona non deve morire perche' il referto non e' riuscito.
    "$AAU_DIR/run.sh" "$AAU_DIR/eval_split_info.py" "${common_args[@]}" \
        || echo "[$(timestamp)] ATTENZIONE: eval_split_info.py fallito, seed non verificato"
}

# aau_eval_done <stage>
# Da chiamare solo dopo il comando python: con set -e e pipefail, se python fallisce lo
# script muore prima di arrivare qui e la sentinella non viene scritta.
aau_eval_done() {
    local stage="$1"
    printf 'stage=%s\ncompleted=%s\njob=%s\nhost=%s\n' \
        "$stage" "$(timestamp)" "${SLURM_JOB_ID:-none}" "$(hostname)" > "$OUT_ROOT/$stage/.done"
    echo "[$(timestamp)] stage '$stage' completato, scritta $OUT_ROOT/$stage/.done"
}
