#!/usr/bin/env bash
#SBATCH --job-name=wbes-eval-frame
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#
# Eval per gruppo di topologia (crop / noisy / resample / all) con lo script dell'autore,
# v2_work/potential/eval_by_topology.py, l'unico che passa il frame di input al modello.
# eval_topology.sbatch no: compare_model_vs_chamfer_topology_breakdown.py da' al modello
# sample["verts"] cosi' come escono dal loader (frame maxabs), quindi un checkpoint addestrato
# con train_fast.py --frame rms verrebbe valutato nel frame sbagliato, senza errori.
#
#   WBES_CKPT=aau/runs/remesh_v1recipe_rms_s2345_1054482/<rundir>/checkpoints/best_by_xtopo_mesh_clean.pth \
#     WBES_FRAME=rms aau/submit.sh eval_frame_topology.sbatch
#
# WBES_FRAME: maxabs (= --frame current, il loader) | rms. Gli argomenti sono quelli con cui
# l'autore ha prodotto v2_work/potential/results/pot_*.json (auto_eval*.sh):
# --use-eval-split --n-subjects 100. Il seed dello split NON si passa: lo script lo legge dagli
# args del checkpoint; --seed resta al default 1234 come nei run dell'autore e decide solo il
# sottocampionamento a 100, che con 100 held-out non scatta.
#
# Output stabile: aau/runs/eval_frame/<run>_<frame>/<run>_<frame>.json, con <run> = la dir
# del training sotto aau/runs (remesh_v1recipe_rms_s2345_1054482) se il checkpoint sta li',
# altrimenti la dir che contiene checkpoints/ (per il v1: mixed_xtopo_rank0p5_id0p25_bs5_best).
# Non il nome col fingerprint: il frame non entra nel fingerprint, quindi braccio rms e
# controllo dello stesso seed hanno la stessa run dir (...seed2345__1b40106f).
# Skip-if-exists su .done (scritta solo se python esce 0); eval_key.txt contiene anche lo
# sha1 del checkpoint, cosi' un best_*.pth riscritto da un training ancora in corso fa fallire
# la rivalutazione invece di riusare in silenzio i numeri vecchi.
#
# Su un T4 l'eval e' 600 forward (100 soggetti x 6 topologie) piu' la lettura degli npz da
# CephFS: --time=1h e' largo. Un'altra GPU: aau/submit.sh eval_frame_topology.sbatch --gres=gpu:a10:1
#
# Bracci del pilota pot (aau/train_pilot_pot.sbatch): se il runs_root del checkpoint contiene
# pilot_arm.txt, il modello passa da aau/models/eval_pilot.py, con gli operatori su cui e'
# stato addestrato (masked: la cartella del pozzo come data dir; dual: standard + pozzo).
# Nessuna variabile da passare, e la riga pilot= entra in eval_key.txt solo per questi run.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

# WBES_CKPT puo' essere anche il runs_root di un training (aau/runs/remesh_v1recipe_..._<job>):
# serve alla coda per i training ancora in attesa, la cui run dir col fingerprint non esiste
# ancora. Si risolve qui, all'avvio, nell'unico <rundir>/checkpoints/best_by_xtopo_mesh_clean.pth.
if [[ -d "$WBES_CKPT" ]]; then
    _matches=("$WBES_CKPT"/*/checkpoints/best_by_xtopo_mesh_clean.pth)
    if [[ ${#_matches[@]} -ne 1 || ! -f "${_matches[0]}" ]]; then
        echo "ERRORE: in $WBES_CKPT attesa una sola */checkpoints/best_by_xtopo_mesh_clean.pth, trovate: ${_matches[*]}" >&2
        exit 2
    fi
    export WBES_CKPT="${_matches[0]}"
    unset _matches
fi

aau_require_venv
aau_require_diffusion_net
aau_require_checkpoint
aau_require_dataset

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

FRAME="${WBES_FRAME:-maxabs}"
case "$FRAME" in
    maxabs) AUTHOR_FRAME=current ;;
    rms)    AUTHOR_FRAME=rms ;;
    *) echo "ERRORE: WBES_FRAME=$FRAME, attesi maxabs|rms" >&2; exit 2 ;;
esac

CKPT_REAL="$(realpath "$WBES_CKPT")"
DATA_REAL="$(realpath "$WBES_DATA_DIR")"
DIST_REAL="$(realpath "$WBES_DIST_NPZ")"

_ckpt_parent="$(dirname "$CKPT_REAL")"
if [[ "$(basename "$_ckpt_parent")" == "checkpoints" ]]; then
    TRAIN_RUN_DIR="$(dirname "$_ckpt_parent")"
else
    TRAIN_RUN_DIR="$_ckpt_parent"
fi
unset _ckpt_parent
TRAIN_RUNS_ROOT="$(dirname "$TRAIN_RUN_DIR")"
if [[ "$(dirname "$TRAIN_RUNS_ROOT")" == "$(realpath "$AAU_RUNS")" ]]; then
    EVAL_RUN_NAME="$(basename "$TRAIN_RUNS_ROOT")"
else
    EVAL_RUN_NAME="$(basename "$TRAIN_RUN_DIR")"
fi

# Frame del training. Un frame sbagliato non cambia nessuna forma di tensore e non da' errori
# (docstring di eval_by_topology.py): lo controlliamo noi. train_remesh_frame.sbatch scrive il
# comando del wrapper in <runs_root>/wrapper_launch.txt, unico posto dove --frame compare
# (train_fast.py lo toglie da sys.argv e non entra in config.json); senza quel file il
# checkpoint viene dal trainer di ricerca, cioe' dal loader maxabs. WBES_FRAME_MISMATCH_OK=1
# per valutare apposta un modello nel frame in cui non e' stato addestrato.
TRAINED_FRAME=current
if [[ -f "$TRAIN_RUNS_ROOT/wrapper_launch.txt" ]]; then
    _f="$(grep -oE -- '--frame [a-z]+' "$TRAIN_RUNS_ROOT/wrapper_launch.txt" | awk '{print $2}' | tail -1 || true)"
    TRAINED_FRAME="${_f:-current}"
    unset _f
fi
if [[ "$TRAINED_FRAME" != "$AUTHOR_FRAME" && "${WBES_FRAME_MISMATCH_OK:-0}" != "1" ]]; then
    echo "ERRORE: il checkpoint e' addestrato con --frame $TRAINED_FRAME, richiesto WBES_FRAME=$FRAME (--frame $AUTHOR_FRAME)" >&2
    echo "  checkpoint: $CKPT_REAL" >&2
    echo "  Se e' voluto: WBES_FRAME_MISMATCH_OK=1" >&2
    exit 2
fi

# Obbligatorio, non opzionale: un checkpoint col pooling mascherato si carica con strict=True
# anche nel modello NON mascherato, e verrebbe valutato senza maschera senza nessun errore.
PILOT=()
PILOT_KEY=""
if [[ -f "$TRAIN_RUNS_ROOT/pilot_arm.txt" ]]; then
    PILOT_ARM="$(sed -n 's/^arm=//p' "$TRAIN_RUNS_ROOT/pilot_arm.txt")"
    POT_TRAIN="$(sed -n 's/^pot_dir_train=//p' "$TRAIN_RUNS_ROOT/pilot_arm.txt")"
    case "$PILOT_ARM" in
        masked) export WBES_DATA_DIR="$POT_TRAIN"
                PILOT=(aau/models/eval_pilot.py --pilot-arm masked --) ;;
        dual)   PILOT=(aau/models/eval_pilot.py --pilot-arm dual --dual-ops-dir "$POT_TRAIN" --) ;;
        *) echo "ERRORE: arm '$PILOT_ARM' in $TRAIN_RUNS_ROOT/pilot_arm.txt" >&2; exit 2 ;;
    esac
    DATA_REAL="$(realpath "$WBES_DATA_DIR")"
    PILOT_KEY="pilot=$PILOT_ARM pot_dir=$(realpath "$POT_TRAIN")"
fi

OUT="$AAU_RUNS/eval_frame/${EVAL_RUN_NAME}_${FRAME}"
TAG="${EVAL_RUN_NAME}_${FRAME}"
mkdir -p "$OUT"

# Come aau_eval_check_key in eval_common.sh, piu' frame e sha1 del checkpoint, e come li'
# PRIMA dello skip: una dir completa il cui checkpoint e' stato poi riscritto fallisce qui
# invece di uscire 0 con i numeri del checkpoint vecchio.
KEY="$(printf 'ckpt=%s\nckpt_sha1=%s\ndata_dir=%s\ndist_npz=%s\nframe=%s\n' \
    "$CKPT_REAL" "$(sha1sum "$CKPT_REAL" | cut -c1-40)" "$DATA_REAL" "$DIST_REAL" "$AUTHOR_FRAME")"
[[ -n "$PILOT_KEY" ]] && KEY="$KEY"$'\n'"$PILOT_KEY"
# WBES_EVAL_PARTIAL: training interrotto prima della fine (p.es. "epoca 108/120, best epoca 106,
# training 1054483 cancellato"). Entra nella chiave e nella .done, cosi' i numeri di un budget
# ridotto non si confondono con quelli a budget pieno.
[[ -n "${WBES_EVAL_PARTIAL:-}" ]] && KEY="$KEY"$'\n'"partial=$WBES_EVAL_PARTIAL"
if [[ -f "$OUT/eval_key.txt" ]]; then
    if [[ "$(cat "$OUT/eval_key.txt")" != "$KEY" ]]; then
        echo "ERRORE: $OUT contiene gia' una eval di un'altra combinazione." >&2
        echo "--- eval_key.txt esistente ---" >&2
        cat "$OUT/eval_key.txt" >&2
        echo "--- richiesto ora ---" >&2
        printf '%s\n' "$KEY" >&2
        echo "  Cancella la dir se i vecchi numeri non servono." >&2
        exit 1
    fi
else
    printf '%s\n' "$KEY" > "$OUT/eval_key.txt"
fi

if [[ -f "$OUT/.done" ]]; then
    echo "[$(timestamp)] eval gia' completa in $OUT:"
    sed 's/^/    /' "$OUT/.done"
    echo "[$(timestamp)] niente da fare (cancella $OUT per rifarla)"
    exit 0
fi

echo "[$(timestamp)] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[$(timestamp)] ckpt=$CKPT_REAL"
echo "[$(timestamp)] frame=$FRAME (--frame $AUTHOR_FRAME, training: $TRAINED_FRAME)"
echo "[$(timestamp)] out=$OUT"
[[ -n "$PILOT_KEY" ]] && echo "[$(timestamp)] $PILOT_KEY data_dir=$WBES_DATA_DIR"
[[ -n "${WBES_EVAL_PARTIAL:-}" ]] && echo "[$(timestamp)] PARZIALE: $WBES_EVAL_PARTIAL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Bloccante, a differenza di aau_eval_begin: eval_by_topology.py, se il checkpoint non ha il
# seed, ripiega su 1234 senza dirlo. WBES_EXPECT_SEED (la coda la passa sempre): fallisce se il
# checkpoint ha un altro seed, cioe' se il braccio non e' quello che la tabella appaiera'.
EXPECT=()
[[ -n "${WBES_EXPECT_SEED:-}" ]] && EXPECT=(--expect-seed "$WBES_EXPECT_SEED")
"$AAU_DIR/run.sh" "$AAU_DIR/eval_frame_split_info.py" \
  --checkpoint "$WBES_CKPT" --data-dir "$WBES_DATA_DIR" --dist-npz "$WBES_DIST_NPZ" \
  --n-subjects 100 "${EXPECT[@]}" 2>&1 | tee "$OUT/split_info.log"

echo "[$(timestamp)] Starting eval_by_topology"
"$AAU_DIR/run.sh" "${PILOT[@]}" v2_work/potential/eval_by_topology.py \
  --checkpoint "$WBES_CKPT" \
  --data-dir "$WBES_DATA_DIR" \
  --dist-npz "$WBES_DIST_NPZ" \
  --use-eval-split --n-subjects 100 \
  --frame "$AUTHOR_FRAME" \
  --device cuda \
  --tag "$TAG" \
  --out-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"

# Con set -e e pipefail si arriva qui solo se python e' uscito 0.
printf 'completed=%s\njob=%s\nhost=%s\nresult=%s\n' \
    "$(timestamp)" "${SLURM_JOB_ID:-none}" "$(hostname)" "$OUT/$TAG.json" > "$OUT/.done"
[[ -n "${WBES_EVAL_PARTIAL:-}" ]] && printf 'partial=%s\n' "$WBES_EVAL_PARTIAL" >> "$OUT/.done"
echo "[$(timestamp)] eval completata, scritta $OUT/.done"
