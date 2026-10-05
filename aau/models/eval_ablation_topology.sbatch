#!/usr/bin/env bash
#SBATCH --job-name=wbes-abl-evalbfm
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#
# Eval in-domain BFM delle ablazioni v3, adattata da aau/eval_frame_topology.sbatch: stessi 100
# held-out e stesse coppie, ma con aau/models/eval_cells.py (gruppi dell'autore + 30 celle
# ordinate del repo, per il margine del criterio) lanciato da aau/models/eval_ablation.py, che
# passa al modello frame, token di taglia e operatori su cui il braccio e' stato addestrato.
#
#   WBES_CKPT=aau/runs/abl3_B_s1234_<job> aau/submit.sh aau/models/eval_ablation_topology.sbatch
#
# Tutto viene da <runs_root>/ablation_arm.txt (aau/models/train_ablation.sbatch): frame,
# cartella degli operatori (data dir dell'eval), tabella dei token. Senza quel file il
# checkpoint e' il controllo: frame da wrapper_launch.txt, operatori standard, niente token.
# Il controllo passa dallo stesso script, cosi' i Delta sono fra numeri dello stesso codice.
#
# Output stabile: aau/runs/ablations_v3/eval_bfm/<runs_root>/<runs_root>.json, .done solo se
# python esce 0, eval_key.txt con lo sha1 del checkpoint (un best_*.pth riscritto fa fallire).
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

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

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

CKPT_REAL="$(realpath "$WBES_CKPT")"
TRAIN_RUNS_ROOT="$(dirname "$(dirname "$(dirname "$CKPT_REAL")")")"
EVAL_RUN_NAME="$(basename "$TRAIN_RUNS_ROOT")"

if [[ -f "$TRAIN_RUNS_ROOT/ablation_arm.txt" ]]; then
    ARM="$(sed -n 's/^arm=//p' "$TRAIN_RUNS_ROOT/ablation_arm.txt")"
    FRAME="$(sed -n 's/^frame=//p' "$TRAIN_RUNS_ROOT/ablation_arm.txt")"
    export WBES_DATA_DIR="$(sed -n 's/^ops_dir_train=//p' "$TRAIN_RUNS_ROOT/ablation_arm.txt")"
    TOKEN_JSON="$(sed -n 's/^size_token_json=//p' "$TRAIN_RUNS_ROOT/ablation_arm.txt")"
else
    ARM=controllo
    FRAME="$(grep -oE -- '--frame [a-z]+' "$TRAIN_RUNS_ROOT/wrapper_launch.txt" 2>/dev/null | awk '{print $2}' | tail -1 || true)"
    FRAME="${FRAME:-current}"
    TOKEN_JSON=""
fi
# Il frame del training e quello passato al forward devono coincidere: lo controlla anche qui,
# perche' un frame sbagliato non cambia nessuna forma di tensore.
_wf="$(grep -oE -- '--frame [a-z]+' "$TRAIN_RUNS_ROOT/wrapper_launch.txt" 2>/dev/null | awk '{print $2}' | tail -1 || true)"
if [[ "${_wf:-current}" != "$FRAME" ]]; then
    echo "ERRORE: ablation_arm.txt dice frame=$FRAME, wrapper_launch.txt --frame ${_wf:-current}" >&2
    exit 2
fi
LAUNCH=(aau/models/eval_ablation.py --frame "$FRAME")
[[ -n "$TOKEN_JSON" ]] && LAUNCH+=(--size-token-json "$TOKEN_JSON")
LAUNCH+=(--)

DATA_REAL="$(realpath "$WBES_DATA_DIR")"
DIST_REAL="$(realpath "$WBES_DIST_NPZ")"
OUT="${WBES_ABL_EVAL_OUT:-$AAU_RUNS/ablations_v3/eval_bfm/$EVAL_RUN_NAME}"
mkdir -p "$OUT"

KEY="$(printf 'ckpt=%s\nckpt_sha1=%s\ndata_dir=%s\ndist_npz=%s\narm=%s\nframe=%s\nsize_token_json=%s\n' \
    "$CKPT_REAL" "$(sha1sum "$CKPT_REAL" | cut -c1-40)" "$DATA_REAL" "$DIST_REAL" "$ARM" "$FRAME" "$TOKEN_JSON")"
if [[ -f "$OUT/eval_key.txt" ]]; then
    if [[ "$(cat "$OUT/eval_key.txt")" != "$KEY" ]]; then
        echo "ERRORE: $OUT contiene gia' una eval di un'altra combinazione." >&2
        diff <(cat "$OUT/eval_key.txt") <(printf '%s\n' "$KEY") >&2 || true
        exit 1
    fi
else
    printf '%s\n' "$KEY" > "$OUT/eval_key.txt"
fi
if [[ -f "$OUT/.done" ]]; then
    echo "[$(timestamp)] eval gia' completa in $OUT:"; sed 's/^/    /' "$OUT/.done"; exit 0
fi

echo "[$(timestamp)] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[$(timestamp)] arm=$ARM frame=$FRAME token=${TOKEN_JSON:-no}"
echo "[$(timestamp)] ckpt=$CKPT_REAL"
echo "[$(timestamp)] data_dir=$DATA_REAL"
echo "[$(timestamp)] out=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

EXPECT=()
[[ -n "${WBES_EXPECT_SEED:-}" ]] && EXPECT=(--expect-seed "$WBES_EXPECT_SEED")
"$AAU_DIR/run.sh" "$AAU_DIR/eval_frame_split_info.py" \
  --checkpoint "$WBES_CKPT" --data-dir "$WBES_DATA_DIR" --dist-npz "$WBES_DIST_NPZ" \
  --n-subjects 100 "${EXPECT[@]}" 2>&1 | tee "$OUT/split_info.log"

"$AAU_DIR/run.sh" "${LAUNCH[@]}" aau/models/eval_cells.py \
  --checkpoint "$WBES_CKPT" --data-dir "$WBES_DATA_DIR" --dist-npz "$WBES_DIST_NPZ" \
  --n-subjects 100 --device cuda --tag "$EVAL_RUN_NAME" --out-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"

printf 'completed=%s\njob=%s\nhost=%s\nresult=%s\n' \
    "$(timestamp)" "${SLURM_JOB_ID:-none}" "$(hostname)" "$OUT/$EVAL_RUN_NAME.json" > "$OUT/.done"
echo "[$(timestamp)] eval completata, scritta $OUT/.done"
