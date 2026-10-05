#!/usr/bin/env bash
#SBATCH --job-name=wbes-ws3b-recon
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%A_%a.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%A_%a.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=0-12%3
#
# WS3b, passo 1: i tre metodi su tutte le immagini frontali di Multiface con GT.
#
#   python3 aau/recon/ws3b_stage_images.py           # prima, sul frontend: symlink+manifest
#   aau/submit.sh recon/ws3b_recon.sbatch
#
# Un task per capture (13 soggetti x 240 immagini x 3 metodi).  Le capture sono
# indipendenti e ognuna dura pochi minuti: l'array serve piu' alla ripartenza che alla
# velocita'.  ``ws3b_stage_images.py --pending-out`` costruisce su /tmp un albero coi soli
# frame la cui mesh non c'e' ancora, quindi risottomettere l'array e' un no-op se e' tutto
# fatto e rifa' solo i buchi altrimenti.
#
# T4 come nello smoke: le L40S servono al training, e qui la GPU fa girare due reti da
# 120x120 pixel.  Misurato nel sondaggio (job 1019484) su immagini Multiface vere
# 2048x1334: 3DDFA_V2 0.174 s/immagine su T4 e 0.131 s su CPU -- i metodi sono CPU-bound
# (rilettura del png e FaceBoxes sull'immagine intera), la GPU non cambia il conto.  PRNet
# gira comunque su CPU (tensorflow-cpu).  Stima: 240 x (0.17 + 0.25 + 0.6) ~ 4 min per
# capture piu' il caricamento dei tre modelli; --time=1h30 e' un fattore 10 di margine.
#
# %3 e non di piu': il tetto e' 12 GPU per utente e il training ne tiene gia' 8.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

OUT="${WBES_WS3B_OUT:-$AAU_RUNS/multiface_ws3b}"
export WBES_WS3B_OUT="$OUT"

mapfile -t CAPTURES < <(find "${WBES_MF_FULL_DIR:-$WBES_ROOT/datasets/Multiface/full}" \
    -maxdepth 1 -mindepth 1 -type d -name 'm--*' -printf '%f\n' | sort)
CAPTURE="${CAPTURES[${SLURM_ARRAY_TASK_ID:-0}]}"

STAGE="${TMPDIR:-/tmp}/ws3b-${SLURM_JOB_ID:-$$}"
trap 'rm -rf "$STAGE"' EXIT

echo "[ws3b-recon] host=$(hostname) job=${SLURM_JOB_ID:-none} task=${SLURM_ARRAY_TASK_ID:-0}"
echo "[ws3b-recon] capture=$CAPTURE out=$OUT"
nvidia-smi --query-gpu=name --format=csv,noheader || true

for METHOD in 3ddfa_v2 synergynet prnet; do
    PENDING="$STAGE/$METHOD"
    python3 aau/recon/ws3b_stage_images.py --out-root "$OUT" \
        --pending-out "$PENDING" --method "$METHOD" --capture "$CAPTURE"
    [[ -d "$PENDING/$CAPTURE" ]] || continue

    t0=$(date +%s)
    case "$METHOD" in
        prnet) "$AAU_DIR/recon/run_recon.sh" prnet aau/recon/prnet_run.py \
                   --images "$PENDING/$CAPTURE" --out "$OUT/recon/$METHOD" ;;
        *)     "$AAU_DIR/recon/run_recon.sh" ddfa "aau/recon/${METHOD/3ddfa_v2/tddfa_v2}_run.py" \
                   --images "$PENDING/$CAPTURE" --out "$OUT/recon/$METHOD" --device cuda ;;
    esac
    echo "[ws3b-recon] $METHOD $CAPTURE: $(( $(date +%s) - t0 )) s"
done

echo "[ws3b-recon] mesh presenti per $CAPTURE:"
for METHOD in 3ddfa_v2 synergynet prnet; do
    echo "  $METHOD: $(find "$OUT/recon/$METHOD" -name "$(echo "$CAPTURE" | sed 's/^m--[0-9]*--[0-9]*--//;s/--GHS$//')__*.npz" | wc -l)"
done
echo "[ws3b-recon] OK"
