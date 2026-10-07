#!/usr/bin/env bash
#SBATCH --job-name=wbes-now-recon
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#
# NoW validation: i tre metodi di WS3b (3DDFA_V2, SynergyNet, PRNet) sulle 352 immagini di
# imagepathsvalidation.txt, piu' MICA se richiesto (venv external/venvs/mica).
#
#   aau/submit.sh recon/now_recon.sbatch
#   WBES_NOW_METHODS=mica aau/submit.sh recon/now_recon.sbatch
#
# Dati e mesh stanno FUORI dal repo (licenza NoW: niente redistribuzione):
#   $WBES_NOW_DIR        estratto degli zip (NoW_Dataset, scans, scans_lmks_onlypp)
#   $WBES_NOW_RECON      <metodo>/<soggetto>__<sfida>__<IMG>.npz|json
# I runner chiamano l'uscita come lo stem dell'immagine, e i nomi IMG_xxxx non sono unici
# fra soggetti: le immagini si espongono quindi come symlink gia' battezzati
# <soggetto>__<sfida>__<IMG>.jpg in una cartella piatta, come ws3b_stage_images.py.
# Le immagini gia' ricostruite non si rifanno: risottomettere riprende.
#
# T4 come ws3b_recon.sbatch: i metodi sono CPU-bound (FaceBoxes sull'immagine intera,
# 2320x3088 qui), e le L40S servono al training.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

NOW_DIR="${WBES_NOW_DIR:-$HOME/data/now}"
OUT="${WBES_NOW_RECON:-$HOME/data/now_recon}"
PICS="$NOW_DIR/NoW_Dataset/final_release_version/iphone_pictures"
LIST="$NOW_DIR/imagepathsvalidation.txt"

echo "[now-recon] host=$(hostname) job=${SLURM_JOB_ID:-none} out=$OUT"
nvidia-smi --query-gpu=name --format=csv,noheader || true

STAGE="${TMPDIR:-/tmp}/now-recon-${SLURM_JOB_ID:-$$}"
trap 'rm -rf "$STAGE"' EXIT

METHODS="${WBES_NOW_METHODS:-3ddfa_v2 synergynet prnet}"
for METHOD in $METHODS; do
    PENDING="$STAGE/$METHOD"
    mkdir -p "$PENDING" "$OUT/$METHOD"
    while IFS= read -r rel; do
        [[ -n "$rel" ]] || continue
        IFS=/ read -r subject challenge img <<< "$rel"
        name="${subject}__${challenge}__${img%.*}"
        [[ -f "$OUT/$METHOD/$name.npz" ]] || ln -s "$PICS/$rel" "$PENDING/$name.jpg"
    done < "$LIST"
    n=$(find "$PENDING" -name '*.jpg' | wc -l)
    echo "[now-recon] $METHOD: $n immagini da ricostruire"
    (( n > 0 )) || continue

    t0=$(date +%s)
    case "$METHOD" in
        prnet) "$AAU_DIR/recon/run_recon.sh" prnet aau/recon/prnet_run.py \
                   --images "$PENDING" --out "$OUT/$METHOD" ;;
        mica)  "$AAU_DIR/recon/run_recon.sh" mica aau/recon/mica_run.py \
                   --images "$PENDING" --out "$OUT/$METHOD" ;;
        *)     "$AAU_DIR/recon/run_recon.sh" ddfa "aau/recon/${METHOD/3ddfa_v2/tddfa_v2}_run.py" \
                   --images "$PENDING" --out "$OUT/$METHOD" --device cuda ;;
    esac
    echo "[now-recon] $METHOD: $(( $(date +%s) - t0 )) s"
done

echo "[now-recon] mesh presenti su $(grep -c . "$LIST") immagini:"
for METHOD in $METHODS; do
    echo "  $METHOD: $(find "$OUT/$METHOD" -name '*.npz' | wc -l)"
done
echo "[now-recon] OK"
