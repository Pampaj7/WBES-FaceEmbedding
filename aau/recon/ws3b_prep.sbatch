#!/usr/bin/env bash
#SBATCH --job-name=wbes-ws3b-prep
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#
# WS3b, passo 2: regione volto su GT e ricostruzioni, protocollo di coppie e operatori.
#
#   aau/submit.sh recon/ws3b_prep.sbatch
#
# Niente GPU: sono decimazioni e autovalori sparsi.  Lo script di calcolo degli operatori
# e' `precompute_operators_npz.py` del repo, usato verbatim come in
# `aau/multiface/multiface_ops.sbatch`: ha la sua Pool e salta i file gia' presenti, quindi
# il job e' ripartibile risottomettendolo.
#
# 16 core: la partizione cpu ha UN SOLO nodo da 32, condiviso con gli altri precompute.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv
aau_require_diffusion_net

OUT="${WBES_WS3B_OUT:-$AAU_RUNS/multiface_ws3b}"
export WBES_WS3B_OUT="$OUT"
K_EIG="${WBES_K_EIG:-128}"
N_CORES="${WBES_WS3B_CORES:-${SLURM_CPUS_PER_TASK:-8}}"
METHODS="${WBES_WS3B_METHODS:-3ddfa_v2 synergynet prnet}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[ws3b-prep] host=$(hostname) job=${SLURM_JOB_ID:-none} out=$OUT k_eig=$K_EIG cores=$N_CORES"

echo "[ws3b-prep] --- ritaglio della GT ---"
"$AAU_DIR/run.sh" aau/recon/ws3b_prepare_meshes.py --stage gt --workers "$N_CORES"

echo "[ws3b-prep] --- 7 landmark NoW sulla topologia tracciata ---"
# Serve al passo dopo: il raggio del ritaglio della ricostruzione e' 95 mm / ex-ex della GT,
# e l'ex-ex della GT si misura su questi vertici. Salta se il json c'e' gia'.
"$AAU_DIR/run.sh" aau/recon/ws3b_landmarks.py --workers "$N_CORES" \
    --methods "$(echo "$METHODS" | tr ' ' ',')"

echo "[ws3b-prep] --- ritaglio e decimazione delle ricostruzioni ---"
"$AAU_DIR/run.sh" aau/recon/ws3b_prepare_meshes.py --stage recon --workers "$N_CORES" \
    --methods "$(echo "$METHODS" | tr ' ' ',')" ${WBES_WS3B_PREP_ARGS:-}

echo "[ws3b-prep] --- protocollo di coppie per l'AUC di identita' ---"
python3 aau/multiface/make_pairs_protocol.py \
    --manifest "$OUT/auc_manifest.csv" --out "$OUT/pairs_protocol.json" \
    --topology recon --max-pairs "${WBES_WS3B_MAX_PAIRS:-2000}" --seed 1234

echo "[ws3b-prep] --- operatori DiffusionNet ---"
for pair in "gt_face:gt_face_withops" $(for m in $METHODS; do echo "recon_face/$m:recon_face_withops/$m"; done); do
    IN_DIR="$OUT/${pair%%:*}"
    OUT_DIR="$OUT/${pair##*:}"
    N_IN=$(find "$IN_DIR" -maxdepth 1 -name '*.npz' -type f | wc -l)
    echo "[ws3b-prep] ${pair%%:*}: $N_IN mesh -> $OUT_DIR"
    t0=$(date +%s)
    "$AAU_DIR/run.sh" face_embedding/gt_encdec/autoencoder/precompute_operators_npz.py \
        --input-dir "$IN_DIR" --output-dir "$OUT_DIR" \
        --input-kind npz --k-eig "$K_EIG" --n-cores "$N_CORES"
    N_OUT=$(find "$OUT_DIR" -maxdepth 1 -name '*.npz' -type f | wc -l)
    echo "[ws3b-prep] ${pair%%:*}: $N_OUT / $N_IN npz in $(( ($(date +%s) - t0) / 60 )) min, $(du -sh "$OUT_DIR" | cut -f1)"
    if (( N_OUT != N_IN )); then
        echo "ERRORE: output incompleto per ${pair%%:*} ($N_OUT/$N_IN). Risottometti: riprende." >&2
        exit 1
    fi
done

echo "[ws3b-prep] OK"
