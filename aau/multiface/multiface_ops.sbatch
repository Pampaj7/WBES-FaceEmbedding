#!/usr/bin/env bash
#SBATCH --job-name=wbes-mf-ops
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#
# WS3a, passo 2b: operatori DiffusionNet k_eig 128 con la convenzione STANDARD (mesh
# lasciate nelle loro unita', come `aau/precompute_ops.sbatch`) per le sei topologie
# Multiface. Serve a valutare il checkpoint v1, che e' stato addestrato cosi'.
#
#   aau/submit.sh multiface/multiface_ops.sbatch
#   WBES_MF_TOPOS=tracked aau/submit.sh multiface/multiface_ops.sbatch          # una sola
#   WBES_MF_TOPOS="crop noisy up" aau/submit.sh multiface/multiface_ops.sbatch  # le dure
#
# Gemello di `multiface_ops_areanorm.sbatch`, che fa le stesse sei topologie con la
# convenzione ad area unitaria. Lo script di calcolo e'
# `face_embedding/gt_encdec/autoencoder/precompute_operators_npz.py`, usato verbatim: ha
# gia' la sua Pool su `--n-cores` e salta i file gia' presenti, quindi il job e'
# ripartibile risottomettendolo.
#
# 8 core e non 24: la partizione cpu ha UN SOLO nodo da 32 core, condiviso con il
# precompute ICT e con i job di ricostruzione.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv
aau_require_diffusion_net

PREP_DIR="${WBES_MF_PREP_DIR:-$WBES_ROOT/datasets/Multiface/prep}"
K_EIG="${WBES_K_EIG:-128}"
N_CORES="${WBES_MF_CORES:-${SLURM_CPUS_PER_TASK:-8}}"
TOPOS="${WBES_MF_TOPOS:-tracked remesh down crop noisy up}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[mf-ops] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[mf-ops] prep=$PREP_DIR k_eig=$K_EIG cores=$N_CORES topologie='$TOPOS'"

rc=0
for topology in $TOPOS; do
    IN_DIR="$PREP_DIR/$topology"
    OUT_DIR="$PREP_DIR/${topology}_withops"
    if [[ ! -d "$IN_DIR" ]]; then
        echo "ERRORE: mesh non trovate in $IN_DIR (gira prima multiface_prep.sbatch)" >&2
        exit 1
    fi
    N_IN=$(find "$IN_DIR" -maxdepth 1 -name '*.npz' -type f | wc -l)
    echo "[mf-ops] $topology: $N_IN mesh -> $OUT_DIR"

    t0=$(date +%s)
    "$AAU_DIR/run.sh" face_embedding/gt_encdec/autoencoder/precompute_operators_npz.py \
        --input-dir "$IN_DIR" --output-dir "$OUT_DIR" \
        --input-kind npz --k-eig "$K_EIG" --n-cores "$N_CORES"
    t1=$(date +%s)

    N_OUT=$(find "$OUT_DIR" -maxdepth 1 -name '*.npz' -type f | wc -l)
    echo "[mf-ops] $topology: $N_OUT / $N_IN npz in $(( (t1 - t0) / 60 )) min, $(du -sh "$OUT_DIR" | cut -f1)"
    if (( N_OUT != N_IN )); then
        echo "ERRORE: output incompleto per $topology ($N_OUT/$N_IN). Risottometti: riprende." >&2
        rc=1
    fi
done
if (( rc != 0 )); then
    exit 1
fi

# --- verifica: chiavi complete e k_eig, su un file per topologia ----------------------
echo "[mf-ops] verifica su una mesh per topologia"
WBES_CHECK_PREP="$PREP_DIR" WBES_CHECK_TOPOS="$TOPOS" WBES_CHECK_SUFFIX="_withops" \
    "$AAU_DIR/run.sh" aau/multiface/check_ops.py
