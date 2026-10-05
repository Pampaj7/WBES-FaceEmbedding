#!/usr/bin/env bash
#SBATCH --job-name=wbes-precompute
# --chdir e i log sono path ASSOLUTI: le direttive #SBATCH non espandono variabili, e un
# --output relativo viene risolto nella submit dir (se non esiste, Slurm scarta il job in
# un secondo senza scrivere niente). Se sposti il repo, aggiorna queste tre righe oppure
# sottometti con aau/submit.sh, che le ricalcola e le passa da riga di comando.
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#
# Calcola gli operatori DiffusionNet (mass, L, evals, evecs, gradX, gradY) per le 3000
# mesh REMESH: npz_data_topo_500 -> npz_data_topo_500_withops. Solo CPU (eigendecomposizione
# scipy), niente GPU.
#
#   aau/submit.sh precompute_ops.sbatch
#
# k_eig=128: e' il default di precompute_operators_npz.py ed e' il valore con cui sono
# stati calcolati gli operatori del progetto (v2_work/STATUS.md righe 50 e 659). Il
# --eig_k 300 della ricetta di training NON c'entra: model_helpers.py lo passa solo a
# `intrinsic_dn`, mentre il modello top e' `xyz_dn` (DiffusionEncoderOnly), che riceve
# evals/evecs interi. Nel training entra solo nel fingerprint della run dir.
#
# Tempi misurati su 16 core della partizione cpu: ~17 s CPU per mesh `original`,
# ~39 s per `up60k` (la piu' pesante, 60k vertici). Media pesata sulle sei topologie
# ~18 s => ~15 CPU-ore, cioe' ~40 min su 24 core. --time=3h e' tre volte la stima.
#
# RIPARTENZA: lo script salta i file gia' presenti in output (nessun --overwrite), quindi
# se il job muore per walltime basta risottometterlo e riprende da dove era arrivato.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv
aau_require_diffusion_net

IN_DIR="${WBES_REMESH_NOOPS_DIR:-$WBES_ROOT/datasets/REMESH/npz_data_topo_500}"
OUT_DIR="${WBES_DATA_DIR:-$WBES_ROOT/datasets/REMESH/npz_data_topo_500_withops}"
K_EIG="${WBES_K_EIG:-128}"
N_CORES="${WBES_OPS_CORES:-${SLURM_CPUS_PER_TASK:-8}}"

if [[ ! -d "$IN_DIR" ]]; then
    echo "ERRORE: dataset mesh-only non trovato in $IN_DIR" >&2
    echo "  Estrailo prima con: aau/submit.sh extract_remesh.sbatch" >&2
    exit 1
fi

# Un processo per core, ognuno single-thread: senza questo torch/BLAS aprono un pool di
# thread per processo e i 24 worker si contendono i core a vicenda.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[ops] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[ops] in=$IN_DIR ($(ls -1 "$IN_DIR"/*.npz | wc -l) npz)"
echo "[ops] out=$OUT_DIR  k_eig=$K_EIG  n_cores=$N_CORES"

"$AAU_DIR/run.sh" face_embedding/gt_encdec/autoencoder/precompute_operators_npz.py \
  --input-dir "$IN_DIR" \
  --output-dir "$OUT_DIR" \
  --input-kind npz \
  --k-eig "$K_EIG" \
  --n-cores "$N_CORES"

echo "[ops] file prodotti: $(ls -1 "$OUT_DIR"/*.npz | wc -l)"
echo "[ops] dimensione: $(du -sh "$OUT_DIR" | cut -f1)"
