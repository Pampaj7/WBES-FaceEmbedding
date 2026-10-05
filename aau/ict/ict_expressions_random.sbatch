#!/usr/bin/env bash
#SBATCH --job-name=wbes-ict-rexpr-gen
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#
# WS5 rifatto, passo 1: 5 espressioni CASUALI E DIVERSE per ognuna delle 500 identita'
# held-out ICT -> 2500 mesh in datasets/ICT/expressions_random/.
#
#   aau/submit.sh ict/ict_expressions_random.sbatch
#
# Gemello di ict_expressions.sbatch, ma il generatore e' aau/ict/ict_expressions_random.py
# (nuovo, niente e' stato toccato in v2_work/genict/): il vecchio applica lo stesso
# blendshape con lo stesso coefficiente a tutti, e sposta i 100 soggetti valutati insieme.
#
# Il job esegue prima `--demo`, che e' il controllo di scala: verifica che jawOpen a
# intensita' 1.00 su id14500 dia 0.042 di spostamento medio e diametro 2.18, cioe' gli
# stessi numeri con cui si legge la tabella. Costa 20 s e fallisce PRIMA delle 2500 mesh.
#
# Serve solo lo split (passo 5 di ICT) e i pesi di identita' (passo 1); non serve il 4.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

# Nodo senza GPU: --nv farebbe fallire singularity per mancanza di libnvidia.
export AAU_NV=""
aau_require_venv

count_npz() { find "$1" -maxdepth 1 -name "$2" -type f | wc -l; }

OUT_DIR="${WBES_ICT_REXPR_DIR:-$WBES_ROOT/datasets/ICT/expressions_random}"
N_CORES="${WBES_ICT_CORES:-${SLURM_CPUS_PER_TASK:-4}}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[ict-rexpr] host=$(hostname) job=${SLURM_JOB_ID:-none} out=$OUT_DIR cores=$N_CORES"

echo "[ict-rexpr] controllo di scala (--demo)"
"$AAU_DIR/run.sh" aau/ict/ict_expressions_random.py --demo

"$AAU_DIR/run.sh" aau/ict/ict_expressions_random.py \
    --out-dir "$OUT_DIR" --n-cores "$N_CORES"

echo "[ict-rexpr] file: $(count_npz "$OUT_DIR" "id*_rexpr_*.npz")  dimensione: $(du -sh "$OUT_DIR" | cut -f1)"
