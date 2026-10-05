#!/usr/bin/env bash
#SBATCH --job-name=wbes-ict-expressions
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#
# ICT-5000, passo 6 (WS5): 5 espressioni x 3 intensita' per le 500 identita' held-out,
# sulla topologia `original` -> 7500 mesh in datasets/ICT/expressions/.
#
#   aau/submit.sh ict/ict_expressions.sbatch --dependency=afterok:<job gt>
#
# Serve solo lo split (dal passo 5) e i pesi (dal passo 1); non serve il passo 4.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

# `ls dir/*.npz` esplode con "Argument list too long" a 30.000 file (misurato: job 1019283
# e' fallito cosi' DOPO aver scritto correttamente tutte le mesh). find non espande niente.
count_npz() { find "$1" -maxdepth 1 -name "$2" -type f -o -maxdepth 1 -name "$2" -type l | wc -l; }
aau_require_venv

OUT_DIR="${WBES_ICT_EXPR_DIR:-$WBES_ROOT/datasets/ICT/expressions}"
N_CORES="${WBES_ICT_CORES:-${SLURM_CPUS_PER_TASK:-4}}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[ict-expr] host=$(hostname) job=${SLURM_JOB_ID:-none} out=$OUT_DIR cores=$N_CORES"

"$AAU_DIR/run.sh" v2_work/genict/make_expressions.py \
    --out-dir "$OUT_DIR" --n-cores "$N_CORES"

echo "[ict-expr] file: $(count_npz "$OUT_DIR" "id*_expr_*.npz")  dimensione: $(du -sh "$OUT_DIR" | cut -f1)"
