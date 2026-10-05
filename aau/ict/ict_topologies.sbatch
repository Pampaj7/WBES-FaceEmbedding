#!/usr/bin/env bash
#SBATCH --job-name=wbes-ict-topologies
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#
# ICT-5000, passo 2: le 6 topologie REMESH per ogni identita' (30.000 mesh).
#
#   aau/submit.sh ict/ict_topologies.sbatch --dependency=afterok:<job identita'>
#
# 8 core e non 24 perche' la partizione cpu ha UN SOLO nodo da 32 core, condiviso con il
# precompute degli operatori. Riavviabile: salta le identita' gia' complete.
# Alla fine rilancia il self-check (--demo) sulle prime 3 identita'.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

# `ls dir/*.npz` esplode con "Argument list too long" a 30.000 file (misurato: job 1019283
# e' fallito cosi' DOPO aver scritto correttamente tutte le mesh). find non espande niente.
count_npz() { find "$1" -maxdepth 1 -name "$2" -type f -o -maxdepth 1 -name "$2" -type l | wc -l; }
aau_require_venv

IN_DIR="${WBES_ICT_IDENTITIES_DIR:-$WBES_ROOT/datasets/ICT/identities}"
OUT_DIR="${WBES_ICT_TOPO_DIR:-$WBES_ROOT/datasets/ICT/topo}"
N_CORES="${WBES_ICT_CORES:-${SLURM_CPUS_PER_TASK:-8}}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[ict-topo] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[ict-topo] in=$IN_DIR out=$OUT_DIR cores=$N_CORES"

"$AAU_DIR/run.sh" v2_work/genict/make_ict_topologies.py \
    --in-dir "$IN_DIR" --out-dir "$OUT_DIR" --n-cores "$N_CORES"

N_OUT=$(count_npz "$OUT_DIR" "*_GTready_*.npz")
N_IN=$(count_npz "$IN_DIR" "ict*.npz")
echo "[ict-topo] file: $N_OUT / $((N_IN * 6))  dimensione: $(du -sh "$OUT_DIR" | cut -f1)"
if (( N_OUT != N_IN * 6 )); then
    echo "ERRORE: output incompleto. Risottometti: lo script riprende i mancanti." >&2
    exit 1
fi

echo "[ict-topo] verifica su 3 identita'"
"$AAU_DIR/run.sh" v2_work/genict/make_ict_topologies.py --demo \
    --in-dir "$IN_DIR" --out-dir "$OUT_DIR"
