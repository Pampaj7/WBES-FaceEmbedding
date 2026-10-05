#!/usr/bin/env bash
#SBATCH --job-name=wbes-precompute-ra1
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%A_%a.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%A_%a.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#
# Operatori robusti ad area 1 (ablazioni C ed E), aau/models/robust_area1_operators.py.
# Array: ogni task lancia WBES_RA1_PROCS processi single-thread (default = cpu del task), shard
# globale = task * procs + p su ntask * procs. Lo shard e' su una permutazione fissa della lista,
# quindi le topologie si mescolano per qualunque numero di shard. Ripartibile: i file gia'
# scritti si saltano, scrittura atomica.
#
#   aau/submit.sh aau/models/precompute_robust_area1.sbatch                       # BFM/REMESH
#   WBES_RA1_IN=$PWD/datasets/ICT/eval_view_heldout \
#   WBES_RA1_OUT=$PWD/datasets/ICT/eval_view_heldout_robust_area1 \
#     aau/submit.sh aau/models/precompute_robust_area1.sbatch --array=0-1 --job-name=wbes-precompute-ra1-ict
#
# Verifica a parte (chiavi, area, k_eig), su tutti i file:
#   aau/run.sh aau/models/robust_area1_operators.py --check --check-all --input-dir IN --output-dir OUT --ref-dir REF
#
# Costo: come il cotangente (precompute_ops.sbatch: ~18 s CPU a mesh in media, ~15 CPU-ore per le
# 3000 BFM), piu' la triangolazione intrinseca, che e' una frazione dell'autosolver.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv
aau_require_diffusion_net
export AAU_NV=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

IN_DIR="${WBES_RA1_IN:-$WBES_ROOT/datasets/REMESH/npz_data_topo_500}"
OUT_DIR="${WBES_RA1_OUT:-$WBES_ROOT/datasets/REMESH/npz_data_topo_500_withops_robust_area1}"
PROCS="${WBES_RA1_PROCS:-${SLURM_CPUS_PER_TASK:-6}}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
NTASK="${SLURM_ARRAY_TASK_COUNT:-1}"
N=$(( NTASK * PROCS ))
FIRST=$(( TASK * PROCS ))
# Fuori dall'array (p.es. un job extra su un nodo con CPU libere): WBES_RA1_FIRST e WBES_RA1_N
# scelgono gli shard FIRST..FIRST+PROCS-1 su N, gli stessi che avrebbero preso i task dell'array.
if [[ -n "${WBES_RA1_FIRST:-}" ]]; then
    FIRST="$WBES_RA1_FIRST"; N="${WBES_RA1_N:?con WBES_RA1_FIRST serve WBES_RA1_N}"
fi

echo "[ra1] host=$(hostname) job=${SLURM_JOB_ID:-none} shard $FIRST..$(( FIRST + PROCS - 1 )) su $N"
echo "[ra1] in=$IN_DIR out=$OUT_DIR"
mkdir -p "$OUT_DIR"
LOGS="$AAU_LOGS/${SLURM_JOB_NAME:-wbes-precompute-ra1}-${SLURM_ARRAY_JOB_ID:-manual}.shards"
mkdir -p "$LOGS"

pids=()
for (( p = 0; p < PROCS; p++ )); do
    s=$(( FIRST + p ))
    "$AAU_DIR/run.sh" aau/models/robust_area1_operators.py \
        --input-dir "$IN_DIR" --output-dir "$OUT_DIR" --k-eig 128 --shard "$s/$N" \
        > "$LOGS/shard$s.log" 2>&1 &
    pids+=($!)
done
rc=0
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "[ra1] ERRORE: shard $(( FIRST + i )) uscito != 0" >&2; rc=1; }
done
for (( p = 0; p < PROCS; p++ )); do
    s=$(( FIRST + p ))
    echo "  shard $s: $(grep -h '^done ' "$LOGS/shard$s.log" || echo 'senza riga done')"
done
echo "[ra1] file in $OUT_DIR: $(ls -1 "$OUT_DIR" | grep -c '^id.*\.npz$')"
exit $rc
