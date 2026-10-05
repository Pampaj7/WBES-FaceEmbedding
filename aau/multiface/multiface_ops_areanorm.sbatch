#!/usr/bin/env bash
#SBATCH --job-name=wbes-mf-ops-areanorm
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#
# WS3a, passo 2c: gli stessi operatori del gemello `multiface_ops.sbatch`, ma con la
# convenzione ad AREA UNITARIA (mesh centrata e divisa per sqrt(area totale) prima del
# calcolo), che e' lo standard del progetto dopo `pot_area` (paper/PLAN_CVPR2027.md:48).
# Serve a valutare i modelli nuovi; la cartella `_withops` senza suffisso serve al v1.
#
#   aau/submit.sh multiface/multiface_ops_areanorm.sbatch
#   WBES_MF_TOPOS=tracked aau/submit.sh multiface/multiface_ops_areanorm.sbatch
#   WBES_MF_TOPOS="crop noisy up" aau/submit.sh multiface/multiface_ops_areanorm.sbatch
#
# Lo script di calcolo e' `v2_work/potential/areanorm_operators.py`, usato verbatim come
# in `aau/precompute_ops_areanorm.sbatch` e in `aau/ict/ict_ops.sbatch`: e' single-process
# con `--shard i/n`, quindi qui si lanciano N processi in background, uno per core.
# Lo sharding e' sulla lista ordinata completa e poi filtrato per i file gia' esistenti,
# quindi il job e' ripartibile: se muore, risottomettilo e riprende dai mancanti.
#
# Qui non serve il trucco dello shard coprimo con 6 dei gemelli REMESH/ICT: dentro una
# cartella le mesh hanno tutte la stessa risoluzione, quindi qualunque stride da' stripe
# equivalenti. Le topologie vengono fatte una dopo l'altra, non insieme, cosi' il numero
# di processi resta quello dei core allocati.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv
aau_require_diffusion_net

PREP_DIR="${WBES_MF_PREP_DIR:-$WBES_ROOT/datasets/Multiface/prep}"
K_EIG="${WBES_K_EIG:-128}"
N_SHARDS="${WBES_MF_CORES:-${SLURM_CPUS_PER_TASK:-8}}"
TOPOS="${WBES_MF_TOPOS:-tracked remesh down crop noisy up}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[mf-areanorm] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[mf-areanorm] prep=$PREP_DIR k_eig=$K_EIG shard=$N_SHARDS topologie='$TOPOS'"

SHARD_LOGS="$AAU_LOGS/${SLURM_JOB_NAME:-wbes-mf-ops-areanorm}-${SLURM_JOB_ID:-manual}.shards"
mkdir -p "$SHARD_LOGS"

rc=0
for topology in $TOPOS; do
    IN_DIR="$PREP_DIR/$topology"
    OUT_DIR="$PREP_DIR/${topology}_withops_areanorm"
    if [[ ! -d "$IN_DIR" ]]; then
        echo "ERRORE: mesh non trovate in $IN_DIR (gira prima multiface_prep.sbatch)" >&2
        exit 1
    fi
    N_IN=$(find "$IN_DIR" -maxdepth 1 -name '*.npz' -type f | wc -l)
    mkdir -p "$OUT_DIR"
    echo "[mf-areanorm] $topology: $N_IN mesh -> $OUT_DIR"

    t0=$(date +%s)
    pids=()
    for (( i = 0; i < N_SHARDS; i++ )); do
        "$AAU_DIR/run.sh" v2_work/potential/areanorm_operators.py \
            --input-dir "$IN_DIR" --output-dir "$OUT_DIR" \
            --k-eig "$K_EIG" --shard "$i/$N_SHARDS" \
            > "$SHARD_LOGS/$topology-shard$i.log" 2>&1 &
        pids+=($!)
    done
    for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
            echo "[mf-areanorm] ERRORE: shard $i di $topology != 0 (vedi $SHARD_LOGS/$topology-shard$i.log)" >&2
            rc=1
        fi
    done
    t1=$(date +%s)

    grep -H '^done ' "$SHARD_LOGS/$topology"-shard*.log || true
    n_fail=$(grep -h '^done ' "$SHARD_LOGS/$topology"-shard*.log \
             | sed 's/.*fail=\([0-9]*\).*/\1/' | awk '{s+=$1} END {print s+0}')
    N_OUT=$(find "$OUT_DIR" -maxdepth 1 -name '*.npz' -type f | wc -l)
    echo "[mf-areanorm] $topology: $N_OUT / $N_IN npz in $(( (t1 - t0) / 60 )) min "\
"(fail riportati: ${n_fail:-?}), $(du -sh "$OUT_DIR" | cut -f1)"
    if (( N_OUT != N_IN )); then
        echo "ERRORE: output incompleto per $topology ($N_OUT/$N_IN). Risottometti: riprende." >&2
        rc=1
    fi
done
if (( rc != 0 )); then
    exit 1
fi

# --- verifica: chiavi complete, k_eig e area totale 1, su un file per topologia -------
echo "[mf-areanorm] verifica su una mesh per topologia"
WBES_CHECK_PREP="$PREP_DIR" WBES_CHECK_TOPOS="$TOPOS" WBES_CHECK_SUFFIX="_withops_areanorm" \
    "$AAU_DIR/run.sh" aau/multiface/check_ops.py
