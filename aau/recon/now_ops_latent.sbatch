#!/usr/bin/env bash
#SBATCH --job-name=wbes-now-latent
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#
# NoW, latent del congiunto: operatori ad area unitaria sulle patch di
# now_prepare_meshes.py, poi now_latent.py.
#
#   WBES_CKPT=<congiunto> aau/submit.sh recon/now_ops_latent.sbatch
#   WBES_NOW_OVERWRITE=1 WBES_CKPT=<congiunto> aau/submit.sh recon/now_ops_latent.sbatch
#
# Operatori e latenti si riprendono dai file esistenti: se le patch sono cambiate (nuova
# now_prepare_meshes.py) va dato WBES_NOW_OVERWRITE=1, che passa --overwrite sia a
# areanorm_operators.py sia a now_latent.py, altrimenti si riusano operatori e latenti VECCHI
# senza nessun errore.
#
# Operatori come ws3b_ops_areanorm.sbatch: v2_work/potential/areanorm_operators.py verbatim,
# un processo per core con --shard i/n, ripartibile.  ~1100 mesh da ~2.7k vertici: pochi
# minuti.  Il latent sono ~1100 forward: T4, la GPU resta quasi ferma comunque.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv
aau_require_diffusion_net
aau_require_checkpoint

WORK="${WBES_NOW_WORK:-$HOME/data/now_eval_work}"
METHODS="${WBES_NOW_METHODS:-3ddfa_v2 synergynet prnet mica}"
OVERWRITE=(); [[ "${WBES_NOW_OVERWRITE:-0}" == "1" ]] && OVERWRITE=(--overwrite)
N_SHARDS="${SLURM_CPUS_PER_TASK:-8}"
SUFFIX="_withops_areanorm"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "[now-latent] host=$(hostname) job=${SLURM_JOB_ID:-none} ckpt=$WBES_CKPT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

SHARD_LOGS="$AAU_LOGS/${SLURM_JOB_NAME:-wbes-now-latent}-${SLURM_JOB_ID:-manual}.shards"
mkdir -p "$SHARD_LOGS"

PAIRS="scan_face:scan_face$SUFFIX"
for m in $METHODS; do PAIRS="$PAIRS recon_face/$m:recon_face$SUFFIX/$m"; done
for pair in $PAIRS; do
    IN_DIR="$WORK/${pair%%:*}"; OUT_DIR="$WORK/${pair##*:}"
    tag="$(echo "${pair%%:*}" | tr '/' '_')"
    [[ -d "$IN_DIR" ]] || { echo "ERRORE: mancano le mesh in $IN_DIR (now_prepare_meshes.py)" >&2; exit 1; }
    mkdir -p "$OUT_DIR"
    t0=$(date +%s); pids=()
    for (( i = 0; i < N_SHARDS; i++ )); do
        "$AAU_DIR/run.sh" v2_work/potential/areanorm_operators.py \
            --input-dir "$IN_DIR" --output-dir "$OUT_DIR" --k-eig 128 --shard "$i/$N_SHARDS" "${OVERWRITE[@]}" \
            > "$SHARD_LOGS/$tag-shard$i.log" 2>&1 &
        pids+=($!)
    done
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "ERRORE: shard $i di $tag (vedi $SHARD_LOGS/$tag-shard$i.log)" >&2; exit 1; }
    done
    N_IN=$(find "$IN_DIR" -maxdepth 1 -name '*.npz' | wc -l)
    N_OUT=$(find "$OUT_DIR" -maxdepth 1 -name '*.npz' | wc -l)
    echo "[now-latent] operatori ${pair%%:*}: $N_OUT/$N_IN in $(( $(date +%s) - t0 )) s"
    (( N_OUT == N_IN )) || { echo "ERRORE: operatori incompleti per ${pair%%:*}" >&2; exit 1; }
done

AAU_NV="--nv" "$AAU_DIR/run.sh" aau/recon/now_latent.py --methods "${METHODS// /,}" "${OVERWRITE[@]}" "$@"

echo "[now-latent] OK"
