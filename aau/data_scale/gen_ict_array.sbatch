#!/usr/bin/env bash
#SBATCH --job-name=wbes-ds-genict
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%A_%a.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%A_%a.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-9%2
#
# ICT nuove, sola geometria (aau/data_scale/PLAN.md, opzione (c)): 50.000 identita' id20000-id69999
# in 200 shard da 250, WBES_DS_N_EXPR espressioni ciascuna (default 8). Ogni task fa 20 shard di fila;
# %2 perche' i 12 slot job dell'utente sono condivisi e la partizione cpu ha un solo nodo da 32 core.
# Riavviabile: gli shard gia' scritti si saltano. Ogni shard passa la guardia held-out di
# gen_ict_shard.py prima di essere scritto; uno shard che fallisce ferma il task con rc 1.
#
#   aau/submit.sh data_scale/gen_ict_array.sbatch
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv
export AAU_NV=
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

OUT="${WBES_DS_OUT:-$WBES_ROOT/datasets/ICT_SCALE/shards}"
N_EXPR="${WBES_DS_N_EXPR:-8}"
PER_TASK=20
TASK="${SLURM_ARRAY_TASK_ID:-0}"
WORK="/tmp/wbes-ds-gen-${SLURM_JOB_ID:-manual}"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT"
echo "[genict] host=$(hostname) task=$TASK shard $((TASK * PER_TASK))-$((TASK * PER_TASK + PER_TASK - 1)) out=$OUT"
for (( s = TASK * PER_TASK; s < (TASK + 1) * PER_TASK; s++ )); do
  t0=$SECONDS
  "$AAU_DIR/run.sh" aau/data_scale/gen_ict_shard.py --shard "$s" --n-per-shard 250 --n-expr "$N_EXPR" \
      --out-dir "$OUT" --work-dir "$WORK" --n-cores "${SLURM_CPUS_PER_TASK:-12}"
  echo "[genict] shard $s: $(( SECONDS - t0 )) s"
done
