#!/usr/bin/env bash
#SBATCH --job-name=wbes-ds-minilot
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#
# Mini-lotto della scala dati ICT (aau/data_scale/PLAN.md, opzione (c)): lo shard 0 ridotto
# a 24 identita' (id20000-id20023, le stesse dell'array: ogni identita' dipende solo dal suo
# id), 8 espressioni ciascuna, solo geometria in aau/data_scale/minilot/ (~70 MB). Poi
# check_shard.py: mesh, pre-pass contro areanorm_operators.py eseguito cosi' com'e', GT,
# modello congiunto. Gli operatori restano su /tmp del nodo (~6 GB, contano contro --mem).
#
#   aau/submit.sh data_scale/minilot.sbatch
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv
aau_require_diffusion_net
export AAU_NV=
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

WORK="/tmp/wbes-ds-minilot-${SLURM_JOB_ID:-manual}"
trap 'rm -rf "$WORK"' EXIT
OUT="$AAU_DIR/data_scale/minilot"
CKPT="$(ls -d "$AAU_RUNS"/x3dmm_joint_bfm_ict_s1234_1019532/*/checkpoints/epoch120.pth)"
N="${SLURM_CPUS_PER_TASK:-16}"

echo "[minilot] host=$(hostname) job=${SLURM_JOB_ID:-none}"
"$AAU_DIR/run.sh" aau/data_scale/gen_ict_shard.py --shard 0 --n-per-shard 24 --n-expr 8 \
    --out-dir "$OUT" --work-dir "$WORK" --n-cores "$N"
"$AAU_DIR/run.sh" aau/data_scale/check_shard.py --shard-tar "$OUT/shard_00000.tar" \
    --work-dir "$WORK/check" --ckpt "$CKPT" --out-json "$OUT/check.json" --n-proc "$N"
