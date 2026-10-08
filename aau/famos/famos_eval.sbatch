#!/usr/bin/env bash
#SBATCH --job-name=wbes-famos-eval
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#
# FaMoS, persone mai viste: un checkpoint e Chamfer sul set di test reale (famos_build.sbatch).
#
#   aau/submit.sh famos/famos_eval.sbatch                       # e108 (default)
#   WBES_FAMOS_CKPT=<.pth> WBES_FAMOS_TAG=<nome> aau/submit.sh famos/famos_eval.sbatch
#
#   a. operatori ad area unitaria k_eig 128 (v2_work/potential/areanorm_operators.py, come
#      zs_zeroshot.sbatch) di tutte le patch, su /tmp del nodo, una volta per job;
#   b. famos_eval.py: latenti, Chamfer verso la galleria, riconoscimento e graduata.
# CPU: ~1300 patch da ~2.7k vertici, il forward su CPU basta (il device ricade su CPU senza CUDA);
# con una GPU (p.es. --partition=prioritized --gres=gpu:l40s:1) il forward va su GPU.
# Uscite in aau/runs/evidence/famos/eval_<tag>; /tmp si cancella all'uscita.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
if [[ -z "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}${GPU_DEVICE_ORDINAL:-}" ]]; then
    export AAU_NV=""
fi
cd "$WBES_ROOT"
aau_require_venv
aau_require_diffusion_net

E108="$AAU_RUNS/data_scale_runs/scale_bfm_ict_gnm_s1234_nocanon_noaug_20261007_1411/mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_ks0_poolmeanmax_noise60_sig5e-4-2e-2_latentnoise_seed1234__3167d36d/checkpoints/epoch108.pth"
CKPT="${WBES_FAMOS_CKPT:-$E108}"
TAG="${WBES_FAMOS_TAG:-e108}"
VIEW="$WBES_ROOT/datasets/FAMOS/test_view"
[[ -f "$CKPT" ]] || { echo "ERRORE: checkpoint assente: $CKPT" >&2; exit 1; }
[[ -f "$VIEW/manifest.csv" && -f "$VIEW/gt_matrix.npz" ]] || { echo "ERRORE: vista assente, lancia famos_build.sbatch" >&2; exit 1; }

STAGE_TMP="/tmp/wbes_famos_${SLURM_JOB_ID:-manual}"
rm -rf "$STAGE_TMP"
mkdir -p "$STAGE_TMP/ops"
trap 'rm -rf "$STAGE_TMP"' EXIT

echo "[famos-eval] host=$(hostname) job=${SLURM_JOB_ID:-none} tag=$TAG ckpt=$CKPT"
echo "[famos-eval] a. operatori (k_eig 128, area unitaria) su /tmp"
N_SHARDS="${SLURM_CPUS_PER_TASK:-8}"
SHARD_LOGS="$AAU_LOGS/${SLURM_JOB_NAME:-wbes-famos-eval}-${SLURM_JOB_ID:-manual}.shards"
mkdir -p "$SHARD_LOGS"
pids=()
for (( i = 0; i < N_SHARDS; i++ )); do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$AAU_DIR/run.sh" v2_work/potential/areanorm_operators.py \
      --input-dir "$VIEW/npz" --output-dir "$STAGE_TMP/ops" --k-eig 128 --shard "$i/$N_SHARDS" \
      > "$SHARD_LOGS/shard$i.log" 2>&1 &
    pids+=($!)
done
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "ERRORE: shard $i (vedi $SHARD_LOGS/shard$i.log)" >&2; exit 1; }
done
N_IN=$(find "$VIEW/npz" -name "*.npz" | wc -l)
N_OPS=$(find "$STAGE_TMP/ops" -name "*.npz" | wc -l)
echo "[famos-eval] operatori: $N_OPS / $N_IN, $(du -sh "$STAGE_TMP/ops" | cut -f1) su /tmp"
(( N_OPS == N_IN )) || { echo "ERRORE: operatori incompleti" >&2; exit 1; }

echo "[famos-eval] b. latenti, Chamfer, metriche"
OMP_NUM_THREADS=4 "$AAU_DIR/run.sh" aau/famos/famos_eval.py --ops-dir "$STAGE_TMP/ops" --checkpoint "$CKPT" \
    --tag "$TAG" --out-dir "$AAU_RUNS/evidence/famos/eval_$TAG" --workers "${SLURM_CPUS_PER_TASK:-8}"
echo "[famos-eval] OK"
