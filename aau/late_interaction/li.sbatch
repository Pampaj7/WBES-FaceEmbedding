#!/usr/bin/env bash
#SBATCH --job-name=wbes-li
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# Interazione tardiva (aau/late_interaction/): un passo per job, LI_STEP.
#   LI_STEP=stage  sbatch --cpus-per-task=32 --time=00:30:00 aau/late_interaction/li.sbatch   (insiemi + operatori del pilota)
#   LI_STEP=run LI_SEED=s1234 sbatch --gres=gpu:l40s:1 --time=03:00:00 aau/late_interaction/li.sbatch
#   LI_STEP=eval   sbatch --cpus-per-task=64 --mem=192G --time=02:00:00 aau/late_interaction/li.sbatch  (calib, pilota,
#                  plug-in e valutazione; LI_EVAL="calib pilot" per fermarsi alla scelta)
#   LI_STEP=cost   sbatch --gres=gpu:l40s:1 --cpus-per-task=16 --time=01:30:00 aau/late_interaction/li.sbatch
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
D=aau/late_interaction
UG=(v3_work/unified_gt/run.sh -u)
GP=(aau/competitors/gw_percettivi/gp_run.sh -u)
echo "[li] passo=${LI_STEP:?} host=$(hostname) job=${SLURM_JOB_ID:-none} gpu=${CUDA_VISIBLE_DEVICES:-} $(date +%T)"
case "$LI_STEP" in
  stage)
    export AAU_NV=""
    "${UG[@]}" $D/li_tokens.py stage
    W=$SLURM_CPUS_PER_TASK
    for v in hifi3d facescape; do
      IN=datasets/V3_OPS_CACHE/late_interaction/pilot_$v/in
      OUT=datasets/V3_OPS_CACHE/late_interaction/pilot_$v/ops
      mkdir -p "$OUT"
      pids=()
      for (( i = 0; i < W; i++ )); do
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 aau/run.sh v2_work/potential/areanorm_operators.py \
          --input-dir "$IN" --output-dir "$OUT" --k-eig 128 --shard "$i/$W" > "/tmp/wbes_li_ops_${SLURM_JOB_ID:-x}.$i.log" 2>&1 & pids+=($!)
      done
      for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "ERRORE: shard $i"; cat "/tmp/wbes_li_ops_${SLURM_JOB_ID:-x}.$i.log"; exit 1; }; done
      rm -f /tmp/wbes_li_ops_"${SLURM_JOB_ID:-x}".*.log
      n=$(find "$IN" -name "*.npz" | wc -l); m=$(find "$OUT" -name "*.npz" ! -name ".*" | wc -l)
      echo "[li] $v: operatori $m / $n"
      (( n == m )) || { echo "ERRORE: operatori incompleti"; exit 1; }
    done
    "${UG[@]}" $D/li_tokens.py stage ;;                 # ricontrolla: ops_missing = 0
  run)
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
    export AAU_NV=--nv TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 SINGULARITYENV_TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
    "${GP[@]}" $D/li_tokens.py run --seed "${LI_SEED:?}" ${LI_ONLY:+--only "$LI_ONLY"} ;;
  eval)
    export AAU_NV=""
    for s in ${LI_EVAL:-calib pilot plugins eval}; do
      "${UG[@]}" $D/li_eval.py "$s" --workers "$SLURM_CPUS_PER_TASK"
    done ;;
  cost)
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 UGT_THREADS=1
    WORK="/tmp/wbes_li_cost_${SLURM_JOB_ID:-manual}"
    trap 'rm -rf "$WORK"' EXIT
    mkdir -p "$WORK"
    CC=(--out aau/runs/evidence/late_interaction/costo/cc --work "$WORK" --reps 3)
    AAU_NV="" "${UG[@]}" aau/costo_confronto/cc.py sample "${CC[@]}"
    AAU_NV="" aau/outlineB/run_o3d.sh -u aau/costo_confronto/cc.py geo --metric icp "${CC[@]}"
    AAU_NV="" "${UG[@]}" aau/costo_confronto/cc.py cmp "${CC[@]}" --n-cmp 1000,10000
    export AAU_NV=--nv TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 SINGULARITYENV_TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
    "${GP[@]}" $D/li_cost.py --work "$WORK" ;;
  *) echo "LI_STEP sconosciuto: $LI_STEP"; exit 2 ;;
esac
echo "[li] fatto $(date +%T)"
