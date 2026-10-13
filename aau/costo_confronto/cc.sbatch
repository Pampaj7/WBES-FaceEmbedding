#!/usr/bin/env bash
#SBATCH --job-name=wbes-cc
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#
# Costo per confronto (aau/costo_confronto/cc.py): tutti i passi in UN job, sullo stesso nodo L40S e con gli stessi 16
# core. Calcolo CPU a un thread per processo (secondi-core); quattro catene in parallelo (ICP, NICP, NoW col
# rilevatore, nostro + ULRSSM + confronti), al piu' ~7 processi a un thread sui 16 core.
#
#   sbatch aau/costo_confronto/cc.sbatch
#   CC_OUT=aau/scratch/costo_smoke CC_REPS=1 CC_SAMPLE="--n-pairs 2 --n-mesh 2" CC_CMP="--n-cmp 100,1000" \
#       sbatch --gres=gpu:a10:1 --time=00:30:00 aau/costo_confronto/cc.sbatch                     (prova)
# Lavoro (render, operatori, cache di ULRSSM) in /tmp del nodo, cancellato a fine job.
set -euo pipefail
export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 UGT_THREADS=1
D=aau/costo_confronto
WORK="/tmp/wbes_cc_${SLURM_JOB_ID:-manual}"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK"
OUTA=()
[[ -n "${CC_OUT:-}" ]] && OUTA=(--out "$CC_OUT")
C=(--work "$WORK" --reps "${CC_REPS:-3}" "${OUTA[@]}")
UG=(v3_work/unified_gt/run.sh -u)
O3=(aau/outlineB/run_o3d.sh -u)
AA=(aau/run.sh -u)
MI=(aau/recon/run_recon.sh mica -u)
CPUS=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)
echo "[cc] host=$(hostname) job=${SLURM_JOB_ID:-none} cpu=$CPUS gpu=${CUDA_VISIBLE_DEVICES:-} $(date +%T)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
lscpu | grep -E "Model name|^CPU\(s\)|Thread|MHz" || true
t0=$(date +%s)
"${UG[@]}" $D/cc.py sample "${C[@]}" ${CC_SAMPLE:-}
pids=()
"${O3[@]}" $D/cc.py geo --metric icp "${C[@]}" & pids+=($!)
"${O3[@]}" $D/cc.py geo --metric nicp "${C[@]}" & pids+=($!)
( "${UG[@]}" $D/cc.py det-render "${C[@]}"
  "${MI[@]}" $D/cc_detect.py "${C[@]}"
  "${UG[@]}" $D/cc.py now "${C[@]}" ) & pids+=($!)
( "${AA[@]}" $D/cc.py ops "${C[@]}"
  AAU_NV=--nv "${AA[@]}" $D/cc.py ours "${C[@]}"
  AAU_NV=--nv "${AA[@]}" $D/cc.py ulrssm "${C[@]}"
  "${UG[@]}" $D/cc.py cmp "${C[@]}" ${CC_CMP:-} ) & pids+=($!)
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
(( fail )) && echo "[cc] ATTENZIONE: almeno una catena e' fallita (vedi sopra)"
# controllo del rilevatore a misure finite (onnxruntime coi thread di default occuperebbe tutti i core)
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK "${MI[@]}" aau/competitors/protocolli/pr_noise_detect.py --dir "$WORK/det_check" \
    && "${UG[@]}" $D/cc.py detcheck "${C[@]}" || fail=1
"${UG[@]}" $D/cc.py report "${OUTA[@]}"
echo "[cc] fatto in $(( $(date +%s) - t0 ))s $(date +%T)"
exit "$fail"
