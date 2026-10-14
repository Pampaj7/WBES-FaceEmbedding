#!/usr/bin/env bash
#SBATCH --job-name=wbes-cc-gpu
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#
# Costo per confronto, il nostro con gli operatori su GPU (E10, cc.py ours --ops-src gpu): stesso nodo L40S e stesso
# campione della misura principale (cc.sbatch, job 1072554). Passi in sequenza: operatori CPU e forward (riferimento
# degli embedding, stesso job), operatori GPU e forward (venv .venv_e10 con cuDSS), report (sezione "Operatori su GPU"
# in results.md; i concorrenti NON si rimisurano). Uscite dei passi in <radice>/gpu/parts.
#
#   sbatch --nodelist=a768-l40s-04 aau/costo_confronto/cc_gpu.sbatch          (il nodo della misura principale)
#   CC_ROOT=aau/scratch/costo_gpu_smoke CC_REPS=1 sbatch --gres=gpu:a10:1 aau/costo_confronto/cc_gpu.sbatch
#       (prova: copia le parti principali in CC_ROOT, non tocca l'evidenza)
set -euo pipefail
export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 UGT_THREADS=1
D=aau/costo_confronto
MAIN=aau/runs/evidence/costo_confronto
ROOT="${CC_ROOT:-$MAIN}"
WORK="/tmp/wbes_ccg_${SLURM_JOB_ID:-manual}"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK" "$ROOT/gpu/parts"
if [[ "$ROOT" != "$MAIN" ]]; then mkdir -p "$ROOT/parts"; cp "$MAIN"/parts/*.json "$ROOT/parts/"; fi
cp "$MAIN/parts/sample.json" "$ROOT/gpu/parts/sample.json"            # lo stesso campione
C=(--work "$WORK" --reps "${CC_REPS:-3}" --out "$ROOT/gpu")
echo "[ccg] host=$(hostname) job=${SLURM_JOB_ID:-none} cpu=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status) $(date +%T)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
t0=$(date +%s)
aau/run.sh -u $D/cc.py ops "${C[@]}"
AAU_NV=--nv aau/run.sh -u $D/cc.py ours "${C[@]}"
AAU_NV=--nv VENV="$WBES_ROOT/.venv_e10" aau/run.sh -u $D/cc.py ours --ops-src gpu "${C[@]}"
v3_work/unified_gt/run.sh -u $D/cc.py report --out "$ROOT"
echo "[ccg] fatto in $(( $(date +%s) - t0 ))s $(date +%T)"
