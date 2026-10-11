#!/usr/bin/env bash
#SBATCH --job-name=wbes-abl2x2-flame
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#
# Ablazione 2x2 (aau/runs/evidence/stream/ablation_2x2/PROTOCOL.md, sez. 4.4): gli insiemi FLAME 2023 di D1
# (flame2023_s1, flame2023: 200 identita' x 5 etichette) rigenerati col codice di D1 (eval_ablation.py flame-gen: mesh in
# datasets/ABL2X2_D1/in, tabella di scala e GT in ablation_2x2/d1flame, controllo contro D1 in d1flame/check.json, uscita
# 1 se diversi), poi gli operatori areanorm k_eig 128 in datasets/V3_OPS_CACHE/abl2x2_d1/ops (come diag.sbatch, passo
# gen). Partizione prioritized senza GPU. Fuori da git; ~12 GB, da cancellare dopo i risultati.
#   sbatch v3_work/stream/slurm/ablation_2x2_flame.sbatch
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
W="${SLURM_CPUS_PER_TASK:-8}"
export AAU_NV=""
RAW="$WBES_ROOT/datasets/ABL2X2_D1/in"
OPS="$WBES_ROOT/datasets/V3_OPS_CACHE/abl2x2_d1"
echo "[abl-flame] host=$(hostname) job=${SLURM_JOB_ID:-none} $(date +%T)"
[[ -f "$OPS/.done" ]] && { echo "[abl-flame] gia' fatto: $OPS/.done"; exit 0; }
t0=$(date +%s)
aau/run.sh v3_work/stream/eval_ablation.py flame-gen --workers "$W"
N=$(find "$RAW" -name "*.npz" | wc -l)
mkdir -p "$OPS/ops"
pids=()
for (( i = 0; i < W - 1; i++ )); do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 aau/run.sh v2_work/potential/areanorm_operators.py \
    --input-dir "$RAW" --output-dir "$OPS/ops" --k-eig 128 --shard "$i/$((W - 1))" \
    > "/tmp/wbes_ablflame_ops_${SLURM_JOB_ID:-x}.$i.log" 2>&1 & pids+=($!)
done
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "ERRORE: shard $i"; cat "/tmp/wbes_ablflame_ops_${SLURM_JOB_ID:-x}.$i.log"; exit 1; }
done
rm -f /tmp/wbes_ablflame_ops_"${SLURM_JOB_ID:-x}".*.log
M=$(find "$OPS/ops" -name "*.npz" ! -name ".*" | wc -l)
echo "[abl-flame] $(date +%T) operatori $M / $N, $(du -sh "$OPS/ops" | cut -f1)"
(( M == N )) || { echo "ERRORE: operatori incompleti"; exit 1; }
touch "$OPS/.done"
echo "[abl-flame] fatto in $(( $(date +%s) - t0 ))s"
