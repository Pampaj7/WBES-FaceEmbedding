#!/usr/bin/env bash
#SBATCH --job-name=wbes-d4-embed
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=aicentre-a100
#SBATCH --qos=unprivileged
#SBATCH --requeue
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#
# Diagnostica D4 (aau/runs/evidence/diagnostics/PROTOCOL_D4.md sez. 2): embedding dei bracci (factorized s1234,
# s2345, ctrlfr s1234) sulle mesh canoniche delle 3 viste, e il controllo della catena K2 (factorized s1234 sulla cache
# dello store di HIFI3D), dopo D4_STEP=prep. Operatori copiati da CephFS su /tmp (bp_e3.sbatch: dalla cache su CephFS
# con 4 processi ~18 mesh/min per processo, da /tmp ~600 mesh/min), una coda per GPU, gli argomenti di zs_embed degli
# store (bp_e3.sbatch). A100 di nv-ai-04 con QoS unprivileged (job breve, eccezione autorizzata): prelazionabile,
# --requeue; gli embedding gia' scritti si saltano. Tempi a parete per embedding nel log (costo, sez. 6).
#   sbatch aau/diagnostics/d4_embed.sbatch
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
T="/tmp/wbes_d4_embed_${SLURM_JOB_ID:-manual}"
rm -rf "$T"
mkdir -p "$T"
trap 'rm -rf "$T"' EXIT
export AAU_NV=""
echo "[d4-embed] host=$(hostname) job=${SLURM_JOB_ID:-none} riavvii=${SLURM_RESTART_COUNT:-0} gpu=${CUDA_VISIBLE_DEVICES:-}"
t0=$(date +%s)
# nv-ai-04: la risoluzione dell'utente di dominio cade a tratti all'avvio di singularity (eval_body.sh): 3 tentativi
retry() { local t; for t in 1 2 3; do "$@" && return 0; (( t < 3 )) && { echo "[d4-embed] tentativo $t fallito: $*" >&2; sleep 60; }; done; return 1; }
for v in facescape hifi3d faceverse_neutral; do
  mkdir -p "$T/$v"
  find "$WBES_ROOT/datasets/V3_OPS_CACHE/diag_d4/$v/ops/" -maxdepth 1 -name "*.npz" ! -name ".*" -print0 \
    | xargs -0 -P 16 -I{} cp {} "$T/$v/"
done
mkdir -p "$T/hifi3d_store"
find "$WBES_ROOT/datasets/V3_OPS_CACHE/8e8f81d5f0204394/" -maxdepth 1 -name "*.npz" ! -name ".*" -print0 \
  | xargs -0 -P 16 -I{} cp {} "$T/hifi3d_store/"
for d in "$T"/*/; do echo "[d4-embed] $(basename "$d"): $(find "$d" -name '*.npz' | wc -l) operatori"; done
echo "[d4-embed] $(date +%T) copia finita in $(( $(date +%s) - t0 )) s, $(du -sh "$T" | cut -f1)"
retry "$AAU_DIR/run.sh" aau/diagnostics/d4_remesh.py tasks --ops-root "$T" > "$T/tasks.txt"
echo "[d4-embed] $(wc -l < "$T/tasks.txt") embedding da calcolare"
export AAU_NV="--nv"
IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
embed() {  # $1 riga del task, $2 gpu
  local opsd tab dist ck out
  IFS='|' read -r opsd tab dist ck out <<< "$1"
  mkdir -p "$(dirname "$out")"
  CUDA_VISIBLE_DEVICES="$2" WBES_V3_SCALE_TABLES="$tab" WBES_V3_FACTORIZED_OUT=full WBES_EVAL_SCENARIOS=clean \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    "$AAU_DIR/run.sh" v3_work/trainer/eval_v3.py -- aau/zs3dmm/zs_embed.py \
      --model_path "$ck" --checkpoint_selector best_by_clean --data_dir "$opsd" --dist_npz "$dist" \
      --subject_split all --eval_fraction 0.2 --max_subjects 0 --max_meshes_per_subject_eval 10 --preload_workers 8 \
      --pair_mode cross_topology --seed 1234 --embeddings_out "$out" > "$(dirname "$out")/embed.log" 2>&1
}
mapfile -t TASKS < "$T/tasks.txt"
qpids=()
for (( q = 0; q < ${#GPUS[@]}; q++ )); do
  (
    for (( k = q; k < ${#TASKS[@]}; k += ${#GPUS[@]} )); do
      s=$(date +%s)
      retry embed "${TASKS[$k]}" "${GPUS[$q]}" || { echo "ERRORE: ${TASKS[$k]}"; exit 1; }
      echo "[d4-embed] $(date +%T) gpu ${GPUS[$q]}: $(( $(date +%s) - s )) s ${TASKS[$k]##*|}"
    done
  ) & qpids+=($!)
done
rc=0
for p in "${qpids[@]}"; do wait "$p" || rc=1; done
echo "[d4-embed] fine: $(( $(date +%s) - t0 )) s, rc=$rc"
exit "$rc"
