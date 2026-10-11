#!/usr/bin/env bash
#SBATCH --job-name=wbes-bp-e3
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=aicentre-a100
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=160
#SBATCH --mem=320G
#SBATCH --time=03:00:00
#SBATCH --requeue
#
# Emendamento 3 dei concorrenti parametrici (POST HOC, aau/runs/evidence/baselines_param/PROTOCOL_emendamento_3.md):
#   1. ritagli dell'ingresso dei bracci alla regione di B (bp_e3.py crop, venv open3d) su /tmp;
#   2. per ritaglio: tabella di scala (build_scale_table.py --view-dir) e operatori areanorm k_eig 128;
#   3. crop degli held-out sintetici (bp_e3.py stage-heldout) e loro operatori; cartella unita con la cache della
#      calibrazione (symlink, invariata);
#   4. embedding (eval_v3 + zs_embed, WBES_V3_FACTORIZED_OUT=full) di ogni braccio su ogni ritaglio, controllo della
#      catena sulla cache dello store di HIFI3D, held-out: una coda per GPU (bp_e3.py tasks).
# Uscite in aau/runs/evidence/baselines_param/e3/ (statistiche ed embedding, fuori da git); /tmp cancellato all'uscita.
# A100 di nv-ai-04 con QoS unprivileged (job breve, eccezione autorizzata): prelazionabile, --requeue; ogni passo e'
# ripartibile (gli embedding gia' scritti si saltano; ritagli e operatori si rifanno su /tmp).
#   sbatch aau/baselines_param/bp_e3.sbatch
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
T="/tmp/wbes_bp_e3_${SLURM_JOB_ID:-manual}"
rm -rf "$T"
mkdir -p "$T"
trap 'rm -rf "$T"' EXIT
W="${SLURM_CPUS_PER_TASK:-8}"
export AAU_NV=""
echo "[bp-e3] host=$(hostname) job=${SLURM_JOB_ID:-none} riavvii=${SLURM_RESTART_COUNT:-0} gpu=${CUDA_VISIBLE_DEVICES:-}"
t0=$(date +%s)
# (messaggi su stderr: lo stdout di ``retry ... tasks`` diventa la lista delle code)
# nv-ai-04: la risoluzione dell'utente di dominio cade a tratti all'avvio di singularity (eval_body.sh): 3 tentativi
retry() { local t; for t in 1 2 3; do "$@" && return 0; (( t < 3 )) && { echo "[bp-e3] tentativo $t fallito: $*" >&2; sleep 60; }; done; return 1; }
ops() {  # $1 ingresso, $2 uscita, $3 shard (coprimi con 6), $4 etichetta dei log
  local i pids=()
  mkdir -p "$2"
  for (( i = 0; i < $3; i++ )); do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$AAU_DIR/run.sh" v2_work/potential/areanorm_operators.py \
      --input-dir "$1" --output-dir "$2" --k-eig 128 --shard "$i/$3" > "$T/ops_$4_$i.log" 2>&1 & pids+=($!)
  done
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "ERRORE: shard $i di $4"; cat "$T/ops_$4_$i.log"; return 1; }; done
  local a b
  a=$(find "$1" -name "*.npz" | wc -l); b=$(find "$2" -name "*.npz" ! -name ".*" | wc -l)
  echo "[bp-e3] operatori $b / $a ($4), $(du -sh "$2" | cut -f1)"
  (( a == b )) || { grep -h FAIL "$T"/ops_"$4"_*.log | head; return 1; }
}

# BP_E3_ONLY_HELDOUT=1: solo l'esperimento 2 (embedding dell'esperimento 1 gia' tutti su disco, job 1067973)
ONLY_HELDOUT="${BP_E3_ONLY_HELDOUT:-0}"
if [[ "$ONLY_HELDOUT" != 1 ]]; then
  echo "[bp-e3] $(date +%T) 1. ritagli"
  retry aau/outlineB/run_o3d.sh aau/baselines_param/bp_e3.py crop --tmp "$T" --workers "$W"
  echo "[bp-e3] $(date +%T) 2. tabelle dei ritagli"
  for d in "$T"/crop/*/; do
    d="${d%/}"
    retry "$AAU_DIR/run.sh" v3_work/trainer/tools/build_scale_table.py --view-dir "$d/in" --domain "$(cat "$d/domain")" \
      --out "$d/scale_table.npz" --workers 32 > "$T/table.log" 2>&1 || { cat "$T/table.log"; exit 1; }
  done
fi
echo "[bp-e3] $(date +%T) crop degli held-out"
retry "$AAU_DIR/run.sh" aau/baselines_param/bp_e3.py stage-heldout --tmp "$T"
# operatori di tutti i ritagli e dei crop held-out insieme: S shard per cartella (ognuna con ~600 mesh, l'avvio di
# singularity si ammortizza), W / 9 cartelle in parallelo
echo "[bp-e3] $(date +%T) 3. operatori"
S=$(( W / 9 )); (( S % 6 == 0 )) && S=$(( S - 1 )); (( S < 1 )) && S=1
opids=()
if [[ "$ONLY_HELDOUT" != 1 ]]; then
  for d in "$T"/crop/*/; do d="${d%/}"; ops "$d/in" "$d/ops" "$S" "$(basename "$d")" & opids+=($!); done
fi
ops "$T/heldout/in" "$T/heldout/ops_crop" "$S" heldout & opids+=($!)
for p in "${opids[@]}"; do wait "$p" || { echo "ERRORE: operatori"; exit 1; }; done
# copie, non symlink: con 4 processi la lettura dei 1.500 operatori della cache da CephFS faceva ~18 mesh/min per
# processo (job 1067973); da /tmp ~600 mesh/min
mkdir -p "$T/heldout/ops"
find "$WBES_ROOT"/datasets/V3_OPS_CACHE/heldout_calib/ops/ -maxdepth 1 -name "*.npz" ! -name ".*" -print0 \
  | xargs -0 -P 16 -I{} cp {} "$T/heldout/ops/"
mv "$T"/heldout/ops_crop/*.npz "$T/heldout/ops/"
echo "[bp-e3] held-out: $(find "$T/heldout/ops" -name '*.npz' | wc -l) operatori nella cartella unita"
rm -rf "$T"/crop/*/in "$T/heldout/in" "$T/heldout/ops_crop"
echo "[bp-e3] $(date +%T) 4. embedding"
retry "$AAU_DIR/run.sh" aau/baselines_param/bp_e3.py tasks --tmp "$T" > "$T/tasks.txt"
echo "[bp-e3] $(wc -l < "$T/tasks.txt") embedding da calcolare"
export AAU_NV="--nv"
IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
embed() {  # $1 riga del task, $2 gpu
  local opsd tab dist ck out
  IFS='|' read -r opsd tab dist ck out <<< "$1"
  mkdir -p "$(dirname "$out")"
  # thread limitati: con 4 processi (piu' 8 worker di preload ciascuno) e i thread OpenMP di default (tutti i core
  # visibili) il job 1067823 non ha superato la lettura dei campioni in 34 minuti; 8 thread: 248 s per 600 mesh
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
      retry embed "${TASKS[$k]}" "${GPUS[$q]}" || { echo "ERRORE: ${TASKS[$k]}"; exit 1; }
      echo "[bp-e3] $(date +%T) gpu ${GPUS[$q]}: ${TASKS[$k]##*|}"
    done
  ) & qpids+=($!)
done
rc=0
for p in "${qpids[@]}"; do wait "$p" || rc=1; done
echo "[bp-e3] fine: $(( $(date +%s) - t0 )) s, rc=$rc"
exit "$rc"
