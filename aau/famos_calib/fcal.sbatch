#!/usr/bin/env bash
#SBATCH --job-name=wbes-fcal
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#
# FaMoS, calibrazione lineare post hoc (aau/famos_calib/fcal.py; regole in aau/runs/evidence/famos_calib/REGOLE.md).
#
#   FC_STEP=prep sbatch aau/famos_calib/fcal.sbatch                     (selezione, preparazione, fold, operatori)
#   FC_STEP=gpu  sbatch --gres=gpu:l40s:1 -c 8 aau/famos_calib/fcal.sbatch   (latenti [s, u] dei due semi)
#   FC_STEP=grid sbatch --array=0-9 aau/famos_calib/fcal.sbatch         (fold = id / 2, metodo icp | now = id % 2)
#   FC_STEP=eval sbatch -c 8 aau/famos_calib/fcal.sbatch                (scelte su calibrazione, metriche di test)
# Dati e intermedi in datasets/FAMOS/calib (fuori da git).
set -euo pipefail
export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 UGT_THREADS=1
STEP="${FC_STEP:?FC_STEP: prep | gpu | grid | eval}"
W="${SLURM_CPUS_PER_TASK:-8}"
D=aau/famos_calib
CAL=datasets/FAMOS/calib
UG=(v3_work/unified_gt/run.sh -u)
O3=(aau/outlineB/run_o3d.sh -u)
AA=(aau/run.sh -u)
MI=(aau/recon/run_recon.sh mica -u)
SH=8
echo "[fcal] host=$(hostname) job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_TASK_ID:-} passo=$STEP cpu=$W gpu=${CUDA_VISIBLE_DEVICES:-} $(date +%T)"
t0=$(date +%s)
detect() {  # $1 raw | patch: SH rilevatori in parallelo su blocchi contigui, poi ricuciti
    local q pids=()
    "${UG[@]}" $D/fcal.py det-split --kind "$1" --shards $SH
    for (( q = 0; q < SH; q++ )); do
        ORT_THREADS=$(( W / SH )) "${MI[@]}" $D/det_threads.py aau/competitors/protocolli/pr_noise_detect.py \
            --dir "$CAL/$1_det/s$q" \
            > "$CAL/$1_det/s$q.log" 2>&1 & pids+=($!)
    done
    for q in "${pids[@]}"; do wait "$q"; done
    "${UG[@]}" $D/fcal.py det-merge --kind "$1" --shards $SH
}
ops() {  # areanorm k_eig 128, come ei.sbatch
    local i pids=()
    mkdir -p "$2"
    for (( i = 0; i < W; i++ )); do
        "${AA[@]}" v2_work/potential/areanorm_operators.py --input-dir "$1" --output-dir "$2" --k-eig 128 \
            --shard "$i/$W" > /dev/null & pids+=($!)
    done
    for i in "${pids[@]}"; do wait "$i"; done
    [[ $(find "$1" -name "*.npz" | wc -l) == $(find "$2" -name "*.npz" | wc -l) ]] || { echo "operatori incompleti" >&2; exit 1; }
}
case "$STEP" in
    prep)  # ripartibile: salta selezione, render e rilevazione della registrazione gia' fatti
        [[ -f $CAL/reg_selected.npz ]] || "${UG[@]}" $D/fcal.py select --workers "$W"
        [[ -f $CAL/raw_det/cams.npz ]] || "${UG[@]}" $D/fcal.py render-raw --workers "$W"
        [[ -f $CAL/raw_det/lmk2d.npz ]] || detect raw
        "${UG[@]}" $D/fcal.py patch --workers "$W"
        "${UG[@]}" $D/fcal.py render-patch --workers "$W"
        detect patch
        "${UG[@]}" $D/fcal.py lmk --workers "$W"
        "${UG[@]}" $D/fcal.py folds
        ops "$CAL/npz" "$CAL/ops"
        "${AA[@]}" v3_work/trainer/tools/build_scale_table.py --view-dir "$CAL/npz" --domain famos \
            --out "$CAL/scale_table.npz" --workers "$W" ;;
    gpu)
        export AAU_NV=--nv WBES_V3_FACTORIZED_OUT=full
        export WBES_V3_SCALE_TABLES="$WBES_ROOT/$CAL/scale_table.npz:$WBES_ROOT/datasets/FAMOS/esterna_id/eval/scale_table.npz"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
        for m in factorized_s1234 factorized_s2345; do
            ck=$("${UG[@]}" -c "import sys; sys.path.insert(0, 'aau/esterna_famos_id'); import ei; print(ei.CKPT['$m'])")
            "${AA[@]}" v3_work/trainer/eval_v3.py -- $D/fcal.py embed --model "$m" --checkpoint "$ck"
        done ;;
    grid)
        id="${SLURM_ARRAY_TASK_ID:?serve --array}"
        fold=$(( id / 2 )); meth=$([[ $(( id % 2 )) == 0 ]] && echo icp || echo now)
        if [[ $meth == now ]]; then R=("${UG[@]}"); else R=("${O3[@]}"); fi
        "${R[@]}" $D/fcal.py grid --fold "$fold" --method "$meth" --workers "$W" ;;
    eval) "${UG[@]}" $D/fcal.py eval ;;
    *) echo "passo sconosciuto: $STEP" >&2; exit 2 ;;
esac
echo "[fcal] OK passo=$STEP in $(( $(date +%s) - t0 ))s $(date +%T)"
