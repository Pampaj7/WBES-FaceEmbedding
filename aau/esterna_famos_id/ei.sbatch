#!/usr/bin/env bash
#SBATCH --job-name=wbes-ei
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#
# Validazione esterna su FaMoS TEST, riconoscimento dell'identita' (aau/esterna_famos_id/ei.py): un passo per job.
#
#   EI_STEP=select sbatch aau/esterna_famos_id/ei.sbatch
#   EI_STEP=prep EI_SET=pilot sbatch aau/esterna_famos_id/ei.sbatch          (pilota della preparazione)
#   EI_STEP=prep EI_SET=eval  sbatch aau/esterna_famos_id/ei.sbatch
#   EI_STEP=gpu  sbatch --gres=gpu:l40s:1 aau/esterna_famos_id/ei.sbatch      (operatori, nostro, ULRSSM)
#   EI_STEP=cpu  sbatch --cpus-per-task=64 aau/esterna_famos_id/ei.sbatch     (ICP mm e cs, NoW, GW)
#   EI_STEP=eval sbatch aau/esterna_famos_id/ei.sbatch                        (distanze del nostro, metriche, results.md)
# Dati e intermedi in datasets/FAMOS/esterna_id (fuori da git).
set -euo pipefail
export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 UGT_THREADS=1
STEP="${EI_STEP:?EI_STEP: select | prep | gpu | cpu | eval}"
S="${EI_SET:-eval}"
W="${SLURM_CPUS_PER_TASK:-8}"
D=aau/esterna_famos_id
DATA=datasets/FAMOS/esterna_id
UG=(v3_work/unified_gt/run.sh -u)
O3=(aau/outlineB/run_o3d.sh -u)
AA=(aau/run.sh -u)
MI=(aau/recon/run_recon.sh mica -u)
GPR=(aau/competitors/gw_percettivi/gp_run.sh -u)
echo "[ei] host=$(hostname) job=${SLURM_JOB_ID:-none} passo=$STEP set=$S cpu=$W gpu=${CUDA_VISIBLE_DEVICES:-} $(date +%T)"
t0=$(date +%s)
detect() {  # onnxruntime con tutti i core del job (come cc.sbatch)
    OMP_NUM_THREADS=$W "${MI[@]}" aau/competitors/protocolli/pr_noise_detect.py --dir "$1"
}
ops() {  # $1 cartella delle mesh, $2 cartella degli operatori: areanorm k_eig 128, come step_famos di eval_body.sh
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
    select) "${UG[@]}" $D/ei.py select ;;
    prep)
        "${UG[@]}" $D/ei.py render-raw --set "$S" --workers "$W"
        detect "$DATA/$S/raw_det"
        "${UG[@]}" $D/ei.py patch --set "$S" --workers "$W"
        "${UG[@]}" $D/ei.py render-patch --set "$S" --workers "$W"
        detect "$DATA/$S/patch_det"
        "${UG[@]}" $D/ei.py lmk --set "$S" --workers "$W"
        "${UG[@]}" $D/ei.py prep-report --set "$S" ;;
    gpu)
        "${UG[@]}" $D/ei.py check-links
        ops "$DATA/$S/npz" "$DATA/$S/ops"
        ops "$DATA/check/npz" "$DATA/check/ops"
        "${AA[@]}" v3_work/trainer/tools/build_scale_table.py --view-dir "$DATA/$S/npz" --domain famos \
            --out "$DATA/$S/scale_table.npz" --workers "$W"
        export AAU_NV=--nv WBES_V3_FACTORIZED_OUT=full
        export WBES_V3_SCALE_TABLES="$WBES_ROOT/$DATA/$S/scale_table.npz:$AAU_RUNS/evidence/trainer_v3/factorized/scale_tables/famos_test.npz"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
        for m in factorized_s1234 factorized_s2345; do
            ck=$("${UG[@]}" -c "import sys; sys.path.insert(0, '$D'); import ei; print(ei.CKPT['$m'])")
            "${AA[@]}" v3_work/trainer/eval_v3.py -- $D/ei.py embed --set "$S" --model "$m" --checkpoint "$ck"
        done
        "${AA[@]}" $D/ei.py ulrssm --set "$S" --workers "$W" ;;
    cpu)
        "${O3[@]}" $D/ei.py icp --set "$S" --workers "$W"
        "${UG[@]}" $D/ei.py now --set "$S" --workers "$W"
        "${GPR[@]}" $D/ei.py gw --set "$S" --workers "$W" ;;
    eval)
        "${UG[@]}" $D/ei.py ours --set "$S"
        "${UG[@]}" $D/ei.py eval --set "$S" ;;
    *) echo "passo sconosciuto: $STEP" >&2; exit 2 ;;
esac
echo "[ei] OK passo=$STEP set=$S in $(( $(date +%s) - t0 ))s $(date +%T)"
