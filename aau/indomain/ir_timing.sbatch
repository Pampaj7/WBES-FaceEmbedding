#!/usr/bin/env bash
#SBATCH --job-name=wbes-ir-timing
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --exclusive
#
# Riconoscimento in dominio, tempi (aau/indomain/ir_timing.py): modello, faceBench e ArcFace in
# fila nello stesso job, cioe' sullo stesso nodo e con lo stesso hardware. Parti CPU a un thread.
# Vuole gli embedding del congiunto (ir_embed.sbatch) e la calibrazione ArcFace degli insiemi bfm
# e ict (ir_arcface.sbatch): lanciarlo con --dependency=afterok sui due.
#
#   aau/submit.sh indomain/ir_timing.sbatch --dependency=afterok:<embed>:<arcface>
#
# --cpus-per-task=4: tutto gira a un thread, ma il job non deve contendersi il core coi propri
# processi di contorno (singularity, I/O).
# --exclusive (revisione 1 del protocollo): la prima misura (job 1058495) girava su un nodo che
# ospitava un job faceBench da 48 core; qui il nodo e' tutto nostro. Vuole anche le iscrizioni del
# template (ir_template.sbatch) e la calibrazione di arcface_best (WBES_IR_ARCFACE_BEST=1).
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

OUT="$AAU_RUNS/indomain_recog/timing"
STAGE="${TMPDIR:-/tmp}/wbes_ir_timing_${SLURM_JOB_ID:-manual}"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN_DIR="mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_ks0_poolmeanmax_noise60_sig5e-4-2e-2_latentnoise_seed1234__9a81466d"

echo "[ir-timing] host=$(hostname) job=${SLURM_JOB_ID:-none} out=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
lscpu | grep -E "Model name|^CPU\(s\)" || true

for part in ${WBES_IR_TIMING_PARTS:-model facebench template arcface}; do
    case "$part" in
        model)
            (
                export WBES_CKPT="$AAU_RUNS/x3dmm_joint_bfm_ict_s1234_1019532/$RUN_DIR/checkpoints/best_by_xtopo_mesh_clean.pth"
                export WBES_DATA_DIR="$WBES_ROOT/datasets/WS2_CROSS3DMM/joint__bfm"
                export WBES_DIST_NPZ="$WBES_ROOT/datasets/JOINT_BFM_ICT/gt_matrix.npz"
                source "$WBES_ROOT/aau/eval_common.sh"
                "$AAU_DIR/run.sh" aau/indomain/ir_timing.py model --stage-dir "$STAGE" --out "$OUT/model.json" \
                    -- "${common_args[@]}" --subject_split all --max_subjects 0
            ) ;;
        facebench|template)
            AAU_NV="" "$AAU_DIR/outlineB/run_o3d.sh" aau/indomain/ir_timing.py "$part" --stage-dir "$STAGE" \
                --out "$OUT/$part.json" ;;
        arcface)
            AAU_NV="" "$AAU_DIR/baselines/run_bl.sh" aau/indomain/ir_timing.py arcface --stage-dir "$STAGE" \
                --out "$OUT/arcface.json" ;;
        *) echo "ERRORE: parte sconosciuta: $part (model|facebench|template|arcface)" >&2; exit 2 ;;
    esac
done
echo "[ir-timing] OK"
