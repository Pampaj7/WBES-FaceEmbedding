#!/usr/bin/env bash
#SBATCH --job-name=wbes-bp
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#
# Concorrenti parametrici e varifold (aau/baselines_param/bp.py; protocollo
# aau/runs/evidence/baselines_param/PROTOCOL.md). Un passo per job:
#
#   BP_STEP=fit BP_ARGS="--models gnm"                                   sbatch aau/baselines_param/bp.sbatch
#   BP_STEP=varifold BP_ARGS="hifi3d --device cuda" sbatch --gres=gpu:l40s:1 aau/baselines_param/bp.sbatch
#   BP_STEP=paired                                                       sbatch aau/baselines_param/bp.sbatch
#   BP_STEP=pilot_e1 BP_ARGS="--models gnm"                              sbatch aau/baselines_param/bp.sbatch
#   BP_STEP=fit_e1 BP_ARGS="--models gnm --views hifi3d"                 sbatch aau/baselines_param/bp.sbatch
#   BP_STEP=paired BP_ARGS="--e1"                                        sbatch aau/baselines_param/bp.sbatch
#   BP_STEP=e2 BP_ARGS="crop --models gnm"                               sbatch aau/baselines_param/bp.sbatch
#   BP_STEP=e2_heldout BP_ARGS="--models gnm,flame2023"                  sbatch aau/baselines_param/bp.sbatch
#   BP_STEP=paired_e2                                                    sbatch aau/baselines_param/bp.sbatch
#   BP_STEP=cost_ops / cost_pairs                                        sbatch aau/baselines_param/bp.sbatch
#   BP_STEP=cost_forward                          sbatch --gres=gpu:l40s:1 aau/baselines_param/bp.sbatch
#
# I passi *_e1 sono dell'emendamento 1 (POST HOC, PROTOCOL_emendamento_1.md): pilota sui soggetti non valutati,
# poi varianti A e B ed errore di superficie (bp_fit_e1.py), poi i delta con le colonne nuove.
# I passi e2* sono dell'emendamento 2 (POST HOC, PROTOCOL_emendamento_2.md): bp_e2.py (crop, pilot, time, sens;
# e2_heldout mette in scena gli held-out sintetici su /tmp con fact_calib.py stage, poi bp_e2.py heldout), poi i delta
# (bp_paired_e2.py). /tmp e' RAM e conta contro --mem: gli held-out grezzi occupano qualche GB.
#
# fit e varifold nel venv con open3d (aau/outlineB/run_o3d.sh, un thread BLAS per processo); paired nel venv della
# GT unificata (importa fact_paired e blmm_eval). Partizione prioritized senza GPU (il nodo cpu e' in fail), tranne il
# varifold delle viste a 500 mesh: una L40S (QoS normal).
# TEMPI (pilota del 10 ottobre, soggetti NON valutati): NICP 1.6-2.5 s per mesh (GNM, 4096 punti) e 0.4-0.9 s (FLAME),
# regressione 0.4 s (GNM) e 7-9 s (FLAME, ricerca della mandibola); varifold 7.8 ms per prodotto interno su L40S
# (blocco 8192), ~17 min per vista a 500 mesh; 1.2 s su un core.
set -euo pipefail

STEP="${BP_STEP:?BP_STEP: fit | varifold | paired | pilot_e1 | fit_e1 | e2 | e2_heldout | paired_e2 | cost_*}"
W="${SLURM_CPUS_PER_TASK:-8}"
export AAU_NV=""
[[ "${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}" != "" ]] && export AAU_NV="--nv"
echo "[bp] host=$(hostname) job=${SLURM_JOB_ID:-none} passo=$STEP args=${BP_ARGS:-}"
t0=$(date +%s)
case "$STEP" in
    fit)      aau/outlineB/run_o3d.sh aau/baselines_param/bp_fit.py --workers "$W" ${BP_ARGS:-} ;;
    varifold) aau/outlineB/run_o3d.sh aau/baselines_param/bp_varifold.py --block 8192 ${BP_ARGS} ;;
    pilot_e1) aau/outlineB/run_o3d.sh aau/baselines_param/bp_fit_e1.py --pilot --workers "$W" ${BP_ARGS:-} ;;
    fit_e1)   aau/outlineB/run_o3d.sh aau/baselines_param/bp_fit_e1.py --workers "$W" ${BP_ARGS:-} ;;
    paired)   UGT_THREADS=1 v3_work/unified_gt/run.sh aau/baselines_param/bp_paired.py --workers "$W" ${BP_ARGS:-} ;;
    e2)       aau/outlineB/run_o3d.sh aau/baselines_param/bp_e2.py ${BP_ARGS} --workers "$W" ;;
    e2_heldout)
        STAGE="/tmp/wbes_bp_e2_${SLURM_JOB_ID:-manual}"
        trap 'rm -rf "$STAGE"' EXIT
        aau/run.sh v3_work/trainer/tools/fact_calib.py stage --in-dir "$STAGE/in" --table "$STAGE/scale_table.npz"
        echo "[bp] stage: $(find "$STAGE/in" -name '*.npz' | wc -l) mesh, $(du -sh "$STAGE" | cut -f1)"
        aau/outlineB/run_o3d.sh aau/baselines_param/bp_e2.py heldout --in-dir "$STAGE/in" \
            --table "$STAGE/scale_table.npz" --workers "$W" ${BP_ARGS:-} ;;
    paired_e2) UGT_THREADS=1 v3_work/unified_gt/run.sh aau/baselines_param/bp_paired_e2.py --workers "$W" ${BP_ARGS:-} ;;
    cost_ops)
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 aau/run.sh aau/baselines_param/bp_cost.py ops --workers "$W" ${BP_ARGS:-} ;;
    cost_pairs) aau/outlineB/run_o3d.sh aau/baselines_param/bp_cost.py pairs ;;
    cost_forward)   # la catena di zs_embed del passo form su HIFI3D (eval_body.sh step_form, eval_common.sh)
        for key in factorized_s1234 ctrlfr_s1234; do
            ck=$(ls aau/runs/evidence/trainer_v3/ablations/c3f_runs/${key%_s1234}/v3_*/checkpoints/epoch072_ema.pth)
            WBES_V3_SCALE_TABLES=aau/runs/evidence/trainer_v3/factorized/scale_tables/hifi3d_eval.npz \
            WBES_V3_FACTORIZED_OUT=full WBES_EVAL_SCENARIOS=clean aau/run.sh v3_work/trainer/eval_v3.py -- \
                aau/baselines_param/bp_cost.py forward --key "$key" --model_path "$ck" --checkpoint_selector best_by_clean \
                --data_dir datasets/V3_OPS_CACHE/8e8f81d5f0204394 --dist_npz datasets/HIFI3D/eval_view/gt_matrix.npz \
                --subject_split eval --eval_fraction 0.2 --max_subjects 500 --max_meshes_per_subject_eval 10 \
                --preload_workers 8 --pair_mode cross_topology --rigid_rot_deg 20 --rigid_trans_scale 0.05 \
                --rigid_rot_deg_min 1 --rigid_trans_scale_min 0.002 --chamfer_batch_pairs 256 \
                --chamfer_cache_verts force --chamfer_cache_verts_max_mb 4096 --seed 1234 --subject_split all \
                --max_subjects 0
        done ;;
    *) echo "passo sconosciuto: $STEP" >&2; exit 2 ;;
esac
echo "[bp] $STEP: $(( $(date +%s) - t0 )) s"
