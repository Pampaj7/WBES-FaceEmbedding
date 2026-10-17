#!/usr/bin/env bash
#SBATCH --job-name=wbes-me
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#
# Valutazione dei due modelli massivi (aau/massive_eval/): un passo per job. ME_MODEL = main | open.
#
#   ME_STEP=calib ME_MODEL=main sbatch --gres=gpu:a10:1 aau/massive_eval/me.sbatch      (c sull'held-out, PRIMA di tutto)
#   ME_STEP=calib_paper ME_MODEL=main sbatch --gres=gpu:a10:1 aau/massive_eval/me.sbatch
#       (controllo: c della catena del paper, fact_calib.py calib-ckpt sugli held-out bfm/ict/gnm di C3M)
#   ME_STEP=dev ME_MODEL=main sbatch --gres=gpu:a10:1 --mem=96G --time=03:00:00 aau/massive_eval/me.sbatch
#       (embedding [s, u] dei test di sviluppo: passi form, fvn, famos di eval_body.sh, invariato)
#   ME_STEP=famos_id_embed ME_MODEL=main sbatch --gres=gpu:a10:1 aau/massive_eval/me.sbatch   (dopo dev: latenti di ei)
#   ME_STEP=famos_id_eval sbatch aau/massive_eval/me.sbatch                                    (distanze e metriche)
#   ME_STEP=matrix sbatch --cpus-per-task=64 --mem=192G --time=03:00:00 aau/massive_eval/me.sbatch   (LEGGE LE GT)
#   ME_STEP=ava_embed sbatch --gres=gpu:a10:1 --mem=48G aau/massive_eval/me.sbatch
#       (prima: python3 aau/massive_eval/me_ava.py prep; i due modelli e il controllo C3M e123 sulle viste di
#       valutazione di Ava-256; GPU della classe embed.gpu dello spec, A10: me_ava.py)
#   ME_STEP=ava_eval sbatch --cpus-per-task=64 --mem=192G --time=03:00:00 aau/massive_eval/me.sbatch  (LEGGE LE GT)
# Uscite: aau/runs/evidence/massive_eval/ (json, csv, md) e aau/runs/massive_eval/ (npz, fuori da git).
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
STEP="${ME_STEP:?ME_STEP: calib | calib_paper | dev | famos_id_embed | famos_id_eval | matrix | ava_embed | ava_eval}"
M="${ME_MODEL:-main}"
W="${SLURM_CPUS_PER_TASK:-8}"
export AAU_NV=""
[[ "${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}" != "" ]] && export AAU_NV="--nv"
echo "[me] host=$(hostname) job=${SLURM_JOB_ID:-none} passo=$STEP modello=$M cpu=$W gpu=${CUDA_VISIBLE_DEVICES:-} $(date +%T)"
t0=$(date +%s)
declare -A RUN=([main]=main_a100 [open]=release_open) EP=([main]=330 [open]=230) ARM=([main]=factorizedmassmain [open]=factorizedmassopen)
RR="$AAU_RUNS/massive_release/${RUN[$M]}/runs"
CK="$(ls "$RR"/v3_*/checkpoints/epoch${EP[$M]}_ema.pth)"
case "$STEP" in
    calib)
        aau/run.sh -u aau/massive_eval/me_calib.py --model "$M" ;;
    calib_paper)
        WBES_CALIB_CKPT="$CK" WBES_CALIB_OUT="$AAU_RUNS/evidence/massive_eval/calib_paper/$M" WBES_CALIB_ARM=factorized \
            bash v3_work/trainer/slurm/calib_ckpt.sbatch ;;          # embeddings.npz accanto (fuori da git)
    dev)
        # eval_body.sh invariato (uscite in c3f_eval e faceverse_neutral sotto il tag del braccio, come C3M): form
        # (HIFI3D, dev FaceScape neutra e con espressioni, FaceVerse con espressioni), fvn (FaceVerse neutra), famos
        WBES_V3_ARM="${ARM[$M]}" WBES_V3_EVAL_RR="$RR" WBES_V3_EVAL_EPOCHS="${EP[$M]}" WBES_V3_EVAL_STEPS="form fvn famos" \
            bash v3_work/trainer/ablations/c3f/eval_body.sh ;;
    famos_id_embed)                              # come il passo gpu di ei.sbatch (operatori e tabella gia' su disco)
        D=datasets/FAMOS/esterna_id
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 WBES_V3_FACTORIZED_OUT=full \
        WBES_V3_SCALE_TABLES="$WBES_ROOT/$D/eval/scale_table.npz:$AAU_RUNS/evidence/trainer_v3/factorized/scale_tables/famos_test.npz" \
            aau/run.sh -u v3_work/trainer/eval_v3.py -- aau/massive_eval/me_famos_id.py embed --set eval --model "massive_$M" \
                --checkpoint "$CK" ;;
    famos_id_eval)
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 UGT_THREADS=1
        v3_work/unified_gt/run.sh -u aau/massive_eval/me_famos_id.py ours --set eval
        v3_work/unified_gt/run.sh -u aau/massive_eval/me_famos_id.py eval --set eval ;;
    matrix)
        UGT_THREADS=1 v3_work/unified_gt/run.sh -u aau/massive_eval/me_matrix.py --workers "$W" ;;
    ava_embed)                                   # conf_embed.py invariato, spec di me_ava.py prep
        CONF_SPEC=aau/runs/evidence/massive_eval/ava256/spec.json AAU_NV=--nv \
            aau/run.sh -u aau/ava256/conf_run.py aau/ava256/conf_embed.py run --spec aau/runs/evidence/massive_eval/ava256/spec.json ;;
    ava_eval)
        CONF_SPEC=aau/runs/evidence/massive_eval/ava256/spec.json UGT_THREADS=1 \
            v3_work/unified_gt/run.sh -u aau/ava256/conf_run.py aau/massive_eval/me_ava.py eval --workers "$W" ;;
    *) echo "passo sconosciuto: $STEP" >&2; exit 2 ;;
esac
echo "[me] OK passo=$STEP modello=$M in $(( $(date +%s) - t0 ))s $(date +%T)"
