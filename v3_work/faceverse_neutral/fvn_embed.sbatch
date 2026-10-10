#!/usr/bin/env bash
#SBATCH --job-name=wbes-fvn-embed
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:30:00
#
# FaceVerse NEUTRA (aau/runs/evidence/faceverse_neutral/PROTOCOL.md): embedding dei metodi a rete sulla eval_view.
#   0. tabella di scala della eval_view (tools/build_scale_table.py) -> faceverse_neutral/scale_tables/fv_eval.npz;
#   1. bracci a 21.096 passi (factorized, factorized2, ctrlfr; s1234, s2345) e C3M (epoca 205): passo ``fvn`` di
#      trainer/ablations/c3f/eval_body.sh (embedding [s, u] con la cache degli operatori, poi eval_factorized.py);
#   2. e108: aau/zs3dmm/zs_zeroshot.sbatch con WBES_ZS_PART=embed e _flip, la pipeline degli embedding con espressioni
#      (ws_faceverse_expr/.../scale_e108_flip_topology).
# Uscite in aau/runs/evidence/faceverse_neutral/{embed,form}. Ripartibile: gli stage con .done si saltano.
#
#   sbatch v3_work/faceverse_neutral/fvn_embed.sbatch        (o v3_work/faceverse_neutral/launch.sh)
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
FO="$AAU_RUNS/evidence/faceverse_neutral"
CK_E108="$AAU_RUNS/data_scale_runs/scale_bfm_ict_gnm_s1234_nocanon_noaug_20261007_1411/mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_ks0_poolmeanmax_noise60_sig5e-4-2e-2_latentnoise_seed1234__3167d36d/checkpoints/epoch108.pth"
mkdir -p "$FO/scale_tables"
echo "[fvn] host=$(hostname) job=${SLURM_JOB_ID:-none}"
[[ -f "$FO/scale_tables/fv_eval.npz" ]] || OMP_NUM_THREADS=1 AAU_NV= "$AAU_DIR/run.sh" v3_work/trainer/tools/build_scale_table.py \
  --view-dir datasets/FACEVERSE_ZS/eval_view/npz --domain faceverse --out "$FO/scale_tables/fv_eval.npz" \
  --workers "${SLURM_CPUS_PER_TASK:-16}"
export WBES_V3_EVAL_STEPS=fvn WBES_FVN_OUT="$FO"
FAILED=()
run_body() {  # come eval.sbatch: il corpo copiato su /tmp (si cancella da solo alla fine); retry dentro il corpo
  local B="/tmp/v3c3f_eval_body_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}.sh"
  cp v3_work/trainer/ablations/c3f/eval_body.sh "$B"
  bash "$B"
}
for arm in factorized factorized2 ctrlfr; do
  for seed in 1234 2345; do
    echo "[fvn] $(date +%T) $arm s$seed"
    WBES_V3_ARM=$arm WBES_V3_SEED=$seed WBES_V3_EVAL_EPOCHS=072 run_body || FAILED+=("$arm-s$seed")
  done
done
echo "[fvn] $(date +%T) C3M e205"
WBES_V3_ARM=factorizedc3m WBES_V3_EVAL_RR="$AAU_RUNS/evidence/trainer_v3/factorized/c3m/runs" WBES_V3_EVAL_EPOCHS=205 \
  run_body || FAILED+=(c3m)
e108() {
  ( unset WBES_ZS_EXPR
    export WBES_ZS_DOMAIN=fv WBES_ZS_FLIP_FACES=1 WBES_ZS_ARMS=scale_e108 WBES_ZS_PART=embed WBES_FV_RUNS="$FO/embed"
    export WBES_EVAL_SCENARIOS=clean WBES_ZS_CKPT_SCALE_E108="$CK_E108"
    bash aau/zs3dmm/zs_zeroshot.sbatch )
}
echo "[fvn] $(date +%T) e108"
ok=0
for t in 1 2 3; do  # come retry di eval_body.sh ("unknown userid" a tratti su nv-ai-04)
  e108 && { ok=1; break; }
  (( t < 3 )) && { echo "[fvn] $(date +%T) e108: tentativo $t fallito, riprovo fra 10 minuti"; sleep 600; }
done
(( ok )) || FAILED+=(e108)
echo "[fvn] $(date +%T) fine: falliti=${FAILED[*]:-nessuno}"
(( ${#FAILED[@]} == 0 ))
