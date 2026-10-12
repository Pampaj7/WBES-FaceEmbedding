#!/usr/bin/env bash
#SBATCH --job-name=wbes-gp
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#
# Gromov-Wasserstein e feature percettive su render (aau/competitors/gw_percettivi/): un passo per job, solo CPU.
#
#   GP_STEP=posecheck GP_VIEW=hifi3d sbatch gp.sbatch           (la posa e' canonica? nessuna GT)
#   GP_STEP=perc      GP_VIEW=pilot_hifi3d sbatch gp.sbatch     (render nelle 3 varianti, DINOv2, CLIP, DreamSim, LPIPS)
#   GP_STEP=perc_pilot_eval sbatch gp.sbatch                    (variante dichiarata: GT dei NON valutati)
#   GP_STEP=perc      GP_VIEW=hifi3d sbatch gp.sbatch
#   GP_STEP=gw_prep   GP_VIEW=pilot_hifi3d sbatch gp.sbatch     (campioni, pesi, geodetiche)
#   GP_STEP=gw_cost   sbatch --cpus-per-task=8 gp.sbatch        (tempi per configurazione)
#   GP_STEP=gw_pilot  sbatch --cpus-per-task=64 gp.sbatch       (distanze del pilota, tutte le configurazioni)
#   GP_STEP=pilot_eval sbatch gp.sbatch                         (scelta con la GT dei NON valutati)
#   GP_STEP=gw_run    GP_VIEW=hifi3d GP_SHARD=0 GP_NSHARDS=2 sbatch --cpus-per-task=64 gp.sbatch
#   GP_STEP=gw_merge  GP_VIEW=hifi3d sbatch gp.sbatch
#   (gw_pilot, gw_run, gw_merge: GP_ARGS="--configs ... --name ..." / "--config euclid|256|cgw2")
#   GP_STEP=eval      sbatch --mem=192G gp.sbatch               (Spearman con IC; LEGGE LE GT di test)
# Dominio nuovo: GP_SPEC=<json> (gp_common.py). Uscite in aau/runs/evidence/competitors_new/ (npz fuori da git).
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
STEP="${GP_STEP:?GP_STEP: posecheck | perc | perc_pilot_eval | gw_prep | gw_cost | gw_pilot | pilot_eval | gw_run | gw_merge | eval}"
W="${SLURM_CPUS_PER_TASK:-8}"
export AAU_NV=""
SPEC=()
[[ -n "${GP_SPEC:-}" ]] && SPEC=(--spec "$GP_SPEC")
D=aau/competitors/gw_percettivi
echo "[gp] host=$(hostname) job=${SLURM_JOB_ID:-none} passo=$STEP vista=${GP_VIEW:-} cpu=$W $(date +%T)"
t0=$(date +%s)
case "$STEP" in
    posecheck) $D/gp_run.sh -u $D/gp_percettivi.py posecheck --view "$GP_VIEW" "${SPEC[@]}" --workers "$W" ;;
    perc)      GP_THREADS="$W" $D/gp_run.sh -u $D/gp_percettivi.py run --view "$GP_VIEW" "${SPEC[@]}" --workers "$W" ;;
    gw_prep)   $D/gp_run.sh -u $D/gp_gw.py prep --view "$GP_VIEW" "${SPEC[@]}" --workers "$W" ${GP_ARGS:-} ;;
    gw_cost)   $D/gp_run.sh -u $D/gp_gw.py cost --workers "$W" ;;
    gw_pilot)  $D/gp_run.sh -u $D/gp_gw.py pilot --workers "$W" ${GP_ARGS:-} ;;
    pilot_eval) UGT_THREADS=1 v3_work/unified_gt/run.sh -u $D/gp_eval.py pilot-gw ;;
    perc_pilot_eval) UGT_THREADS=1 v3_work/unified_gt/run.sh -u $D/gp_eval.py pilot-perc ;;
    gw_run)    $D/gp_run.sh -u $D/gp_gw.py run --view "$GP_VIEW" --shard "${GP_SHARD:-0}" --nshards "${GP_NSHARDS:-1}" \
                   --workers "$W" ${GP_ARGS:-} ;;
    gw_merge)  $D/gp_run.sh -u $D/gp_gw.py merge --view "$GP_VIEW" "${SPEC[@]}" ${GP_ARGS:-} ;;
    eval)      UGT_THREADS=1 v3_work/unified_gt/run.sh -u $D/gp_eval.py spearman --workers "$W" ;;
    *) echo "passo sconosciuto: $STEP" >&2; exit 2 ;;
esac
echo "[gp] OK passo=$STEP in $(( $(date +%s) - t0 ))s"
