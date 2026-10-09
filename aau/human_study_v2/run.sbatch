#!/usr/bin/env bash
#SBATCH --job-name=wbes-hs2
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#
# Studio umano v2: tutta la catena su CPU, nel venv della GT unificata (open3d per il ray casting).
#
#   sbatch aau/human_study_v2/run.sbatch                       # tutto
#   HS2_STEPS="gt select" sbatch aau/human_study_v2/run.sbatch   # dopo la consegna di E12 (render solo se
#                                                                # il frame di BFM cambia: select lo controlla)
#
#   gt        gt_v2.py              GT di E12 sui 100 soggetti BFM REMESH held-out
#   render    render_v2.py          3 viste a scala assoluta, camera unica in mm
#   select    select_triplets_v2.py triplette informative -> triplets.json, docs/human_study_v2/
#   selftest  analyze_v2.py --self-test
#   power     power_v2.py           calcolo di potenza (simulazione)
set -euo pipefail

STEPS="${HS2_STEPS:-gt render select selftest}"
RUN="v3_work/unified_gt/run.sh"
export UGT_THREADS="${UGT_THREADS:-8}"
echo "[hs2] host=$(hostname) job=${SLURM_JOB_ID:-none} steps: $STEPS"
for step in $STEPS; do
    t0=$(date +%s)
    case "$step" in
        gt)       $RUN aau/human_study_v2/gt_v2.py ;;
        render)   $RUN aau/human_study_v2/render_v2.py ;;
        select)   $RUN aau/human_study_v2/select_triplets_v2.py ${HS2_SELECT_ARGS:-} ;;
        selftest) $RUN aau/human_study_v2/analyze_v2.py --self-test ;;
        power)    $RUN aau/human_study_v2/power_v2.py ${HS2_POWER_ARGS:-} ;;
        *) echo "passo sconosciuto: $step" >&2; exit 2 ;;
    esac
    echo "[hs2] $step: $(( $(date +%s) - t0 )) s"
done
