#!/usr/bin/env bash
#SBATCH --job-name=wbes-ugt
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#
# GT unificata (D2 / E8), tutta la catena su CPU. Venv: setup_env.sbatch.
#
#   sbatch v3_work/unified_gt/pipeline.sbatch                 # tutto
#   UGT_STEPS="shapes evidence report" sbatch v3_work/unified_gt/pipeline.sbatch
#
#   region    flame_region.py   regione del volto in topologia FLAME
#   corr      correspond.py     landmark, NICP, mappe baricentriche (8 domini, ~2 min l'uno)
#   unified   unified.py        regione comune, canonical_transforms.json, media globale
#   shapes    shapes.py         s_i di ogni identita' (ICT 55.000, GNM 10.100, ...) e GT dei set di test
#   native    native_gt.py      GT intermedie sulla patch nativa (da maxabs all'unificata)
#   ccheck    check_canonical.py canonical_transforms.json applicato a mesh di dati dal disco
#   evidence  evidence.py       (a) (b) (c)
#   methods   eval_methods.py   metodi esistenti con la GT unificata (richiesta del critic)
#   report    report.py         aau/runs/evidence/e8/summary.md
#
# Memoria: il core FaceScape (4.9 GB) si legge una volta (correspond), le forme ICT in float32 ~1 GB.
set -euo pipefail

STEPS="${UGT_STEPS:-region corr unified shapes native ccheck evidence methods report}"
RUN="v3_work/unified_gt/run.sh"
export UGT_THREADS="${UGT_THREADS:-8}"
echo "[ugt] host=$(hostname) job=${SLURM_JOB_ID:-none} steps: $STEPS"
for step in $STEPS; do
    t0=$(date +%s)
    case "$step" in
        region)   $RUN v3_work/unified_gt/flame_region.py ;;
        corr)     $RUN v3_work/unified_gt/correspond.py ;;
        unified)  $RUN v3_work/unified_gt/unified.py ;;
        shapes)   $RUN v3_work/unified_gt/shapes.py ;;
        native)   $RUN v3_work/unified_gt/native_gt.py ;;
        ccheck)   $RUN v3_work/unified_gt/check_canonical.py ;;
        evidence) $RUN v3_work/unified_gt/evidence.py ;;
        methods)  UGT_THREADS=1 $RUN v3_work/unified_gt/eval_methods.py ;;
        report)   $RUN v3_work/unified_gt/report.py ;;
        *) echo "passo sconosciuto: $step" >&2; exit 2 ;;
    esac
    echo "[ugt] $step: $(( $(date +%s) - t0 )) s"
done
echo "[ugt] OK"
