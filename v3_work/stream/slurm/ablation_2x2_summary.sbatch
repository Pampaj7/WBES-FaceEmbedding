#!/usr/bin/env bash
#SBATCH --job-name=wbes-abl2x2-summary
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#
# Ablazione 2x2 (aau/runs/evidence/stream/ablation_2x2/PROTOCOL.md, sez. 4-5): Spearman appaiati, effetti e letture
# (v3_work/stream/eval_ablation.py summary), dopo le valutazioni delle quattro celle (ablation_2x2_eval.sbatch).
# Uscite in aau/runs/evidence/stream/ablation_2x2: spearman.csv, effects.csv, readings.json, controls.json, results.md.
#   sbatch --dependency=afterok:<eval A>:<eval B>:<eval C>:<eval D> v3_work/stream/slurm/ablation_2x2_summary.sbatch
# Argomenti in piu' passati a eval_ablation.py (prova: --root <dir> --epoch <N> --n-boot 50 --out <dir>).
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
W="${SLURM_CPUS_PER_TASK:-8}"
echo "[abl-summary] host=$(hostname) job=${SLURM_JOB_ID:-none} $(date +%T) argomenti: $*"
UGT_THREADS=1 v3_work/unified_gt/run.sh v3_work/stream/eval_ablation.py summary --workers "$W" "$@"
