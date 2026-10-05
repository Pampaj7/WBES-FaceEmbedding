#!/usr/bin/env bash
#SBATCH --job-name=wbes-bl-rank
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#
# Tabella 2 estesa: Spearman/Pearson + CI bootstrap da tutte le matrici presenti.
#   aau/submit.sh baselines/rank.sbatch
#   WBES_BL_ARGS="--subject-set facebench_first100 --out-root aau/runs/baselines_fb100" \
#     aau/submit.sh baselines/rank.sbatch
#
# La riga di controllo bbox_proxy si calcola qui dentro: sono quattro numeri per mesh, non
# merita un job suo. WBES_BL_ARGS vale per tutti e due gli script (hanno gli stessi
# --subject-set/--out-root/--settings).
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

"$AAU_DIR/run.sh" aau/baselines/proxy_matrix.py ${WBES_BL_ARGS:-}
"$AAU_DIR/run.sh" aau/baselines/rank_from_matrix.py ${WBES_BL_ARGS:-}
