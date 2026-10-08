#!/usr/bin/env bash
#SBATCH --job-name=wbes-ir-template
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#
# Riconoscimento in dominio, baseline "NICP su template" (aau/indomain/ir_template.py, protocollo
# revisione 1 punto B): template per dominio e iscrizione di ogni mesh di bfm, rexpr e ict992.
# Legge le mesh dalle viste (CephFS): solo verts, e una volta per mesh.
#
#   aau/submit.sh indomain/ir_template.sbatch
#
# TEMPI: smoke 2.3-2.6 s per mesh a worker (8 worker); 648 + 534 + 5952 = 7134 mesh, ~40 min a 8 core.
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
echo "[ir-template] host=$(hostname) job=${SLURM_JOB_ID:-none}"
"$AAU_DIR/outlineB/run_o3d.sh" aau/indomain/ir_template.py --workers "${SLURM_CPUS_PER_TASK:-8}" ${WBES_IR_TEMPLATE_ARGS:-}
echo "[ir-template] OK"
