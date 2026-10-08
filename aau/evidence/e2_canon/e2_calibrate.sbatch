#!/usr/bin/env bash
#SBATCH --job-name=wbes-e2-calib
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#
# E2, passo 0 (solo CPU): faccia media ICT del training e calibrazione della canonicalizzazione sui
# domini di training (soggetti held-out), con la soglia di fallimento. Nessun dominio di test.
#
#   aau/submit.sh evidence/e2_canon/e2_calibrate.sbatch
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export AAU_NV= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT="$AAU_RUNS/evidence/e2"
REF="$OUT/reference_ict_mean.npz"
W="${SLURM_CPUS_PER_TASK:-8}"
[[ -f "$REF" ]] || "$AAU_DIR/run.sh" aau/evidence/e2_canon/canon.py reference --out "$REF" --workers "$W"
"$AAU_DIR/run.sh" aau/evidence/e2_canon/canon.py calibrate --reference "$REF" --out-dir "$OUT/calibration" \
    --n-subjects "${WBES_E2_CAL_N:-40}" --workers "$W"
"$AAU_DIR/run.sh" aau/evidence/e2_canon/calib_report.py --dir "$OUT/calibration"
