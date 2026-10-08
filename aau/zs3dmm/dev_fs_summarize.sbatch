#!/usr/bin/env bash
#SBATCH --job-name=wbes-devfs-sum
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#
# Set di sviluppo FaceScape: tabelle di dev_fs_summarize.py dai risultati delle due viste (impronta
# dei dati calcolata qui, come negli sbatch di eval) in aau/runs/evidence/dev_facescape.
#
#   aau/submit.sh zs3dmm/dev_fs_summarize.sbatch
set -euo pipefail

export AAU_NV=""
cd "${WBES_ROOT:-$PWD}"
fp_of() {  # <0|1> -> $ZS_RUNS/data_<fp> della vista
    ( export WBES_ZS_EXPR="$1"; source aau/zs3dmm/dev_facescape_env.sh >/dev/null
      source "$WBES_ROOT/aau/zs3dmm/zs_env.sh"; echo "$ZS_RUNS/data_$(zs_data_fp)" )
}
RUNS_N="$(fp_of 0)"
RUNS_E="$(fp_of 1)"
source aau/zs3dmm/dev_facescape_env.sh
echo "[devfs-sum] host=$(hostname) job=${SLURM_JOB_ID:-none} neutra=$RUNS_N espressioni=$RUNS_E"
"$AAU_DIR/run.sh" aau/zs3dmm/dev_fs_summarize.py --runs-neutral "$RUNS_N" --runs-expr "$RUNS_E" \
    --root "$WBES_FV_ROOT" --out-dir "$AAU_RUNS/evidence/dev_facescape" --seed 1234 --eval-seed 1234
