#!/usr/bin/env bash
#SBATCH --job-name=wbes-arcface-sum
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#
# Tabella di riconoscimento di ArcFace su render (zs_arcface_summarize.py) per un dominio, poi
# ricompone aau/runs/arcface_render_zs/summary.md.
#
#   WBES_ARCFACE_DOMAIN=fv_expr aau/submit.sh zs3dmm/zs_arcface_summarize.sbatch --dependency=afterok:<arcface>
#   WBES_ARCFACE_DOMAIN=hifi3d  aau/submit.sh zs3dmm/zs_arcface_summarize.sbatch
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv

case "${WBES_ARCFACE_DOMAIN:-}" in
    fv_expr) RUNS=aau/runs/ws_faceverse_expr/data_736f96956a; VIEW=datasets/FACEVERSE_ZS/expr_view/npz
             JOINT="$RUNS/joint_flip_topology/zs_zeroshot"; REF=(--reference-csv aau/runs/ws_faceverse_expr/recognition.csv) ;;
    hifi3d)  RUNS=aau/runs/ws_hifi3d/data_328f2bfc1a; VIEW=datasets/HIFI3D/eval_view/npz
             JOINT="$RUNS/joint_frame-xmymz_flip_ranking/zs_zeroshot"; REF=() ;;
    *) echo "ERRORE: WBES_ARCFACE_DOMAIN='${WBES_ARCFACE_DOMAIN:-}' (fv_expr|hifi3d)" >&2; exit 2 ;;
esac
echo "[arcface-sum] dominio=$WBES_ARCFACE_DOMAIN host=$(hostname) job=${SLURM_JOB_ID:-none}"
"$AAU_DIR/run.sh" aau/zs3dmm/zs_arcface_summarize.py --domain "$WBES_ARCFACE_DOMAIN" --runs "$RUNS" \
    --view-dir "$VIEW" --joint-stage "$JOINT" "${REF[@]}" --eval-seed "${WBES_EVAL_SEED:-1234}"
