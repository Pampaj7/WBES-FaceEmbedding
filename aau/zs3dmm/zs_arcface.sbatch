#!/usr/bin/env bash
#SBATCH --job-name=wbes-arcface-zs
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#
# ArcFace su render di sola geometria (pipeline di Multiface) sulle 600 mesh valutate di un
# dominio zero-shot: render ombreggiati + crop calibrato + embedding, poi normal map con lo stesso
# crop. Risultati in aau/runs/arcface_render_zs/<dominio>/{shaded,normals}/arcface_views.npz.
#
#   WBES_ARCFACE_DOMAIN=fv_expr aau/submit.sh zs3dmm/zs_arcface.sbatch
#   WBES_ARCFACE_DOMAIN=hifi3d  aau/submit.sh zs3dmm/zs_arcface.sbatch
#   WBES_ARCFACE_MODES=shaded ...     solo l'ombreggiato
#
# CPU e niente GPU: ArcFace e il detector girano su onnxruntime CPU, come in ws3a_perceptual.sbatch.
# Rotazione verso il frame del renderer scelta con --frame-check (png in <out>/shaded/frame_check):
# FaceVerse gia' alto -y / naso -z, HIFI3D Rx(180).
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

case "${WBES_ARCFACE_DOMAIN:-}" in
    fv_expr) VIEW=datasets/FACEVERSE_ZS/expr_view/npz; ROT=none
             SUBJ=aau/runs/ws_faceverse_expr/data_736f96956a/joint_flip_topology/subjects.json ;;
    hifi3d)  VIEW=datasets/HIFI3D/eval_view/npz; ROT=x180
             SUBJ=aau/runs/ws_hifi3d/data_328f2bfc1a/joint_frame-xmymz_flip/subjects.json ;;
    *) echo "ERRORE: WBES_ARCFACE_DOMAIN='${WBES_ARCFACE_DOMAIN:-}' (fv_expr|hifi3d)" >&2; exit 2 ;;
esac
OUT="$AAU_RUNS/arcface_render_zs/$WBES_ARCFACE_DOMAIN"
CORES="${SLURM_CPUS_PER_TASK:-8}"

# Un thread per processo: il parallelismo e' sui processi (vedi ws3a_perceptual.sbatch).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "[arcface-zs] dominio=$WBES_ARCFACE_DOMAIN host=$(hostname) job=${SLURM_JOB_ID:-none} out=$OUT"
for mode in ${WBES_ARCFACE_MODES:-shaded normals}; do
    extra=()
    [[ "$mode" == normals ]] && extra=(--calibration-from "$OUT/shaded/renders/arcface_align.json")
    "$AAU_DIR/baselines/run_bl.sh" aau/zs3dmm/zs_arcface_render.py --view-dir "$VIEW" --subjects-json "$SUBJ" \
        --out-root "$OUT/$mode" --base-rotation "$ROT" --mode "$mode" --workers "$CORES" "${extra[@]}"
done
echo "[arcface-zs] OK"
