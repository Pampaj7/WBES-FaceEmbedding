#!/usr/bin/env bash
#SBATCH --job-name=wbes-be-render-bfm
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#
# Render di sola geometria dei 100 soggetti BFM held-out di WS1 con la pipeline di ArcFace
# (zs_arcface_render.py, invariata): ombreggiati + crop calibrato, poi normal map con lo stesso crop.
# Sono i render su cui vfm_embed.py calcola CLIP e DINOv2 per la Spearman in dominio
# (aau/runs/baselines_extra/protocol.md, punto c). subjects.json scritto prima, a mano (job 1060586).
#
#   aau/submit.sh baselines_extra/be_render_bfm.sbatch
#
# Frame: --frame-check 3 (job 1060586) -> `none` da' il volto dritto e convesso.
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

VIEW=datasets/REMESH/npz_data_topo_500
OUT="$AAU_RUNS/baselines_extra/bfm_heldout"
SUBJ="$OUT/subjects.json"
CORES="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "[be-render] host=$(hostname) job=${SLURM_JOB_ID:-none} out=$OUT"
for mode in shaded normals; do
    extra=()
    [[ "$mode" == normals ]] && extra=(--calibration-from "$OUT/shaded/renders/arcface_align.json")
    "$AAU_DIR/baselines/run_bl.sh" aau/zs3dmm/zs_arcface_render.py --view-dir "$VIEW" --subjects-json "$SUBJ" \
        --out-root "$OUT/$mode" --base-rotation none --mode "$mode" --workers "$CORES" "${extra[@]}"
done
echo "[be-render] OK"
