#!/usr/bin/env bash
#SBATCH --job-name=wbes-ir-arcface
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#
# Riconoscimento in dominio: ArcFace su render di sola geometria (zs_arcface_render.py via
# aau/indomain/ir_arcface_render.py) sui tre insiemi, ombreggiato (serve alla calibrazione del
# crop) e poi normal map con lo stesso crop, come zs_arcface.sbatch. CPU e niente GPU.
#
#   WBES_IR_ARCFACE_FRAMECHECK=1 aau/submit.sh indomain/ir_arcface.sbatch --time=00:15:00   # prima
#   aau/submit.sh indomain/ir_arcface.sbatch
#   WBES_IR_ARCFACE_BEST=1 aau/submit.sh indomain/ir_arcface.sbatch
#     configurazione "migliore" (protocollo, revisione 1, punto C): inquadratura per mesh
#     (--camera mesh) e normali smussate per vertice (--normals vertex); out in arcface_best/.
#
# Render su /tmp del nodo (contano contro --mem), poi si copia indietro tutto tranne i png dei
# render: arcface_views.npz, camera e calibrazione, png di controllo. Disco: pochi MB per insieme.
# Rotazione verso il frame del renderer (alto -y, naso -z): BFM nessuna, ICT Rx(180), fissate col
# frame check (protocol.md).
# TEMPI: HIFI3D normal map, 600 mesh x 3 viste, 40 s a 32 core (job 1057319).
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

OUT="$AAU_RUNS/indomain_recog/arcface"
BEST=()
if [[ "${WBES_IR_ARCFACE_BEST:-0}" == 1 ]]; then
    OUT="$AAU_RUNS/indomain_recog/arcface_best"
    BEST=(--camera mesh)
fi
TMP_OUT="${TMPDIR:-/tmp}/wbes_ir_arcface_${SLURM_JOB_ID:-manual}"
trap 'rm -rf "$TMP_OUT"' EXIT
CORES="${SLURM_CPUS_PER_TASK:-8}"
declare -A ROT=([bfm]=none [ict]=x180 [rexpr]=x180)

# Un thread per processo: il parallelismo e' sui processi (vedi ws3a_perceptual.sbatch).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "[ir-arcface] host=$(hostname) job=${SLURM_JOB_ID:-none} out=$OUT tmp=$TMP_OUT"
for set in ${WBES_IR_ARCFACE_SETS:-bfm ict rexpr}; do
    SUBJ="$AAU_RUNS/indomain_recog/subjects/$set.json"
    VIEW="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["view_dir"])' "$SUBJ")"
    if [[ "${WBES_IR_ARCFACE_FRAMECHECK:-0}" == 1 ]]; then
        [[ "$set" == rexpr ]] && continue
        "$AAU_DIR/baselines/run_bl.sh" aau/indomain/ir_arcface_render.py "$set" --view-dir "$VIEW" \
            --subjects-json "$SUBJ" --out-root "$OUT/$set/shaded" --base-rotation none --frame-check 3
        continue
    fi
    if [[ -f "$OUT/$set/normals/arcface_views.npz" ]]; then
        echo "[ir-arcface] $set: gia' fatto, salto"
        continue
    fi
    for mode in shaded normals; do
        extra=()
        [[ "$mode" == normals ]] && extra=(--calibration-from "$TMP_OUT/$set/shaded/renders/arcface_align.json")
        [[ "$mode" == normals && ${#BEST[@]} -gt 0 ]] && extra+=(--normals vertex)
        "$AAU_DIR/baselines/run_bl.sh" aau/indomain/ir_arcface_render.py "$set" --view-dir "$VIEW" \
            --subjects-json "$SUBJ" --out-root "$TMP_OUT/$set/$mode" --base-rotation "${ROT[$set]}" \
            --mode "$mode" --workers "$CORES" "${BEST[@]}" "${extra[@]}"
    done
    mkdir -p "$OUT/$set"
    rsync -a --exclude='renders/*/' "$TMP_OUT/$set/" "$OUT/$set/"
    rm -rf "${TMP_OUT:?}/$set"
    echo "[ir-arcface] $set: copiato in $OUT/$set"
done
echo "[ir-arcface] OK"
