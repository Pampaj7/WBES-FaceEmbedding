#!/usr/bin/env bash
#SBATCH --job-name=wbes-e3b-eval
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#
# E3b ed E3 (diagnosi), embedding di e108 senza training (WBES_E3_DOMAIN):
#   hifi  a. rimesh uniforme al test (remesh.py, L dichiarata in aau/runs/evidence/e2/protocol.md) di ogni
#            mesh -> operatori -> embedding;
#         b. mesh come sono (zs_stage.py) -> operatori -> embed_variants.py: base (controllo), centro per
#            area, pooling pesato per area, entrambi.
#   fv    a. FaceVerse con espressioni, rimesh uniforme e poi facce invertite (convenzione BFM, quella della
#            riga e108 di riferimento) -> operatori -> embedding;
#         b. FaceVerse NEUTRO, convenzione BFM (zs_stage.py --flip-faces) -> operatori -> embedding: serve a
#            separare espressione e topologia nella dispersione intra-identita'.
# Uscite in aau/runs/evidence/e3/<dominio>/.
#
#   WBES_E3_DOMAIN=hifi aau/submit.sh evidence/e3_breakdown/e3b_eval.sbatch
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
source "$AAU_DIR/evidence/e2_canon/ops_embed.sh"
# Su CPU (sottomesso con --partition=cpu --gres=none): niente --nv a singularity.
[[ -n "${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}" ]] || export AAU_NV=
# Un thread BLAS per processo: gli stadi in Python sono pool da un processo per core (canon.py, remesh.py,
# chamfer_canon.py, now_geometric.py); senza, ogni processo apre un thread per core del nodo. L'embedding
# (ev_embed) e i latenti NoW rialzano i thread al numero di CPU del job.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
aau_require_venv
aau_require_diffusion_net

DOM="${WBES_E3_DOMAIN:?WBES_E3_DOMAIN=hifi|fv}"
[[ -f "$AAU_RUNS/evidence/e2/protocol.md" ]] || { echo "ERRORE: protocollo non dichiarato" >&2; exit 2; }
OUT="$AAU_RUNS/evidence/e3/$DOM"; mkdir -p "$OUT"
NW="${SLURM_CPUS_PER_TASK:-8}"
TMPD="/tmp/wbes_e3_${SLURM_JOB_ID:-manual}_$DOM"
rm -rf "$TMPD"; mkdir -p "$TMPD"
trap 'rm -rf "$TMPD"' EXIT
echo "[e3] dominio=$DOM host=$(hostname) job=${SLURM_JOB_ID:-none} ckpt=$E108_CKPT"

if [[ "$DOM" == hifi ]]; then
    VIEW="$WBES_ROOT/datasets/HIFI3D/eval_view"; FLIP=()
else
    VIEW="$WBES_ROOT/datasets/FACEVERSE_ZS/expr_view"; FLIP=(--flip-faces)
fi

echo "[e3] a. rimesh uniforme di $VIEW/npz"
AAU_NV= "$AAU_DIR/run.sh" aau/evidence/e3_breakdown/remesh.py stage --view-dir "$VIEW/npz" --seed 1234 \
    --out-dir "$TMPD/rm/in" --records "$OUT/remesh_records.csv" --workers "$NW" "${FLIP[@]}"
cp "$TMPD/rm/subjects.json" "$OUT/remesh_subjects.json"
ev_ops "$TMPD/rm/in" "$TMPD/rm/ops" "${DOM}_remesh"
ev_embed "$TMPD/rm/ops" "$VIEW/gt_matrix.npz" "$OUT/remesh_e108"
rm -rf "$TMPD/rm"

if [[ "$DOM" == hifi ]]; then
    echo "[e3] b. varianti di centro e pooling sulle mesh come sono"
    AAU_NV= "$AAU_DIR/run.sh" aau/zs3dmm/zs_stage.py --view-dir "$VIEW/npz" --out-dir "$TMPD/base/in" --seed 1234
    ev_ops "$TMPD/base/in" "$TMPD/base/ops" hifi_base
    ev_embed "$TMPD/base/ops" "$VIEW/gt_matrix.npz" "$OUT/variants_e108" \
        aau/evidence/e3_breakdown/embed_variants.py --variants base,areacenter,areapool,areaboth
else
    NEU="$WBES_ROOT/datasets/FACEVERSE_ZS/eval_view"
    echo "[e3] b. FaceVerse neutro, convenzione BFM"
    AAU_NV= "$AAU_DIR/run.sh" aau/zs3dmm/zs_stage.py --view-dir "$NEU/npz" --out-dir "$TMPD/neu/in" --seed 1234 --flip-faces
    cp "$TMPD/neu/subjects.json" "$OUT/neutral_subjects.json"
    ev_ops "$TMPD/neu/in" "$TMPD/neu/ops" fv_neutral
    ev_embed "$TMPD/neu/ops" "$NEU/gt_matrix.npz" "$OUT/neutral_e108_flip"
fi
echo "[e3] $DOM OK"
