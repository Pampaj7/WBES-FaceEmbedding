#!/usr/bin/env bash
#SBATCH --job-name=wbes-devfs-build
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#
# Set di sviluppo FaceScape bilineare (D1), dati: zs_build.sbatch + zs_build_expr.sbatch per un
# dominio che quei due non conoscono, con i loro stessi passi e gli stessi script dove possibile.
#
#   aau/submit.sh zs3dmm/dev_fs_build.sbatch
#
#   1. identita'      aau/zs3dmm/dev_fs_identities.py      -> identities/ (500, seed 1234, N(0,1))
#   2. topologie      aau/zs3dmm/make_zs_topologies.py     -> topo/ (6 per identita')   [esistente]
#      controllo crop aau/zs3dmm/zs_check_crop.py                                       [esistente]
#   3. GT             aau/zs3dmm/build_zs_gt.py            -> gt/ (maxabs + coefficienti) [esistente]
#   4. vista          aau/zs3dmm/make_zs_view.py           -> eval_view/ (id930000+)    [esistente]
#   5. espressioni    aau/zs3dmm/dev_fs_expr_topologies.py -> expr_topo/ (una per mesh)
#   6. vista espr.    aau/zs3dmm/make_zs_view.py           -> expr_view/ (GT NEUTRE)     [esistente]
#
# Percorsi da dev_facescape_env.sh (slot "fv" ridiretto su datasets/DEV_FACESCAPE), prefisso dei
# file fs. Riprendibile come zs_build: identita' riscritte sempre, topologie saltate se gia' scritte
# sotto lo stesso manifest delle identita'.
#
# Il core FaceScape occupa ~5 GB al caricamento (2.4 GB dopo la restrizione alla patch), una volta
# nel processo principale dei passi 1 e 5; i worker ricevono solo mesh o matrici da 8 MB.
set -euo pipefail

if [[ -z "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}${GPU_DEVICE_ORDINAL:-}" ]]; then
    export AAU_NV=""
fi
source "${WBES_ROOT:-$PWD}/aau/zs3dmm/dev_facescape_env.sh"
source "$WBES_ROOT/aau/zs3dmm/zs_env.sh"
cd "$WBES_ROOT"

count_npz() { find "$1" -maxdepth 1 -name "$2" -type f -o -maxdepth 1 -name "$2" -type l | wc -l; }
aau_require_venv
zs_require_model

N_CORES="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
P="$DEV_FS_PREFIX"
EXPR_TOPO="$ZS_ROOT/expr_topo"
EXPR_VIEW="$ZS_ROOT/expr_view"

echo "[devfs-build] host=$(hostname) job=${SLURM_JOB_ID:-none} cores=$N_CORES modello=$ZS_MODEL_FILE"
echo "[devfs-build] n=$ZS_N_IDENTITIES seed=$ZS_SEED offset=$ZS_ID_OFFSET root=$ZS_ROOT"

echo "[devfs-build] 1/6 identita'"
rm -f "$ZS_IDENTITIES_DIR/$P"[0-9]*.npz
OMP_NUM_THREADS="$N_CORES" MKL_NUM_THREADS="$N_CORES" OPENBLAS_NUM_THREADS="$N_CORES" \
    "$AAU_DIR/run.sh" aau/zs3dmm/dev_fs_identities.py \
    --n-identities "$ZS_N_IDENTITIES" --seed "$ZS_SEED" --out-dir "$ZS_IDENTITIES_DIR" --device cpu

STAMP="$ZS_TOPO_DIR/.identities_manifest.json"
if [[ -f "$STAMP" ]] && ! cmp -s "$STAMP" "$ZS_IDENTITIES_DIR/manifest.json"; then
    echo "[devfs-build] identita' cambiate rispetto alle topologie esistenti: rifaccio topo, viste ed espressioni"
    rm -rf "$ZS_TOPO_DIR" "$ZS_VIEW_DIR" "$EXPR_TOPO" "$EXPR_VIEW"
fi

echo "[devfs-build] 2/6 topologie"
"$AAU_DIR/run.sh" aau/zs3dmm/make_zs_topologies.py --prefix "$P" \
    --in-dir "$ZS_IDENTITIES_DIR" --out-dir "$ZS_TOPO_DIR" --n-cores "$N_CORES"
cp "$ZS_IDENTITIES_DIR/manifest.json" "$STAMP"
N_ID=$(count_npz "$ZS_IDENTITIES_DIR" "$P[0-9]*.npz")
N_TOPO=$(count_npz "$ZS_TOPO_DIR" "$P*_GTready_*.npz")
echo "[devfs-build] topologie: $N_TOPO / $((N_ID * 6))"
(( N_TOPO == N_ID * 6 )) || { echo "ERRORE: topologie incomplete" >&2; exit 1; }
"$AAU_DIR/run.sh" aau/zs3dmm/make_zs_topologies.py --demo --prefix "$P" \
    --in-dir "$ZS_IDENTITIES_DIR" --out-dir "$ZS_TOPO_DIR"
"$AAU_DIR/run.sh" aau/zs3dmm/zs_check_crop.py --topo-dir "$ZS_TOPO_DIR" --prefix "$P" --write

echo "[devfs-build] 3/6 GT"
OMP_NUM_THREADS="$N_CORES" "$AAU_DIR/run.sh" aau/zs3dmm/build_zs_gt.py --prefix "$P" \
    --topo-dir "$ZS_TOPO_DIR" --identities-dir "$ZS_IDENTITIES_DIR" --out-dir "$ZS_GT_DIR" --device cpu

echo "[devfs-build] 4/6 vista id + GT rinominate"
"$AAU_DIR/run.sh" aau/zs3dmm/make_zs_view.py --prefix "$P" \
    --topo-dir "$ZS_TOPO_DIR" --gt-dir "$ZS_GT_DIR" --out-dir "$ZS_VIEW_DIR" --id-offset "$ZS_ID_OFFSET"
N_VIEW=$(count_npz "$ZS_DATA_DIR" "id*_GTready_*.npz")
(( N_VIEW == N_TOPO )) || { echo "ERRORE: vista incompleta ($N_VIEW / $N_TOPO)" >&2; exit 1; }

echo "[devfs-build] 5/6 topologie con espressione"
STAMP_E="$EXPR_TOPO/.identities_manifest.json"
if [[ -f "$STAMP_E" ]] && ! cmp -s "$STAMP_E" "$ZS_IDENTITIES_DIR/manifest.json"; then
    rm -rf "$EXPR_TOPO" "$EXPR_VIEW"
fi
MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 "$AAU_DIR/run.sh" aau/zs3dmm/dev_fs_expr_topologies.py \
    --identities-dir "$ZS_IDENTITIES_DIR" --out-dir "$EXPR_TOPO" --gt-dir "$ZS_GT_DIR" \
    --seed "$ZS_SEED" --n-cores "$N_CORES"
cp "$ZS_IDENTITIES_DIR/manifest.json" "$STAMP_E"
N_ETOPO=$(count_npz "$EXPR_TOPO" "$P*_GTready_*.npz")
(( N_ETOPO == N_ID * 6 )) || { echo "ERRORE: topologie con espressione incomplete ($N_ETOPO)" >&2; exit 1; }
"$AAU_DIR/run.sh" aau/zs3dmm/make_zs_topologies.py --demo --prefix "$P" \
    --in-dir "$ZS_IDENTITIES_DIR" --out-dir "$EXPR_TOPO"

echo "[devfs-build] 6/6 vista con espressioni + GT neutre"
"$AAU_DIR/run.sh" aau/zs3dmm/make_zs_view.py --prefix "$P" \
    --topo-dir "$EXPR_TOPO" --gt-dir "$ZS_GT_DIR" --out-dir "$EXPR_VIEW" --id-offset "$ZS_ID_OFFSET" \
    --extra-manifest "$EXPR_TOPO/manifest.json"
for g in gt_matrix.npz gt_coef_matrix.npz; do
    cmp -s "$EXPR_VIEW/$g" "$ZS_VIEW_DIR/$g" || { echo "ERRORE: $EXPR_VIEW/$g diversa dalla vista neutra" >&2; exit 1; }
done
N_EVIEW=$(count_npz "$EXPR_VIEW/npz" "id*_GTready_*.npz")
(( N_EVIEW == N_ETOPO )) || { echo "ERRORE: vista con espressioni incompleta ($N_EVIEW)" >&2; exit 1; }
echo "[devfs-build] spazio:"; du -sh "$ZS_IDENTITIES_DIR" "$ZS_TOPO_DIR" "$ZS_GT_DIR" "$ZS_VIEW_DIR" "$EXPR_TOPO" "$EXPR_VIEW"
echo "[devfs-build] OK"
