#!/usr/bin/env bash
#SBATCH --job-name=wbes-zs-build-expr
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#
# Modalita' espressioni (WBES_ZS_EXPR=1, fv e gnm), dati: la vista con un'espressione casuale per
# mesh accanto a quella neutra di zs_build.sbatch, che deve esistere gia' (identita' e GT).
#
#   WBES_ZS_DOMAIN=fv  WBES_ZS_EXPR=1 aau/submit.sh zs3dmm/zs_build_expr.sbatch
#   WBES_ZS_DOMAIN=gnm WBES_ZS_EXPR=1 aau/submit.sh zs3dmm/zs_build_expr.sbatch
#
#   1. topologie con espressione  make_zs_expr_topologies.py -> expr_topo/ (6 per identita')
#   2. vista                      make_zs_view.py            -> expr_view/ (symlink + GT NEUTRE)
#
# La GT non si ricalcola: make_zs_view.py copia quelle di gt/ (forme neutre), e qui si verifica
# che siano identiche a quelle della vista neutra. Il manifest delle espressioni entra in quello
# della vista, quindi nell'impronta dei risultati (zs_data_fp).
#
# TEMPI: la build neutra di FaceVerse fa le 6 topologie di 500 identita' in ~12 min a 32 core
# (job 1055780); qui lo stesso lavoro, piu' 6 patch con espressione per identita'.
set -euo pipefail

export AAU_NV=""
export WBES_ZS_EXPR=1
source "${WBES_ROOT:-$PWD}/aau/env.sh"
source "$WBES_ROOT/aau/zs3dmm/zs_env.sh"
cd "$WBES_ROOT"

count_npz() { find "$1" -maxdepth 1 -name "$2" -type f -o -maxdepth 1 -name "$2" -type l | wc -l; }
aau_require_venv
zs_require_model
for f in "$ZS_IDENTITIES_DIR/manifest.json" "$ZS_IDENTITIES_DIR/identity_weights.json" "$ZS_GT_DIR/manifest.json" \
         "$ZS_NEUTRAL_VIEW_DIR/gt_matrix.npz"; do
    if [[ ! -f "$f" ]]; then
        echo "ERRORE: dati neutri assenti ($f): lancia prima zs_build.sbatch senza WBES_ZS_EXPR" >&2
        exit 1
    fi
done

N_CORES="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[zs-build-expr] dominio=$WBES_ZS_DOMAIN ($ZS_LABEL) host=$(hostname) job=${SLURM_JOB_ID:-none} cores=$N_CORES"
echo "[zs-build-expr] identita'=$ZS_IDENTITIES_DIR -> $ZS_EXPR_TOPO_DIR -> $ZS_VIEW_DIR"

# Riprendibile per nome, come le topologie neutre: se le identita' sono cambiate si butta tutto.
STAMP="$ZS_EXPR_TOPO_DIR/.identities_manifest.json"
if [[ -f "$STAMP" ]] && ! cmp -s "$STAMP" "$ZS_IDENTITIES_DIR/manifest.json"; then
    echo "[zs-build-expr] identita' cambiate: rifaccio topologie e vista"
    rm -rf "$ZS_EXPR_TOPO_DIR" "$ZS_VIEW_DIR"
fi

# GNM: nel manifest anche lo spostamento relativo al soggetto piu' vicino (--gt-dir), la misura su
# cui e' tarata la sua ampiezza. Non per fv: il suo manifest cambierebbe rispetto a quello della
# build esistente, e con lui l'impronta dei risultati (zs_data_fp).
GT_ARGS=()
[[ "$WBES_ZS_DOMAIN" == gnm ]] && GT_ARGS=(--gt-dir "$ZS_GT_DIR")
echo "[zs-build-expr] 1/2 topologie con espressione"
"$AAU_DIR/run.sh" aau/zs3dmm/make_zs_expr_topologies.py --domain "$WBES_ZS_DOMAIN" --prefix "$ZS_PREFIX" \
    --model-file "$ZS_MODEL_FILE" --identities-dir "$ZS_IDENTITIES_DIR" \
    --out-dir "$ZS_EXPR_TOPO_DIR" --n-cores "$N_CORES" "${GT_ARGS[@]}"
cp "$ZS_IDENTITIES_DIR/manifest.json" "$STAMP"
N_ID=$(count_npz "$ZS_IDENTITIES_DIR" "$ZS_PREFIX[0-9]*.npz")
N_TOPO=$(count_npz "$ZS_EXPR_TOPO_DIR" "$ZS_PREFIX*_GTready_*.npz")
echo "[zs-build-expr] topologie: $N_TOPO / $((N_ID * 6))"
if (( N_TOPO != N_ID * 6 )); then
    echo "ERRORE: topologie incomplete" >&2
    exit 1
fi
"$AAU_DIR/run.sh" aau/zs3dmm/make_zs_topologies.py --demo --prefix "$ZS_PREFIX" \
    --in-dir "$ZS_IDENTITIES_DIR" --out-dir "$ZS_EXPR_TOPO_DIR"

echo "[zs-build-expr] 2/2 vista id + GT neutre"
"$AAU_DIR/run.sh" aau/zs3dmm/make_zs_view.py --prefix "$ZS_PREFIX" \
    --topo-dir "$ZS_EXPR_TOPO_DIR" --gt-dir "$ZS_GT_DIR" \
    --out-dir "$ZS_VIEW_DIR" --id-offset "$ZS_ID_OFFSET" \
    --extra-manifest "$ZS_EXPR_TOPO_DIR/manifest.json"
for g in gt_matrix.npz gt_coef_matrix.npz; do
    if ! cmp -s "$ZS_VIEW_DIR/$g" "$ZS_NEUTRAL_VIEW_DIR/$g"; then
        echo "ERRORE: $ZS_VIEW_DIR/$g diversa da quella della vista neutra" >&2
        exit 1
    fi
done
N_VIEW=$(count_npz "$ZS_DATA_DIR" "id*_GTready_*.npz")
echo "[zs-build-expr] vista: $N_VIEW symlink in $ZS_DATA_DIR, GT (neutre) $ZS_DIST_NPZ + $ZS_COEF_NPZ"
if (( N_VIEW != N_TOPO )); then
    echo "ERRORE: vista incompleta ($N_VIEW / $N_TOPO)" >&2
    exit 1
fi
echo "[zs-build-expr] spazio nella home:"; du -sh "$ZS_EXPR_TOPO_DIR" "$ZS_VIEW_DIR"
echo "[zs-build-expr] OK"
