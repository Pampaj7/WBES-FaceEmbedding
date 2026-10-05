#!/usr/bin/env bash
#SBATCH --job-name=wbes-zs-build
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#
# Zero-shot su un 3DMM mai visto, dati: i passi 1-3 e 5 della catena ICT (ict_identities,
# ict_topologies, ict_gt_matrix, ict_train_ready) in un solo job, per il dominio WBES_ZS_DOMAIN.
#
#   WBES_ZS_DOMAIN=hifi aau/submit.sh zs3dmm/zs_build.sbatch
#   WBES_ZS_DOMAIN=fv   aau/submit.sh zs3dmm/zs_build.sbatch
#
#   1. identita'     aau/zs3dmm/zs_identities.py      -> identities/
#   2. topologie     aau/zs3dmm/make_zs_topologies.py -> topo/ (6 per identita', solo V/F)
#   3. GT            aau/zs3dmm/build_zs_gt.py        -> gt/ (maxabs + coefficienti)
#   4. vista         aau/zs3dmm/make_zs_view.py       -> eval_view/ (symlink a topo/ + GT)
#
# Il passo degli operatori (ict_ops) NON c'e': nella home sta solo la geometria (vincolo di
# spazio: <= 15 GB nuovi in tutto). Gli operatori li calcola zs_zeroshot.sbatch su /tmp.
#
# Riprendibile: le topologie saltano i file gia' scritti. Le identita' invece si riscrivono
# sempre (sono secondi), cosi' un cambio di modello o di seed non lascia mesh vecchie sotto un
# manifest nuovo.
#
# Partizione prioritized senza GPU: la cpu ha UN nodo da 32 core, condiviso col lavoro FLAME.
set -euo pipefail

if [[ -z "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}${GPU_DEVICE_ORDINAL:-}" ]]; then
    export AAU_NV=""
fi
source "${WBES_ROOT:-$PWD}/aau/env.sh"
source "$WBES_ROOT/aau/zs3dmm/zs_env.sh"
cd "$WBES_ROOT"

# `ls dir/*.npz` esplode con "Argument list too long" a decine di migliaia di file (job 1019283).
count_npz() { find "$1" -maxdepth 1 -name "$2" -type f -o -maxdepth 1 -name "$2" -type l | wc -l; }
aau_require_venv
zs_require_model

N_CORES="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[zs-build] dominio=$WBES_ZS_DOMAIN ($ZS_LABEL) host=$(hostname) job=${SLURM_JOB_ID:-none} cores=$N_CORES"
echo "[zs-build] modello=$ZS_MODEL_FILE"
echo "[zs-build] n=$ZS_N_IDENTITIES seed=$ZS_SEED n_shape=$ZS_N_SHAPE trunc=$ZS_TRUNC offset=$ZS_ID_OFFSET"
echo "[zs-build] root=$ZS_ROOT"

# I passi con una matrice tutte-le-coppie (controllo Z1mX e GT) girano su torch CPU: a loro
# servono tutti i core, a differenza dei worker a un thread delle topologie.
echo "[zs-build] 1/4 identita'"
rm -f "$ZS_IDENTITIES_DIR/$ZS_PREFIX"[0-9]*.npz
if [[ "$WBES_ZS_DOMAIN" == ict ]]; then
    # Controllo della pipeline: le identita' held-out di ICT-5000, non un campionamento nuovo.
    "$AAU_DIR/run.sh" aau/zs3dmm/zs_link_identities.py --source-dir "$WBES_ICTZS_SOURCE_DIR" \
        --first "$WBES_ICTZS_FIRST" --n "$ZS_N_IDENTITIES" --out-dir "$ZS_IDENTITIES_DIR"
else
    OMP_NUM_THREADS="$N_CORES" "$AAU_DIR/run.sh" aau/zs3dmm/zs_identities.py \
        --domain "$WBES_ZS_DOMAIN" --model-file "$ZS_MODEL_FILE" \
        --n-identities "$ZS_N_IDENTITIES" --seed "$ZS_SEED" \
        --n-shape "$ZS_N_SHAPE" --trunc "$ZS_TRUNC" \
        --out-dir "$ZS_IDENTITIES_DIR" --device cpu
fi

# Le topologie sono riprendibili per nome: se le identita' sono cambiate (altro modello, altro
# seed, altri modi) le vecchie vanno buttate, altrimenti verrebbero "saltate".
STAMP="$ZS_TOPO_DIR/.identities_manifest.json"
if [[ -f "$STAMP" ]] && ! cmp -s "$STAMP" "$ZS_IDENTITIES_DIR/manifest.json"; then
    echo "[zs-build] identita' cambiate rispetto alle topologie esistenti: rifaccio topo e vista"
    rm -rf "$ZS_TOPO_DIR" "$ZS_VIEW_DIR"
fi

echo "[zs-build] 2/4 topologie"
"$AAU_DIR/run.sh" aau/zs3dmm/make_zs_topologies.py --prefix "$ZS_PREFIX" \
    --in-dir "$ZS_IDENTITIES_DIR" --out-dir "$ZS_TOPO_DIR" --n-cores "$N_CORES"
cp "$ZS_IDENTITIES_DIR/manifest.json" "$STAMP"
N_ID=$(count_npz "$ZS_IDENTITIES_DIR" "$ZS_PREFIX[0-9]*.npz")
N_TOPO=$(count_npz "$ZS_TOPO_DIR" "$ZS_PREFIX*_GTready_*.npz")
echo "[zs-build] topologie: $N_TOPO / $((N_ID * 6))"
if (( N_TOPO != N_ID * 6 )); then
    echo "ERRORE: topologie incomplete" >&2
    exit 1
fi
"$AAU_DIR/run.sh" aau/zs3dmm/make_zs_topologies.py --demo --prefix "$ZS_PREFIX" \
    --in-dir "$ZS_IDENTITIES_DIR" --out-dir "$ZS_TOPO_DIR"

# Il crop di mesh_ops torna la mesh intatta se toglierebbe piu' del 15% dei vertici: qui si
# esige che sia diverso dalla original per OGNI identita' (zs_check_crop.py).
"$AAU_DIR/run.sh" aau/zs3dmm/zs_check_crop.py --topo-dir "$ZS_TOPO_DIR" --prefix "$ZS_PREFIX" --write

echo "[zs-build] 3/4 GT"
OMP_NUM_THREADS="$N_CORES" "$AAU_DIR/run.sh" aau/zs3dmm/build_zs_gt.py --prefix "$ZS_PREFIX" \
    --topo-dir "$ZS_TOPO_DIR" --identities-dir "$ZS_IDENTITIES_DIR" \
    --out-dir "$ZS_GT_DIR" --device cpu

echo "[zs-build] 4/4 vista id + GT rinominate"
"$AAU_DIR/run.sh" aau/zs3dmm/make_zs_view.py --prefix "$ZS_PREFIX" \
    --topo-dir "$ZS_TOPO_DIR" --gt-dir "$ZS_GT_DIR" \
    --out-dir "$ZS_VIEW_DIR" --id-offset "$ZS_ID_OFFSET"
N_VIEW=$(count_npz "$ZS_DATA_DIR" "id*_GTready_*.npz")
echo "[zs-build] vista: $N_VIEW symlink in $ZS_DATA_DIR, GT $ZS_DIST_NPZ + $ZS_COEF_NPZ"
if (( N_VIEW != N_TOPO )); then
    echo "ERRORE: vista incompleta ($N_VIEW / $N_TOPO)" >&2
    exit 1
fi
echo "[zs-build] spazio nella home:"; du -sh "$ZS_IDENTITIES_DIR" "$ZS_TOPO_DIR" "$ZS_GT_DIR" "$ZS_VIEW_DIR"
echo "[zs-build] OK"
