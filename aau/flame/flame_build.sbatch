#!/usr/bin/env bash
#SBATCH --job-name=wbes-flame-build
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#
# WS-FLAME, dati: i passi di v2_work/genflame/build_flame_5000.sh portati sul cluster, in un
# solo job (500 identita', non 5000). Dove lo script storico gira qui, si usa lui.
# Coefficienti: WBES_FLAME_SAMPLING=ict (default, N(0,1) senza troncamento come ICT) o legacy
# (troncata a 2.5 sigma come build_flame_5000.sh); vedi flame_env.sh.
#
#   1. identita'     v2_work/genflame/generate_identities.py --frame flame (storico, teste
#                    intere, SOLO su /tmp del nodo), poi aau/flame/flame_crop.py (patch del
#                    volto dalla maschera ufficiale) -> $WBES_FLAME_IDENTITIES_DIR
#   2. topologie     aau/flame/make_flame_topologies.py (le regole ICT/genict, igl)
#                    -> $WBES_FLAME_TOPO_DIR (6 per identita', npz compressi float32)
#   3. GT            v2_work/genflame/build_flame_gt_matrix.py (storico) -> $WBES_FLAME_GT_DIR
#   4. vista         v2_work/genflame/make_train_ready.py (storico), id offset 1000, sulla
#                    sola geometria -> $WBES_FLAME_MESH_DIR (symlink) + $WBES_FLAME_DIST_NPZ
#
# SPAZIO: in home finiscono solo geometria compressa e GT. Il passo "operatori" dello script
# storico NON e' qui: gli operatori (non compressi) li calcola ogni job di eval sul suo /tmp
# (flame_stage_ops in flame_env.sh), come per HIFI3D.
#
# Cosa NON e' storico, e perche': il passo 2 storico (genflame/make_flame_topologies.py) vuole
# open3d, il modulo datasets.expand_remesh_topologies che non c'e' nel checkout e la
# corrispondenza BFM_to_FLAME_corr.npz (asset FLAME); gli operatori storici
# (precompute_operators_npz.py) sono senza normalizzazione d'area, mentre i due modelli
# valutati sono addestrati ad area unitaria.
#
#   aau/submit.sh flame/flame_build.sbatch      # modello e maschere in v2_work/genflame/official/
#   WBES_FLAME_MODEL=<generic_model.pkl> WBES_FLAME_MASKS=<FLAME_masks.pkl> \
#     aau/submit.sh flame/flame_build.sbatch
#
# Riprendibile: le topologie saltano i file gia' scritti, ma solo sotto un timbro uguale al
# manifest delle identita' (vedi sotto). Le identita' invece si riscrivono sempre (sono
# secondi). Gli output degli eval stanno sotto l'impronta del timbro (flame_fingerprint), quindi
# dati nuovi non riusano mai risultati vecchi.
#
# Senza GPU: la GT su 500 x ~2k vertici e' una frazione di secondo anche su CPU.
set -euo pipefail

if [[ -z "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}${GPU_DEVICE_ORDINAL:-}" ]]; then
    export AAU_NV=""
fi
source "${WBES_ROOT:-$PWD}/aau/env.sh"
source "$WBES_ROOT/aau/flame/flame_env.sh"
cd "$WBES_ROOT"

# `ls dir/*.npz` esplode con "Argument list too long" a decine di migliaia di file (job 1019283).
count_npz() { find "$1" -maxdepth 1 -name "$2" -type f -o -maxdepth 1 -name "$2" -type l | wc -l; }
aau_require_venv
aau_require_diffusion_net
flame_require_model

N_CORES="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Teste intere su /tmp: percorso stabile perche' finisce nel manifest delle identita', che
# fa da timbro per le topologie qui sotto (un percorso col job id le farebbe rifare sempre).
FULL_DIR="$WBES_FLAME_TMP/build/identities_full"
trap 'rm -rf "$WBES_FLAME_TMP/build"' EXIT

echo "[flame-build] host=$(hostname) job=${SLURM_JOB_ID:-none} cores=$N_CORES"
echo "[flame-build] modello=$WBES_FLAME_MODEL crop=$WBES_FLAME_CROP maschera=${WBES_FLAME_MASKS:-} regione=$WBES_FLAME_MASK_REGION"
echo "[flame-build] n=$WBES_FLAME_N_IDENTITIES seed=$WBES_FLAME_SEED n_shape=$WBES_FLAME_N_SHAPE sampling=$WBES_FLAME_SAMPLING (trunc=$WBES_FLAME_TRUNC)"
echo "[flame-build] root=$WBES_FLAME_ROOT"

echo "[flame-build] 1/4 identita' (generatore storico, su /tmp) + patch del volto"
rm -rf "$FULL_DIR"
rm -f "$WBES_FLAME_IDENTITIES_DIR"/flame[0-9]*.npz "$WBES_FLAME_IDENTITIES_DIR/manifest.json"
"$AAU_DIR/run.sh" v2_work/genflame/generate_identities.py \
    --n-identities "$WBES_FLAME_N_IDENTITIES" --seed "$WBES_FLAME_SEED" \
    --n-shape "$WBES_FLAME_N_SHAPE" --trunc "$WBES_FLAME_TRUNC" --frame flame --out-dir "$FULL_DIR"
"$AAU_DIR/run.sh" aau/flame/flame_crop.py \
    --in-dir "$FULL_DIR" --out-dir "$WBES_FLAME_IDENTITIES_DIR" \
    --crop "$WBES_FLAME_CROP" --mask-region "$WBES_FLAME_MASK_REGION"

# Le topologie sono riprendibili per nome, quindi valgono solo se il loro timbro e' il manifest
# delle identita' appena scritte. Timbro diverso O ASSENTE (topo/ di una build interrotta prima
# del timbro, o di una versione vecchia di questo script) => si butta tutto cio' che ne deriva.
# Il timbro nuovo si scrive PRIMA del passo 2: una build interrotta durante le topologie lascia
# file parziali col timbro giusto, e la build successiva o li riprende (stessi input) o li butta.
STAMP="$WBES_FLAME_TOPO_DIR/$FLAME_STAMP_NAME"
if ! cmp -s "$STAMP" "$WBES_FLAME_IDENTITIES_DIR/manifest.json"; then
    if [[ -d "$WBES_FLAME_TOPO_DIR" || -d "$WBES_FLAME_VIEW_DIR" || -d "$WBES_FLAME_GT_DIR" ]]; then
        echo "[flame-build] timbro assente o diverso dalle identita' nuove: butto topo, GT e vista"
    fi
    rm -rf "$WBES_FLAME_TOPO_DIR" "$WBES_FLAME_VIEW_DIR" "$WBES_FLAME_GT_DIR"
fi
mkdir -p "$WBES_FLAME_TOPO_DIR"
cp "$WBES_FLAME_IDENTITIES_DIR/manifest.json" "$STAMP"
# La vista diventa valida solo alla fine (passo 4): fino ad allora niente timbro li'.
rm -f "$WBES_FLAME_VIEW_DIR/$FLAME_STAMP_NAME"
echo "[flame-build] impronta dei dati: $(flame_fingerprint)"

echo "[flame-build] 2/4 topologie"
# Il controllo del crop per identita' (ripiego di mesh_ops.trim_far_boundary_band) e' dentro
# lo script: elenca le identita' col ripiego e fallisce oltre il 2%.
"$AAU_DIR/run.sh" aau/flame/make_flame_topologies.py \
    --in-dir "$WBES_FLAME_IDENTITIES_DIR" --out-dir "$WBES_FLAME_TOPO_DIR" --n-cores "$N_CORES"
N_ID=$(count_npz "$WBES_FLAME_IDENTITIES_DIR" "flame[0-9]*.npz")
N_TOPO=$(count_npz "$WBES_FLAME_TOPO_DIR" "flame*_GTready_*.npz")
echo "[flame-build] topologie: $N_TOPO / $((N_ID * 6))"
if (( N_TOPO != N_ID * 6 )); then
    echo "ERRORE: topologie incomplete" >&2
    exit 1
fi
"$AAU_DIR/run.sh" aau/flame/make_flame_topologies.py --demo \
    --in-dir "$WBES_FLAME_IDENTITIES_DIR" --out-dir "$WBES_FLAME_TOPO_DIR"

echo "[flame-build] 3/4 GT"
"$AAU_DIR/run.sh" v2_work/genflame/build_flame_gt_matrix.py \
    --topo-dir "$WBES_FLAME_TOPO_DIR" --out-dir "$WBES_FLAME_GT_DIR"

echo "[flame-build] 4/4 vista idNNNN (sola geometria) + GT rinominata"
"$AAU_DIR/run.sh" v2_work/genflame/make_train_ready.py \
    --withops-dir "$WBES_FLAME_TOPO_DIR" \
    --gt-npz "$WBES_FLAME_GT_DIR/flame_matrix_distances_maxabs.npz" \
    --out-dir "$WBES_FLAME_VIEW_DIR" --id-offset "$WBES_FLAME_ID_OFFSET"
N_VIEW=$(count_npz "$WBES_FLAME_MESH_DIR" "id*_GTready_*.npz")
echo "[flame-build] vista: $N_VIEW symlink in $WBES_FLAME_MESH_DIR, GT $WBES_FLAME_DIST_NPZ"
if (( N_VIEW != N_TOPO )); then
    echo "ERRORE: vista incompleta ($N_VIEW / $N_TOPO)" >&2
    exit 1
fi
cp "$STAMP" "$WBES_FLAME_VIEW_DIR/$FLAME_STAMP_NAME"
echo "[flame-build] impronta $(flame_fingerprint): output degli eval in $WBES_FLAME_RUNS/$(flame_fingerprint)/"
echo "[flame-build] in home: $(du -sh "$WBES_FLAME_ROOT" | cut -f1) in $WBES_FLAME_ROOT"
echo "[flame-build] OK"
