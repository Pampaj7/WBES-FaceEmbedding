#!/usr/bin/env bash
#SBATCH --job-name=wbes-zs-expr-extra
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# Modalita' espressioni (WBES_ZS_EXPR=1), le baseline che servono al riconoscimento d'identita' e
# che zs_baselines.sbatch non scrive. Sullo stesso set di 100 soggetti, nella stessa
# $ZS_RUNS/data_<fp>/baselines:
#   same    zs_bl_same.py         faceBench (Chamfer, ICP rigido, NICP) sulle coppie stesso
#                                 soggetto / topologie diverse -> matrices_same/
#   region  zs_region_chamfer.py  Chamfer su regione stabile all'espressione e mesh intera, tutte
#                                 le coppie delle 600 mesh -> region_chamfer.npz
#
#   WBES_ZS_DOMAIN=fv WBES_ZS_EXPR=1 aau/submit.sh zs3dmm/zs_expr_extra.sbatch
#   WBES_ZS_EXTRA_STEPS=region ...     un passo solo
#
# TEMPI: same = 30 x 100 coppie faceBench, ~2.7 s per coppia a un core (job 1054623) -> ~10 min a
# 32 worker; region = 2 x 179.700 coppie di nuvole da 4096 punti.
set -euo pipefail

export AAU_NV=""
export WBES_ZS_EXPR="${WBES_ZS_EXPR:-1}"
source "${WBES_ROOT:-$PWD}/aau/env.sh"
source "$WBES_ROOT/aau/zs3dmm/zs_env.sh"
cd "$WBES_ROOT"
zs_require_view

# Assegnazioni SENZA default (vedi hifi_env.sh): common.py legge queste due all'import.
export WBES_REMESH_NOOPS_DIR="$ZS_DATA_DIR"
export WBES_DIST_NPZ="$ZS_DIST_NPZ"
export WBES_EVAL_SEED="${WBES_EVAL_SEED:-1234}"
ZS_FP="$(zs_data_fp)"
OUT="$ZS_RUNS/data_$ZS_FP/baselines"
mkdir -p "$OUT"
STEPS="${WBES_ZS_EXTRA_STEPS:-same region}"
echo "[zs-expr-extra] dominio=$WBES_ZS_DOMAIN host=$(hostname) job=${SLURM_JOB_ID:-none} passi=$STEPS out=$OUT"

for step in $STEPS; do
    case "$step" in
        same)
            "$AAU_DIR/outlineB/run_o3d.sh" aau/zs3dmm/zs_bl_same.py "$WBES_ZS_DOMAIN" "$ZS_ID_OFFSET" \
                --out-root "$OUT" --workers "${SLURM_CPUS_PER_TASK:-8}"
            ;;
        region)
            "$AAU_DIR/run.sh" aau/zs3dmm/zs_region_chamfer.py --domain "$WBES_ZS_DOMAIN" \
                --view-dir "$ZS_DATA_DIR" --out "$OUT/region_chamfer.npz" --seed "$WBES_EVAL_SEED" \
                --workers "${SLURM_CPUS_PER_TASK:-8}"
            ;;
        *) echo "ERRORE: passo sconosciuto: $step (same|region)" >&2; exit 2 ;;
    esac
done
echo "[zs-expr-extra] OK"
