#!/usr/bin/env bash
#SBATCH --job-name=wbes-eq-view
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#
# Vista a supporto equalizzato di un dominio (WBES_ZS_DOMAIN=bfm|hifi|ict): eqsupport_view.py sul
# pool intero della vista, in $ZS_EQ_DIR (= $ZS_ROOT/eqsupport_view). Solo geometria, niente GPU.
#
#   WBES_ZS_DOMAIN=hifi aau/submit.sh zs3dmm/eqsupport_view.sbatch
#
# Per bfm scrive anche $WBES_BFMZS_ROOT/manifest.json (sorgente e sha1 della GT), l'unico
# manifest dell'impronta dei dati di quel dominio (bfm_env.sh): serve anche alla base.
#
# Spazio: ~5/6 della vista (crop e' un symlink); BFM ~3 GB, HIFI3D/ICT ~0.5 GB.
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
source "$WBES_ROOT/aau/zs3dmm/zs_env.sh"
cd "$WBES_ROOT"
aau_require_venv
zs_require_view

if [[ "$WBES_ZS_DOMAIN" == bfm && ! -f "$WBES_BFMZS_ROOT/manifest.json" ]]; then
    mkdir -p "$WBES_BFMZS_ROOT"
    cat > "$WBES_BFMZS_ROOT/manifest.json" <<EOF
{"domain": "bfm", "view_dir": "$(realpath "$WBES_BFMZS_SOURCE_DIR")",
 "gt": "$(realpath "$WBES_BFMZS_DIST_NPZ")", "gt_sha1": "$(sha1sum "$WBES_BFMZS_DIST_NPZ" | cut -c1-40)",
 "subjects": "rebuild_subject_split(pool di 500, eval_fraction 0.2, seed 1234) = common.heldout_subjects"}
EOF
    echo "[eq-view] scritto $WBES_BFMZS_ROOT/manifest.json"
fi

echo "[eq-view] dominio=$WBES_ZS_DOMAIN host=$(hostname) job=${SLURM_JOB_ID:-none} vista=$ZS_DATA_DIR -> $ZS_EQ_DIR"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
"$AAU_DIR/run.sh" aau/zs3dmm/eqsupport_view.py --view-dir "$ZS_DATA_DIR" --out-dir "$ZS_EQ_DIR" \
    --n-cores "${SLURM_CPUS_PER_TASK:-8}" ${WBES_EQ_VIEW_ARGS:-}
du -sh "$ZS_EQ_DIR"
