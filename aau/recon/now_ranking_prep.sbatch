#!/usr/bin/env bash
#SBATCH --job-name=wbes-now-rank-prep
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# NoW ranking (aau/runs/evidence/now_ranking/REGOLE.md), passi su CPU per i metodi NUOVI (WBES_RANKING_NEW, default
# 3ddfa_v3) e NoW metrical per TUTTI (WBES_RANKING_METHODS):
#   1. export nel layout ufficiale e NoW originale (compute_error.py) dei metodi nuovi;
#   2. export in mm e NoW metrical (metrical_eval=True) di tutti, col controllo scansione-su-se-stessa senza scala;
#   3. patch alla taglia del template dei nuovi (now_prepare_meshes.py --recon-only) e patch metriche
#      (--metric: scan_face_metric e <metodo>_metric) dei metodi con unita' note (WBES_RANKING_METRIC);
#   4. Chamfer grezza e ICP + Chamfer dei nuovi (now_geometric.py --mode gt).
# csv in aau/runs/evidence/now_ranking/tables (WBES_NOW_OUT); mesh e patch in ~/data/now_eval_work, fuori dal repo.
#
#   aau/submit.sh recon/now_ranking_prep.sbatch
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

export WBES_NOW_OUT="${WBES_NOW_OUT:-$AAU_RUNS/evidence/now_ranking/tables}"
NEW="${WBES_RANKING_NEW:-3ddfa_v3}"
ALL="${WBES_RANKING_METHODS:-3ddfa_v2,synergynet,prnet,3ddfa_v3}"
METRIC="${WBES_RANKING_METRIC:-3ddfa_v3}"
W="${SLURM_CPUS_PER_TASK:-8}"
export MPLBACKEND=Agg
mkdir -p "$WBES_NOW_OUT"
echo "[now-rank-prep] host=$(hostname) job=${SLURM_JOB_ID:-none} nuovi=$NEW tutti=$ALL metrici=$METRIC out=$WBES_NOW_OUT"

now_py() {  # venv del codice NoW ufficiale (psbody.mesh, chumpy), come now_official.sbatch
    VENV="$WBES_ROOT/external/venvs/now" singularity exec "$CONTAINER" bash -c \
        'source "$VENV/bin/activate"; exec python3 "$@"' _ "$@"
}

"$AAU_DIR/run.sh" aau/recon/now_export.py --methods "$NEW" --selfcheck-subjects 0
now_py aau/recon/now_official.py --methods "$NEW" --nproc "$W"

"$AAU_DIR/run.sh" aau/recon/now_export.py --metrical --methods "$ALL"
now_py aau/recon/now_official.py --metrical --selfcheck --nproc "$W"
now_py aau/recon/now_official.py --metrical --methods "$ALL" --nproc "$W"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$AAU_DIR/run.sh" aau/recon/now_prepare_meshes.py --recon-only --methods "$NEW" --workers "$W"
"$AAU_DIR/run.sh" aau/recon/now_prepare_meshes.py --metric --methods "$METRIC" --workers "$W"
"$AAU_DIR/run.sh" aau/recon/now_geometric.py --mode gt --methods "$NEW" --workers "$W"

echo "[now-rank-prep] OK"
