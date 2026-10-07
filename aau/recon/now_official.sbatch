#!/usr/bin/env bash
#SBATCH --job-name=wbes-now-official
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# NoW, errore ufficiale: export nel layout NoW (venv aau, ha igl per leggere le scansioni;
# gli export gia' presenti si saltano),
# controllo scansione-su-se-stessa, poi compute_error.py per metodo (venv now, vedi
# now_setup_env.sbatch).  Solo CPU: nessuna GPU richiesta.
#
#   aau/submit.sh recon/now_official.sbatch [--methods 3ddfa_v2]
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

echo "[now-official] host=$(hostname) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?}"

"$AAU_DIR/run.sh" aau/recon/now_export.py "$@"

export VENV="$WBES_ROOT/external/venvs/now"
export MPLBACKEND=Agg
NOW_PY=(singularity exec "$CONTAINER" bash -c 'source "$VENV/bin/activate"; exec python3 "$@"' _)
"${NOW_PY[@]}" aau/recon/now_official.py --selfcheck || echo "[now-official] ATTENZIONE: selfcheck fallito, vedi .err"
"${NOW_PY[@]}" aau/recon/now_official.py "$@"

echo "[now-official] OK"
