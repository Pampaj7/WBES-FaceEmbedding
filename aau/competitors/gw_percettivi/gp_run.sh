#!/usr/bin/env bash
# Come aau/run.sh, ma nel venv di gw_percettivi (gp_env.sh, gp_setup.sbatch).
#
#   aau/competitors/gw_percettivi/gp_run.sh aau/competitors/gw_percettivi/gp_gw.py pilot --workers 64
#
# Solo sui nodi di calcolo (singularity). Thread BLAS: GP_THREADS (default 1: la parallelizzazione e' per processo).
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/env.sh"
source "$AAU_DIR/competitors/gw_percettivi/gp_env.sh"
export VENV="$GP_VENV"
aau_require_venv
_t="${GP_THREADS:-1}"
export OMP_NUM_THREADS=$_t OPENBLAS_NUM_THREADS=$_t MKL_NUM_THREADS=$_t

cd "$WBES_ROOT"
exec singularity exec ${AAU_NV} "$CONTAINER" bash -c '
source "$VENV/bin/activate"
exec python3 "$@"
' _ "$@"
