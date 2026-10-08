#!/usr/bin/env bash
# Come aau/outlineB/run_o3d.sh, ma nel venv della GT unificata (.venv_ugt, setup_env.sbatch):
# open3d + mediapipe + h5py.
#
#   v3_work/unified_gt/run.sh v3_work/unified_gt/flame_region.py
#
# Solo sui nodi di calcolo (singularity). Thread BLAS: UGT_THREADS (default 1, come run_o3d.sh:
# la parallelizzazione, dove c'e', e' per processo).
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/aau/env.sh"
export VENV="${WBES_UGT_VENV:-$WBES_ROOT/.venv_ugt}"
aau_require_venv
_t="${UGT_THREADS:-1}"
export OMP_NUM_THREADS=$_t OPENBLAS_NUM_THREADS=$_t MKL_NUM_THREADS=$_t
export O3D_APT="$WBES_ROOT/.venv_open3d/apt/root/usr/lib/x86_64-linux-gnu"

cd "$WBES_ROOT"
exec singularity exec ${AAU_NV} "$CONTAINER" bash -c '
source "$VENV/bin/activate"
export LD_LIBRARY_PATH="$O3D_APT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec python3 "$@"
' _ "$@"
