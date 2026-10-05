#!/usr/bin/env bash
# Come aau/run.sh, ma nel venv delle baseline (.venv_baselines).
#
#   aau/baselines/run_bl.sh aau/baselines/perceptual_matrix.py --device cuda
#
# Sul frontend NON funziona: singularity esiste solo sui nodi di calcolo.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
export VENV="${WBES_BL_VENV:-$WBES_ROOT/.venv_baselines}"
aau_require_venv
aau_require_diffusion_net

cd "$WBES_ROOT"
exec singularity exec ${AAU_NV} "$CONTAINER" bash -c '
source "$VENV/bin/activate"
exec python3 "$@"
' _ "$@"
