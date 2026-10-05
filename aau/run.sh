#!/usr/bin/env bash
# Esegue python3 dentro il container con il venv aau attivo.
# Usabile in sbatch, sotto srun, o direttamente su un nodo di calcolo:
#
#   aau/run.sh scripts/check_twotower_robust_env.py
#   aau/run.sh -c "import diffusion_net; print('ok')"
#
# Sul frontend NON funziona: singularity esiste solo sui nodi di calcolo.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
aau_require_venv
aau_require_diffusion_net

cd "$WBES_ROOT"
exec singularity exec ${AAU_NV} "$CONTAINER" bash -c '
source "$VENV/bin/activate"
exec python3 "$@"
' _ "$@"
