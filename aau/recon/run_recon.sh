#!/usr/bin/env bash
# Come aau/run.sh, ma nel venv di uno dei metodi di ricostruzione 3D (WS3b).
#
#   aau/recon/run_recon.sh ddfa  aau/recon/tddfa_v2_run.py --images ... --out ...
#   aau/recon/run_recon.sh prnet aau/recon/prnet_run.py    --images ... --out ...
#
# Il primo argomento e' il nome del venv sotto external/venvs/: `ddfa` per 3DDFA_V2 e
# SynergyNet (stesse dipendenze torch), `prnet` per PRNet (tensorflow, incompatibile con
# il resto).  Sul frontend NON funziona: singularity esiste solo sui nodi di calcolo.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

if [[ $# -lt 2 ]]; then
    echo "uso: aau/recon/run_recon.sh <ddfa|prnet> <script.py> [args...]" >&2
    exit 2
fi

export VENV="$WBES_ROOT/external/venvs/$1"
shift

if [[ ! -x "$VENV/bin/python3" ]]; then
    echo "ERRORE: venv assente in $VENV" >&2
    echo "  Crealo con: aau/submit.sh recon/setup_env.sbatch" >&2
    exit 1
fi

cd "$WBES_ROOT"
exec singularity exec ${AAU_NV} "$CONTAINER" bash -c '
source "$VENV/bin/activate"
exec python3 "$@"
' _ "$@"
