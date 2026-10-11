#!/usr/bin/env bash
# Come aau/outlineB/run_o3d.sh, ma nel venv con pymeshlab (.venv_d4, d4.sbatch passo setup), per il remesher di D4.
#
#   aau/diagnostics/d4_run.sh aau/diagnostics/d4_remesh.py ...
#
# Il wheel di pymeshlab e i suoi plugin linkano libGL, libOpenGL, libX11, libxcb, assenti nel container NGC: i .deb di
# jammy estratti in $VENV/apt/root dal passo setup vanno in LD_LIBRARY_PATH. Nessuna apre un display.
# Un thread per processo: la parallelizzazione e' per mesh (un worker per core).
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
export VENV="$WBES_ROOT/.venv_d4"
aau_require_venv
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$WBES_ROOT"
exec singularity exec ${AAU_NV} "$CONTAINER" bash -c '
source "$VENV/bin/activate"
export LD_LIBRARY_PATH="$VENV/apt/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec python3 "$@"
' _ "$@"
