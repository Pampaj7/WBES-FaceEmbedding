#!/usr/bin/env bash
# Come aau/run.sh, ma nel venv con open3d (.venv_open3d, vedi setup_open3d.sbatch).
#
#   aau/outlineB/run_o3d.sh aau/outlineB/alignment_matrix.py --subject-set heldout
#
# Sul frontend NON funziona: singularity esiste solo sui nodi di calcolo.
# Un thread per processo: la parallelizzazione e' per coppia di mesh (un worker per core),
# e open3d/BLAS con tutti i thread del nodo dentro ogni worker si pesterebbero i piedi.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
export VENV="${WBES_O3D_VENV:-$WBES_ROOT/.venv_open3d}"
aau_require_venv
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$WBES_ROOT"
exec singularity exec ${AAU_NV} "$CONTAINER" bash -c '
source "$VENV/bin/activate"
export LD_LIBRARY_PATH="$VENV/apt/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec python3 "$@"
' _ "$@"
