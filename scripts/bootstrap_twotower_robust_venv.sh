#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${1:-$ROOT/.venv_twotower_robust}"

export WBES_DIFFUSION_NET_SRC="${WBES_DIFFUSION_NET_SRC:-/equilibrium/lpampaloni/diffusion-net/src}"

python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

pip install --upgrade pip setuptools wheel
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy scikit-learn tqdm tensorboard potpourri3d robust-laplacian libigl

python "$ROOT/scripts/check_twotower_robust_env.py"

cat <<EOF

Bootstrap completato.

Per usare l'env:
  source "$VENV_PATH/bin/activate"
  export WBES_DIFFUSION_NET_SRC="$WBES_DIFFUSION_NET_SRC"

EOF
