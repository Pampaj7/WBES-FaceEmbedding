#!/usr/bin/env bash
#SBATCH --job-name=wbes-now-setup
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#
# Venv per la valutazione NoW UFFICIALE (external/now_evaluation, compute_error.py):
#   external/venvs/now   psbody-mesh + chumpy + sbody/smpl_webuser di flame-fitting
#
#   aau/submit.sh recon/now_setup_env.sbatch
#
# E' la ricetta del Dockerfile del repo ufficiale, adattata al container NGC:
#   - MPI-IS/mesh al commit 49e7042, compilato con i SOLI header di boost 1.83
#     (external/now_deps/boost_1_83_0, scaricati sul frontend: il container non li ha);
#   - flame-fitting al commit ca806ce: sbody/ e smpl_webuser/ copiati dentro il clone di
#     now_evaluation, e l'estensione sample2meshdist compilata con Eigen 3.4.0;
#   - numpy 1.23.5 nel venv: chumpy 0.70 fa `from numpy import bool, int, ...`, che
#     numpy >= 1.24 ha tolto.  scipy e matplotlib vanno in versioni costruite per quella
#     numpy, altrimenti quelli del container (costruiti su 1.26) non si importano.
# Prerequisito (frontend, ha git e internet): i cloni in external/now_evaluation e
# external/now_deps/{mesh,flame-fitting,eigen,boost_1_83_0}.
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

export EXTERNAL="$WBES_ROOT/external"
export VENV_NOW="$EXTERNAL/venvs/now"
export DEPS="$EXTERNAL/now_deps"
export NOW_EVAL="$EXTERNAL/now_evaluation"

echo "[now-setup] host=$(hostname) job=${SLURM_JOB_ID:-none} CONTAINER=$CONTAINER"
for d in "$NOW_EVAL" "$DEPS/mesh" "$DEPS/flame-fitting" "$DEPS/eigen" "$DEPS/boost_1_83_0/boost"; do
    [[ -d "$d" ]] || { echo "ERRORE: manca $d" >&2; exit 1; }
done

singularity exec "$CONTAINER" bash -c '
set -euo pipefail
if [[ ! -x "$VENV_NOW/bin/python3" ]]; then
    echo "[now-setup] creo il venv $VENV_NOW"
    python3 -m venv --system-site-packages --without-pip "$VENV_NOW"
fi
source "$VENV_NOW/bin/activate"
python3 -m pip install --no-input "numpy==1.23.5" "scipy==1.10.1" "matplotlib==3.7.5" \
    "cython<3" six tqdm pyopengl pyzmq pyyaml pillow "opencv-python-headless==4.8.1.78"
python3 -m pip install --no-input --no-build-isolation "chumpy==0.70"

echo "[now-setup] --- psbody-mesh ---"
export CPLUS_INCLUDE_PATH="$DEPS/boost_1_83_0${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
(cd "$DEPS/mesh" && python3 -m pip install --no-input --no-deps --no-build-isolation .)

echo "[now-setup] --- sbody + smpl_webuser + sample2meshdist ---"
rm -rf "$NOW_EVAL/sbody" "$NOW_EVAL/smpl_webuser"
cp -r "$DEPS/flame-fitting/sbody" "$NOW_EVAL/sbody"
cp -r "$DEPS/flame-fitting/smpl_webuser" "$NOW_EVAL/smpl_webuser"
cd "$NOW_EVAL/sbody/alignment/mesh_distance"
sed -i "s|^EIGEN_DIR = .*|EIGEN_DIR = \"$DEPS/eigen\"|" setup.py
python3 setup.py build_ext --inplace
python3 -m pip list --local
'

echo "[now-setup] --- import test ---"
singularity exec "$CONTAINER" bash -c '
source "$VENV_NOW/bin/activate"
cd "$NOW_EVAL"
python3 -c "
import numpy, scipy, chumpy
from psbody.mesh import Mesh
import scan2mesh_computations as s2m
from sbody.mesh_distance import ScanToMesh
print(\"numpy\", numpy.__version__, \"scipy\", scipy.__version__, \"chumpy ok, psbody ok, sbody ok\")
"
'
echo "[now-setup] OK"
