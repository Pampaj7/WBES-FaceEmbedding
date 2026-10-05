#!/usr/bin/env bash
#SBATCH --job-name=wbes-oB-setup
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#
# Venv per la pipeline faceBench (rigid ICP / NICP della Tabella 2): .venv_open3d.
#   aau/submit.sh outlineB/setup_open3d.sbatch
#
# facebench.icp_align usa open3d, che non c'e' ne' nel container ne' negli altri due venv
# (chamfer_matrix.py lo dice: per questo il Chamfer della Tabella 2 era stato rifatto senza
# ICP). Versione 0.18.0, quella di faceBench/requirements.txt. Due ostacoli, trovati
# provando (job 1054596-1054619):
#  - il wheel linka libX11.so.6 e libGL.so.1, assenti nel container NGC (Ubuntu 22.04):
#    si scaricano i .deb di jammy con `apt-get download` (niente root: liste e cache in
#    una dir del venv) e si estraggono in $VENV/apt/root; run_o3d.sh li mette in
#    LD_LIBRARY_PATH. Nessuno di loro apre un display: servono solo a risolvere i simboli.
#  - facebench/__init__ importa anche l'aligner elastico, che vuole cvxpy.
# numpy resta quello del container (<2), come in .venv_baselines.
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
export VENV="$WBES_ROOT/.venv_open3d"
cd "$WBES_ROOT"

echo "[oB-setup] host=$(hostname) job=${SLURM_JOB_ID:-none} VENV=$VENV"
if [[ "${WBES_FORCE_VENV:-0}" == "1" && -d "$VENV" ]]; then
    rm -rf "$VENV"
fi

singularity exec "$CONTAINER" bash -c '
set -euo pipefail
if [[ ! -x "$VENV/bin/python3" ]]; then
    python3 -m venv --system-site-packages --without-pip "$VENV"
fi
source "$VENV/bin/activate"
printf "numpy<2\n" > "$VENV/pip-constraints.txt"
export PIP_CONSTRAINT="$VENV/pip-constraints.txt"
python3 -m pip install --no-input open3d==0.18.0 cvxpy
# i tre pacchetti di .venv_aau, alle sue versioni: common.ict_heldout_subjects importa
# rebuild_subject_split, e robustness/data_utils importa dataset_gtready, che vuole igl
# (job 1054655: ModuleNotFoundError).
python3 -m pip install --no-input libigl==2.6.3 potpourri3d==1.4.0 robust-laplacian==1.1.0
# M3DFB (v2_work/m3dfb/m3dfb_adapter.py, clonato in external/M3DFB) importa anche trimesh.
python3 -m pip install --no-input trimesh

A="$VENV/apt"
mkdir -p "$A/lists/partial" "$A/cache/archives/partial" "$A/debs" "$A/root"
OPTS=(-o "Dir::State::Lists=$A/lists" -o "Dir::Cache=$A/cache" -o Debug::NoLocking=1
      -o "APT::Sandbox::User=$(id -un)")
apt-get "${OPTS[@]}" update
cd "$A/debs"
apt-get "${OPTS[@]}" download libx11-6 libx11-data libxcb1 libxau6 libxdmcp6 libbsd0 libmd0 \
    libgl1 libglvnd0 libglx0
for deb in *.deb; do dpkg-deb -x "$deb" "$A/root"; done
'

echo "[oB-setup] --- prova d import ---"
"$AAU_DIR/outlineB/run_o3d.sh" -c "
import sys; sys.path.insert(0, 'faceBench'); sys.path.insert(0, 'faceBench/latentVSpipeline')
import open3d, facebench, run_facebench_remesh
print('open3d', open3d.__version__, 'facebench ok')"
echo "[oB-setup] OK"
