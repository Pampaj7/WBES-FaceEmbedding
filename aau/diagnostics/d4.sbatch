#!/usr/bin/env bash
#SBATCH --job-name=wbes-d4
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#
# Diagnostica D4 (aau/runs/evidence/diagnostics/PROTOCOL_D4.md), passi CPU; un passo per job:
#   D4_STEP=setup                                    sbatch aau/diagnostics/d4.sbatch   (venv .venv_d4 con pymeshlab)
# pymeshlab non c'e' ne' nel container ne' nei venv del repo: .venv_d4 (fuori da git) eredita torch / numpy / scipy del
# container (--system-site-packages, numpy < 2 come .venv_open3d) e ci aggiunge solo pymeshlab.
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
STEP="${D4_STEP:?D4_STEP: setup}"
W="${SLURM_CPUS_PER_TASK:-8}"
export AAU_NV=""
D4VENV="$WBES_ROOT/.venv_d4"
echo "[d4] host=$(hostname) job=${SLURM_JOB_ID:-none} passo=$STEP $(date +%T)"
t0=$(date +%s)
case "$STEP" in
    setup)
        VENV="$D4VENV" singularity exec "$CONTAINER" bash -c '
set -euo pipefail
[[ -x "$VENV/bin/python3" ]] || python3 -m venv --system-site-packages --without-pip "$VENV"
source "$VENV/bin/activate"
printf "numpy<2\n" > "$VENV/pip-constraints.txt"
export PIP_CONSTRAINT="$VENV/pip-constraints.txt"
python3 -m pip install --no-input pymeshlab==2025.7.post1
# il wheel e i suoi plugin linkano libGL, libOpenGL, libX11, libxcb, assenti nel container NGC: .deb di jammy
# scaricati senza root ed estratti in $VENV/apt/root (come aau/outlineB/setup_open3d.sbatch, piu libopengl0)
A="$VENV/apt"
mkdir -p "$A/lists/partial" "$A/cache/archives/partial" "$A/debs" "$A/root"
OPTS=(-o "Dir::State::Lists=$A/lists" -o "Dir::Cache=$A/cache" -o Debug::NoLocking=1
      -o "APT::Sandbox::User=$(id -un)")
apt-get "${OPTS[@]}" update
cd "$A/debs"
apt-get "${OPTS[@]}" download libx11-6 libx11-data libxcb1 libxau6 libxdmcp6 libbsd0 libmd0 \
    libgl1 libglvnd0 libglx0 libopengl0
for deb in *.deb; do dpkg-deb -x "$deb" "$A/root"; done
'
        aau/diagnostics/d4_run.sh -c "import numpy, pymeshlab; ms = pymeshlab.MeshSet(); \
ms.meshing_isotropic_explicit_remeshing; print('pymeshlab ok, numpy', numpy.__version__)" ;;
    *)         echo "passo sconosciuto: $STEP"; exit 2 ;;
esac
echo "[d4] passo $STEP finito in $(( $(date +%s) - t0 ))s"
