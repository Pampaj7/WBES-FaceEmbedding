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
#   D4_STEP=prep  sbatch --cpus-per-task=64 --mem=64G --time=01:00:00 aau/diagnostics/d4.sbatch
#                 (remesh canonico, tabelle di scala, operatori areanorm k 128, validita'; poi d4_embed.sbatch)
#   D4_STEP=cost  sbatch --time=01:00:00 aau/diagnostics/d4.sbatch                  (costo degli operatori, sez. 6)
#   D4_STEP=stats sbatch --cpus-per-task=64 --mem=128G aau/diagnostics/d4.sbatch     (dopo d4_embed.sbatch)
# Mesh canoniche in datasets/DIAG_D4/<vista>/in, operatori in datasets/V3_OPS_CACHE/diag_d4/<vista>/ops (fuori da git,
# ~20 GB, da cancellare a fine lavoro).
# pymeshlab non c'e' ne' nel container ne' nei venv del repo: .venv_d4 (fuori da git) eredita torch / numpy / scipy del
# container (--system-site-packages, numpy < 2 come .venv_open3d) e ci aggiunge solo pymeshlab.
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
STEP="${D4_STEP:?D4_STEP: setup | prep | cost | stats}"
W="${SLURM_CPUS_PER_TASK:-8}"
export AAU_NV=""
D4VENV="$WBES_ROOT/.venv_d4"
D=aau/diagnostics
D4="$AAU_RUNS/evidence/diagnostics/d4"
STAGE="$WBES_ROOT/datasets/DIAG_D4"
OPS="$WBES_ROOT/datasets/V3_OPS_CACHE/diag_d4"
VIEWS="facescape hifi3d faceverse_neutral"
echo "[d4] host=$(hostname) job=${SLURM_JOB_ID:-none} passo=$STEP $(date +%T)"
t0=$(date +%s)
ops() {  # <cartella delle mesh> <cartella degli operatori>: areanorm k_eig 128 in W-1 shard, controllo del conteggio
  local src="$1" out="$2" i N M pids=()
  N=$(find "$src" -name "*.npz" | wc -l)
  mkdir -p "$out"
  for (( i = 0; i < W - 1; i++ )); do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 aau/run.sh v2_work/potential/areanorm_operators.py \
      --input-dir "$src" --output-dir "$out" --k-eig 128 --shard "$i/$((W - 1))" \
      > "/tmp/wbes_d4_ops_${SLURM_JOB_ID:-x}.$i.log" 2>&1 & pids+=($!)
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "ERRORE: shard $i"; cat "/tmp/wbes_d4_ops_${SLURM_JOB_ID:-x}.$i.log"; return 1; }
  done
  grep -h FAIL /tmp/wbes_d4_ops_"${SLURM_JOB_ID:-x}".*.log || true
  rm -f /tmp/wbes_d4_ops_"${SLURM_JOB_ID:-x}".*.log
  M=$(find "$out" -name "*.npz" ! -name ".*" | wc -l)
  echo "[d4] $(date +%T) operatori $M / $N in $out, $(du -sh "$out" | cut -f1)"
}
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
    prep)
        aau/diagnostics/d4_run.sh $D/d4_remesh.py remesh --workers "$((W - 1))"
        for v in $VIEWS; do
          mkdir -p "$D4/$v"
          aau/run.sh v3_work/trainer/tools/build_scale_table.py --view-dir "$STAGE/$v/in" --domain "$(cat "$STAGE/$v/domain")" \
            --out "$D4/$v/scale_table.npz" --workers 32 | tail -2
        done
        echo "[d4] $(date +%T) tabelle fatte"
        for v in $VIEWS; do ops "$STAGE/$v/in" "$OPS/$v/ops"; done
        aau/run.sh $D/d4_remesh.py check --workers "$W" ;;
    cost)      aau/run.sh $D/d4_cost.py ;;
    stats)     UGT_THREADS=1 v3_work/unified_gt/run.sh $D/d4_stats.py --workers "$W" ;;
    *)         echo "passo sconosciuto: $STEP"; exit 2 ;;
esac
echo "[d4] passo $STEP finito in $(( $(date +%s) - t0 ))s"
