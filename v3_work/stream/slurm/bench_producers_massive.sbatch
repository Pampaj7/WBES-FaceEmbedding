#!/usr/bin/env bash
#SBATCH --job-name=wbes-stream-pbench
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#
# Viste fresche/s dei produttori nella configurazione del run massivo (fonti massive, --provenance --canonical-gt
# --expr-frac 0.5, autovettori fp32) con k 64 e k 128, PBENCH_SECONDS per configurazione, P processi (PBENCH_P, lista;
# default tutte le CPU del job). Misure in aau/runs/evidence/stream/producers_massive_<job>/k<k>_p<P>.json.
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export AAU_NV=
E="$AAU_RUNS/evidence/stream/producers_massive_${SLURM_JOB_ID:-manual}"
mkdir -p "$E"
trap 'rm -rf "/tmp/${SLURM_JOB_ID:-manual}_pb"' EXIT
echo "[pbench] host=$(hostname) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) nproc=$(nproc)"
for k in ${PBENCH_K:-64 128}; do
for P in ${PBENCH_P:-$(nproc)}; do
  R="/tmp/${SLURM_JOB_ID:-manual}_pb/k$k"
  "$AAU_DIR/run.sh" v3_work/stream/producer.py --ring "$R" --ring-gb 16 --n-proc "$P" --k-eig "$k" \
    --evecs-dtype fp32 --sources massive --provenance --canonical-gt --expr-frac 0.5 --duration "${PBENCH_SECONDS:-300}" \
    --stats-every 60 --summary "$E/k${k}_p$P.json" 2>&1 | grep "^\[producer\] *[0-9]" || true
  rm -rf "$R"
  python3 -c "import json; s = json.load(open('$E/k${k}_p$P.json')); print('k=$k P=$P', round(s['views_per_s_steady'], 1), 'viste/s,', \
round(s['bytes_per_view'] / 2**20, 2), 'MiB/vista, verts', round(s['mean_verts_per_view']), 'fallite', s['failures'])"
done
done
