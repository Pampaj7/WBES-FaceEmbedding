#!/bin/bash
# Poisson-variant generation as an LSF array job: make_poisson_variants.py already shards
# by subject (--shard i/n), so each task is one shard. ~55-60s/subject (2 variants) serial;
# at 40 tasks that is minutes of wall time instead of ~17h.
#
# Usage: bash v2_work/poisson_aug/poisson_array.sh <domain: bfm|flame> [n_subjects] [n_tasks] [queue]
set -euo pipefail
ROOT=/dtu/p1/leopam/WBES-FaceEmbedding
DOMAIN=${1:?domain: bfm or flame}; NSUBJ=${2:-0}; NTASK=${3:-40}; QUEUE=${4:-hpc}
cd "$ROOT"
mkdir -p v2_work/logs/array
TASK="$ROOT/v2_work/poisson_aug/.task_${DOMAIN}.sh"

{
  echo '#!/bin/bash'
  echo 'set -euo pipefail'
  echo "cd $ROOT"
  echo "export WBES_DIFFUSION_NET_SRC=$ROOT/diffusion-net/src"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4"
  echo "exec .conda_env/bin/python v2_work/poisson_aug/make_poisson_variants.py \\"
  echo "    --domain $DOMAIN --n-subjects $NSUBJ \\"
  echo "    --shard \$((LSB_JOBINDEX-1))/$NTASK"
} > "$TASK"
chmod +x "$TASK"

bsub -q "$QUEUE" -J "pois_${DOMAIN}[1-$NTASK]" -n 4 -W 4:00 \
     -R "span[hosts=1] rusage[mem=8GB]" \
     -o "$ROOT/v2_work/logs/array/pois_${DOMAIN}_%J_%I.out" \
     -e "$ROOT/v2_work/logs/array/pois_${DOMAIN}_%J_%I.err" \
     "$TASK"
