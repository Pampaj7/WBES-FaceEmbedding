#!/bin/bash
# Support-variant bank as an LSF array: the generator already shards by subject
# (--shard i/n), so each task is one shard. 500 BFM subjects x 5 variants is
# ~5.7 h on one core; at 40 tasks it is minutes of wall time.
#
# Usage: bash v2_work/cluster/support_bank_array.sh <in_dir> <out_dir> <n_subjects> [n_variants] [n_tasks] [queue]
set -euo pipefail
ROOT=/dtu/p1/leopam/WBES-FaceEmbedding
IN=${1:?original-mesh dir}; OUT=${2:?out dir}; NSUBJ=${3:-0}; NVAR=${4:-5}
NTASK=${5:-40}; QUEUE=${6:-hpc}
cd "$ROOT"; IN=$(readlink -f "$IN"); OUT=$(readlink -f "$OUT")
mkdir -p "$OUT" v2_work/logs/array
TASK="$OUT/.task.sh"

{
  echo '#!/bin/bash'
  echo 'set -euo pipefail'
  echo "cd $ROOT"
  echo "export WBES_DIFFUSION_NET_SRC=$ROOT/diffusion-net/src"
  echo 'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1'
  echo "exec .conda_env/bin/python v2_work/train_v2/make_support_bank.py \\"
  echo "    --in-dir '$IN' --out-dir '$OUT' --n-subjects $NSUBJ --n-variants $NVAR \\"
  echo "    --shard \$((LSB_JOBINDEX-1))/$NTASK"
} > "$TASK"
chmod +x "$TASK"

bsub -q "$QUEUE" -J "supp[1-$NTASK]" -n 1 -W 4:00 \
     -R "span[hosts=1] rusage[mem=8GB]" \
     -o "$ROOT/v2_work/logs/array/supp_%J_%I.out" \
     -e "$ROOT/v2_work/logs/array/supp_%J_%I.err" \
     "$TASK"
