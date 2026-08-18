#!/bin/bash
# FLAME topology generation as an LSF array job. Same slicing trick as
# precompute_array.sh: each task gets a symlink view of its slice of identities.
# Usage: bash v2_work/cluster/topo_array.sh <identities_dir> <out_dir> [n_tasks] [queue] [cores]
set -euo pipefail
ROOT=/dtu/p1/leopam/WBES-FaceEmbedding
IN=${1:?identities dir}; OUT=${2:?out dir}; NTASK=${3:-40}; QUEUE=${4:-hpc}; CORES=${5:-1}
cd "$ROOT"; IN=$(readlink -f "$IN"); OUT=$(readlink -f "$OUT")
SLICE_ROOT="$OUT/.slices"; mkdir -p "$OUT" "$SLICE_ROOT" v2_work/logs/array

.conda_env/bin/python - "$IN" "$SLICE_ROOT" "$NTASK" "$OUT" <<'PY'
import sys
from pathlib import Path
in_dir, slice_root, ntask, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
VARIANTS = ("original", "remesh", "crop", "noisy", "down8k", "up60k")
ids = sorted(p for p in in_dir.glob("flame*.npz"))
todo = [p for p in ids
        if not all((out_dir / f"{p.stem}_GTready_{v}.npz").exists() for v in VARIANTS)]
print(f"{len(ids)} identities, {len(todo)} incomplete")
for t in range(ntask):
    d = slice_root / f"task{t+1:03d}"
    if d.exists():
        for old in d.iterdir():
            old.unlink()
    d.mkdir(parents=True, exist_ok=True)
for i, p in enumerate(todo):
    (slice_root / f"task{i % ntask + 1:03d}" / p.name).symlink_to(p.resolve())
PY

TASK="$SLICE_ROOT/task.sh"
{
  echo '#!/bin/bash'
  echo 'set -euo pipefail'
  echo "cd $ROOT"
  echo 'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1'
  echo "SLICE=\$(printf '$SLICE_ROOT/task%03d' \$LSB_JOBINDEX)"
  echo "exec .conda_env/bin/python v2_work/genflame/make_flame_topologies.py --in-dir \"\$SLICE\" --out-dir '$OUT' --n-cores $CORES"
} > "$TASK"
chmod +x "$TASK"

bsub -q "$QUEUE" -J "topo[1-$NTASK]" -n "$CORES" -W 4:00 \
     -R "span[hosts=1] rusage[mem=6GB]" \
     -o "$ROOT/v2_work/logs/array/topo_%J_%I.out" \
     -e "$ROOT/v2_work/logs/array/topo_%J_%I.err" \
     "$TASK"
