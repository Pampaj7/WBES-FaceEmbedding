#!/bin/bash
# Operator precompute as an LSF array job: each task takes a disjoint slice of the
# input files, so 30k meshes take one slice-time instead of the serial 3.5h.
#
# The upstream script has no slicing option, so slicing is done by giving each task
# its own input view: a directory of symlinks to its slice. Cheap, and it leaves
# datasets/ and the shared scripts untouched.
#
# Usage:
#   bash v2_work/cluster/precompute_array.sh <in_dir> <out_dir> [n_tasks] [queue]
set -euo pipefail

ROOT=/dtu/p1/leopam/WBES-FaceEmbedding
IN=${1:?input dir}
OUT=${2:?output dir}
NTASK=${3:-40}
QUEUE=${4:-milan}

cd "$ROOT"
IN=$(readlink -f "$IN"); OUT=$(readlink -f "$OUT")
SLICE_ROOT="$OUT/.slices"
mkdir -p "$OUT" "$SLICE_ROOT" v2_work/logs/array

# Build the per-task input views once (idempotent: rebuilt each launch so a
# re-run after adding meshes picks them up).
.conda_env/bin/python - "$IN" "$SLICE_ROOT" "$NTASK" "$OUT" <<'PY'
import sys, os
from pathlib import Path
in_dir, slice_root, ntask, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
files = sorted(p for p in in_dir.glob("*.npz"))
todo = [p for p in files if not (out_dir / p.name).exists()]
print(f"{len(files)} inputs, {len(todo)} still to compute")
for t in range(ntask):
    d = slice_root / f"task{t+1:03d}"
    if d.exists():
        for old in d.iterdir():
            old.unlink()
    d.mkdir(parents=True, exist_ok=True)
for i, p in enumerate(todo):
    d = slice_root / f"task{i % ntask + 1:03d}"
    (d / p.name).symlink_to(p.resolve())
PY

cat > "$SLICE_ROOT/task.sh" <<EOF
#!/bin/bash
set -euo pipefail
cd $ROOT
export WBES_DIFFUSION_NET_SRC=$ROOT/diffusion-net/src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
SLICE=\$(printf "$SLICE_ROOT/task%03d" \$LSB_JOBINDEX)
exec .conda_env/bin/python face_embedding/gt_encdec/autoencoder/precompute_operators_npz.py \\
    --input-dir "\$SLICE" --output-dir "$OUT" --k-eig 128 --n-cores 4
EOF
chmod +x "$SLICE_ROOT/task.sh"

bsub -q "$QUEUE" -J "ops[1-$NTASK]" -n 4 -W 8:00 \
     -R "span[hosts=1] rusage[mem=8GB]" \
     -o "$ROOT/v2_work/logs/array/ops_%J_%I.out" \
     -e "$ROOT/v2_work/logs/array/ops_%J_%I.err" \
     "$SLICE_ROOT/task.sh"
