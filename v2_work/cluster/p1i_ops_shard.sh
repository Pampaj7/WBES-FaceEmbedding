#!/bin/bash
# One p1i worker for the standard operator precompute. The v1 script has no --shard flag and
# is frozen, so the slice is expressed as a symlink view of the inputs this task owns.
set -u
cd /dtu/p1/leopam/WBES-FaceEmbedding
export WBES_DIFFUSION_NET_SRC=$PWD/diffusion-net/src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
IN=$1; OUT=$2; S=$3; N=$4
SL="$OUT/.p1islices/t$(printf '%02d' "$S")"
.conda_env/bin/python - "$IN" "$OUT" "$SL" "$S" "$N" <<'PY'
import sys
from pathlib import Path
src, out, sl, s, n = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
out.mkdir(parents=True, exist_ok=True); sl.mkdir(parents=True, exist_ok=True)
for f in sl.iterdir(): f.unlink()
todo = [p for p in sorted(src.glob("*.npz")) if not (out / p.name).exists()][s::n]
for p in todo: (sl / p.name).symlink_to(p.resolve())
print(f"shard {s}/{n}: {len(todo)} meshes", flush=True)
PY
exec .conda_env/bin/python face_embedding/gt_encdec/autoencoder/precompute_operators_npz.py \
    --input-dir "$SL" --output-dir "$OUT" --k-eig 128 --n-cores 4
