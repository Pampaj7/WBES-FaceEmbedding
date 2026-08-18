#!/bin/bash
# End-to-end test of the potential-well hypothesis, on the BFM benchmark.
# Prediction stated in potential_operators.py before running: the well must raise the
# crop-involving and Poisson-mate numbers and leave pure retessellation alone.
set -u
cd /dtu/p1/leopam/WBES-FaceEmbedding
export WBES_DIFFUSION_NET_SRC=$PWD/diffusion-net/src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
NPROC=${1:-12}

echo "[$(date +%T)] potential operators for BFM (3000 meshes, $NPROC shards)"
IN=datasets/REMESH/npz_data_topo_500
OUT=v2_work/potential/bfm_withwell
mkdir -p "$OUT" v2_work/potential/_slices
# shard by symlink views, same trick as the LSF array launchers
.conda_env/bin/python - "$IN" v2_work/potential/_slices "$NPROC" "$OUT" <<'PY'
import sys
from pathlib import Path
src, sl, n, out = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
todo = [p for p in sorted(src.glob("*.npz")) if not (out / p.name).exists()]
print(f"{len(todo)} meshes still to compute")
for t in range(n):
    d = sl / f"t{t:02d}"
    if d.exists():
        for f in d.iterdir():
            f.unlink()
    d.mkdir(parents=True, exist_ok=True)
for i, p in enumerate(todo):
    (sl / f"t{i % n:02d}" / p.name).symlink_to(p.resolve())
PY
for i in $(seq 0 $((NPROC-1))); do
  i=$(printf "%02d" $i)
  .conda_env/bin/python v2_work/potential/potential_operators.py \
      --input-dir v2_work/potential/_slices/t$i --output-dir "$OUT" --k-eig 128 \
      > v2_work/logs/potential_t$i.log 2>&1 &
done
wait
echo "[$(date +%T)] done: $(ls $OUT | wc -l) files"
