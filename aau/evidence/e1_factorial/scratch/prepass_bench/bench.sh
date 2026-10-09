#!/usr/bin/env bash
# un'identita' ICT nuova (14 mesh) col pre-pass del trainer, 1 processo: CPU-s per mesh su questo nodo
set -euo pipefail
source aau/env.sh
export AAU_NV="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
O=/tmp/e1_prepass_bench_${SLURM_JOB_ID}
rm -rf $O; mkdir -p $O
SH=datasets/SCALE_ALL/shards
t0=$(date +%s)
aau/run.sh aau/data_scale/prepass_ops.py --compress --out-dir $O/out --subjects aau/evidence/e1_factorial/scratch/prepass_bench/subjects.txt \
  --convention areanorm --n-proc ${NPROC:-1} --tars $(ls $SH/*shard_*.tar) --tar-index $SH/index.npz 2>&1 | grep -E "n_meshes|cpu_seconds" | tail -1
echo "host=$(hostname) cpu=$(lscpu | grep 'Model name' | sed 's/.*: *//') wall=$(( $(date +%s) - t0 )) s"
rm -rf $O
