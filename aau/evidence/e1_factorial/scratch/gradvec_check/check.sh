#!/usr/bin/env bash
# E1: il pre-pass del trainer con il build_grad originale e con quello vettorizzato (gradvec_site), stesse mesh,
# 1 processo ciascuno su questo nodo: tempi (cpu_seconds del pre-pass) e confronto dei tensori salvati.
set -euo pipefail
source aau/env.sh
export AAU_NV="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
O=/tmp/e1_gradvec_${SLURM_JOB_ID}
rm -rf $O; mkdir -p $O
SH=datasets/SCALE_ALL/shards
D=aau/evidence/e1_factorial/scratch/gradvec_check
run() {
    aau/run.sh aau/data_scale/prepass_ops.py --compress --out-dir $O/$1 --subjects $D/subjects.txt \
        --convention areanorm --n-proc 1 --tars $(ls $SH/*shard_*.tar) --tar-index $SH/index.npz > $O/$1.log 2>&1
    grep -E "cpu_seconds" $O/$1.log | tail -1 > $D/$1.json.txt
}
run orig
WBES_E1_GRADVEC=1 PYTHONPATH="$PWD/aau/evidence/e1_factorial/gradvec_site:${PYTHONPATH:-}" run vec
echo "righe [e1-gradvec] nel log vettorizzato: $(grep -c '\[e1-gradvec\]' $O/vec.log); nell'originale: $(grep -c '\[e1-gradvec\]' $O/orig.log || true)"
aau/run.sh $D/compare.py $O/orig $O/vec $D/orig.json.txt $D/vec.json.txt "$(hostname)" "$(grep -c '\[e1-gradvec\]' $O/vec.log)"
rm -rf $O
