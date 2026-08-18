#!/bin/bash
# Launch N p1i interactive shards for one domain, backgrounded.
#
# hpc/milan batch queues are hard-limited to 0 concurrent jobs for this account
# (bjobs -l shows "JOBS limit ... Limit Name: limit-p1-non-dtu-users, Limit Value: 0" --
# confirmed empirically, and v2_work/potential/milan_array.sh's own potops[1-8] array sits
# PEND under the same limit). p1i interactive (bsub -I, -app h100app, 4 cores/job) is the
# pattern that actually runs, mirrored here from v2_work/potential/shard_job.sh.
#
# Usage: bash launch_p1i_shards.sh <domain: bfm|flame> [n_shards]
set -u
ROOT=/dtu/p1/leopam/WBES-FaceEmbedding
DOMAIN=${1:?domain}; N=${2:-6}
cd "$ROOT"
export ESUB_BYPASS=1 ESUB_QUIET=1
mkdir -p v2_work/logs
for i in $(seq 0 $((N-1))); do
  bsub -I -q p1i -app h100app -n 2 -R "span[hosts=1] rusage[mem=2GB]" \
       -gpu "num=1:mode=shared" -W 600 -J "pois_${DOMAIN}_${i}" \
       "bash $ROOT/v2_work/poisson_aug/p1i_shard.sh $DOMAIN $i $N" \
       > "$ROOT/v2_work/logs/pois_${DOMAIN}_${i}.log" 2>&1 &
done
wait
echo "all $N $DOMAIN shards finished"
