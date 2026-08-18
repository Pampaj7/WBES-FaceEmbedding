#!/bin/bash
# When the potential-well operators are complete, start the `well` arm on the private node
# ALONGSIDE the control, and also queue it on the public queues as a backup.
#
# Running both arms concurrently on the same GPU is deliberate, not a compromise: each
# training uses ~5 GB of an 80 GB card, so memory is not the constraint, and equal
# contention makes the two arms comparable. One arm on an idle GPU and the other on a
# contended one would differ in wall-clock only, but it also risks one finishing and the
# other being killed by walltime — which would leave the A/B half-done.
set -u
cd /dtu/p1/leopam/WBES-FaceEmbedding
TARGET=3000
while true; do
  N=$(ls v2_work/potential/bfm_withwell 2>/dev/null | wc -l)
  echo "[$(date +%T)] $N/$TARGET operators"
  [ "$N" -ge "$TARGET" ] && break
  sleep 120
done
echo "[$(date +%T)] operators complete, launching the well arm"

# The thread caps are inserted here and not in _node_pot_plain.sh because the plain arm is
# already running and cannot be changed; capping only the well arm changes its wall-clock, not
# the comparison (same data, same seed, and the model math runs on the GPU).
sed -e "s|datasets/REMESH/npz_data_topo_500_withops|v2_work/potential/bfm_withwell|" \
    -e "s|runs/pot_plain|runs/pot_well|" \
    -e "s|--cache-workers 16|--cache-workers 4|" \
    -e "/^export PYTORCH_CUDA_ALLOC_CONF/a export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4" \
    v2_work/potential/_node_pot_plain.sh > v2_work/potential/_node_pot_well.sh
chmod +x v2_work/potential/_node_pot_well.sh

# private node first: it dispatches immediately
export ESUB_BYPASS=1 ESUB_QUIET=1
setsid nohup bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=90GB]" \
     -gpu "num=1:mode=shared" -W 720 -J pot_well \
     bash /dtu/p1/leopam/WBES-FaceEmbedding/v2_work/potential/_node_pot_well.sh \
     > v2_work/logs/runs/pot_well_p1i.log 2>&1 < /dev/null &

# public queues as a backup in case the interactive job is killed while pending
for q in gpua10 gpua100; do
  bsub -q $q -J pot_well -n 4 -R "span[hosts=1] rusage[mem=90GB]" \
       -gpu "num=1:mode=exclusive_process" -W 24:00 \
       -o v2_work/logs/runs/pot_well_${q}_%J.out \
       -e v2_work/logs/runs/pot_well_${q}_%J.err \
       "bash /dtu/p1/leopam/WBES-FaceEmbedding/v2_work/potential/_node_pot_well.sh"
done
echo "[$(date +%T)] well arm submitted (p1i + 2 public queues)"
