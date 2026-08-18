#!/usr/bin/env bash
# Closes the FLAME side of the Poisson pipeline.
#
# finish_poisson.sh completed at 22:43 while a generation worker was still producing the last
# 294 pois1 meshes: its wait loop had already given up (correctly -- at that moment no worker
# was alive) and moved on, so those meshes exist but have no operators and never reached the
# view. BFM is unaffected and complete (4000/4000).
#
# Pinned to the same cores the rest of the Poisson work used, per DTU support's request.
set -u
cd /dtu/p1/leopam/WBES-FaceEmbedding
export WBES_DIFFUSION_NET_SRC=$PWD/diffusion-net/src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
CPUS=0-15

# wait for generation to be complete before computing operators for it
while true; do
  n=$(ls v2_work/poisson_aug/flame_poisson 2>/dev/null | wc -l)
  echo "[$(date +%T)] flame_poisson: $n/1200"
  [ "$n" -ge 1200 ] && break
  [ "$(bjobs -w 2>/dev/null | grep -cE 'pf[01]')" -eq 0 ] && { echo "nessun worker, procedo con $n"; break; }
  sleep 120
done

echo "[$(date +%T)] operatori FLAME (skippa i 906 già fatti)"
taskset -c "$CPUS" .conda_env/bin/python \
    face_embedding/gt_encdec/autoencoder/precompute_operators_npz.py \
    --input-dir v2_work/poisson_aug/flame_poisson \
    --output-dir v2_work/poisson_aug/flame_poisson_withops \
    --k-eig 128 --n-cores 8 2>&1 | tail -5
echo "[$(date +%T)] flame operatori: $(ls v2_work/poisson_aug/flame_poisson_withops | wc -l)/1200"

echo "[$(date +%T)] ricostruzione viste"
taskset -c "$CPUS" .conda_env/bin/python v2_work/poisson_aug/build_views.py 2>&1 | tail -5

for v in bfm_view flame_view; do
  d=v2_work/poisson_aug/$v
  echo "--- $v: $(ls $d 2>/dev/null | wc -l) file ---"
  ls "$d" 2>/dev/null | sed -E 's/.*_(original|remesh|crop|noisy|down8k|up60k|pois0|pois1)\.npz/\1/' \
    | sort | uniq -c
done
echo "[$(date +%T)] FLAME completo"
