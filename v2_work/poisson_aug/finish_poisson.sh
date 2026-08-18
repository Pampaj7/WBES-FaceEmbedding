#!/usr/bin/env bash
# Chains the rest of the Poisson pipeline so it completes unattended:
#   generation (already running) -> operators -> rebuilt views
#
# Everything stays pinned to cores 0-23 with taskset. The operator precompute is as
# parallel-happy as Open3D's Poisson was, and OMP_NUM_THREADS demonstrably does not hold it
# (see STATUS 22:10), so affinity is the only limit that actually binds. 24 cores of 128 keeps
# us under the "at most half the CPU" the user asked for even with both A/B trainings running.
set -u
cd /dtu/p1/leopam/WBES-FaceEmbedding
export WBES_DIFFUSION_NET_SRC=$PWD/diffusion-net/src
CPUS=0-23
WANT_BFM=1000     # 500 subjects x pois0,pois1
WANT_FLAME=1200   # 600 subjects x pois0,pois1

wait_for () {  # dir, expected count, label
  while true; do
    n=$(ls "$1" 2>/dev/null | grep -c pois)
    echo "[$(date +%T)] $3: $n/$2"
    [ "$n" -ge "$2" ] && break
    # stop waiting if no generation job is left to make progress
    if [ "$(bjobs -w 2>/dev/null | grep -cE 'p0_|p1_')" -eq 0 ]; then
      echo "[$(date +%T)] $3: no generation job alive, stopping at $n"
      break
    fi
    sleep 60
  done
}

wait_for v2_work/poisson_aug/bfm_poisson   "$WANT_BFM"   "bfm generation"
wait_for v2_work/poisson_aug/flame_poisson "$WANT_FLAME" "flame generation"

echo "[$(date +%T)] generation done, computing operators"
for dom in bfm flame; do
  IN=v2_work/poisson_aug/${dom}_poisson
  OUT=v2_work/poisson_aug/${dom}_poisson_withops
  mkdir -p "$OUT"
  taskset -c "$CPUS" .conda_env/bin/python \
      face_embedding/gt_encdec/autoencoder/precompute_operators_npz.py \
      --input-dir "$IN" --output-dir "$OUT" --k-eig 128 --n-cores 12 \
      2>&1 | tail -20
  echo "[$(date +%T)] $dom operators: $(ls $OUT | wc -l)"
done

echo "[$(date +%T)] rebuilding views"
taskset -c "$CPUS" .conda_env/bin/python v2_work/poisson_aug/build_views.py 2>&1 | tail -10

for v in bfm_view flame_view; do
  d=v2_work/poisson_aug/$v
  echo "--- $v: $(ls $d 2>/dev/null | wc -l) files ---"
  ls "$d" 2>/dev/null | sed -E 's/.*_(original|remesh|crop|noisy|down8k|up60k|pois0|pois1)\.npz/\1/' \
    | sort | uniq -c
done
echo "[$(date +%T)] poisson pipeline complete"
