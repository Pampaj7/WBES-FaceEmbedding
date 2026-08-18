#!/bin/bash
# One p1i shard worker for poisson-variant generation.
# Mirrors v2_work/potential/shard_job.sh: the h100app profile caps a job at 4 cores,
# so parallelism comes from several of these rather than one wide job.
set -u
cd /dtu/p1/leopam/WBES-FaceEmbedding
export WBES_DIFFUSION_NET_SRC=$PWD/diffusion-net/src
# Open3D's Poisson reconstruction spawns a thread per core and IGNORES OMP_NUM_THREADS: setting
# the env var still left four workers driving n-62-12-83 to a load of ~100 (killing them dropped
# it to 41), and open3d 0.19 exposes no thread-count API at all -- o3d.utility and o3d.core have
# no such symbol. So the env vars below are belt-and-braces only; the real limit is taskset,
# which pins the process to a fixed CPU set at the kernel level. A library cannot spawn its way
# out of an affinity mask: extra threads just time-share the cores it is allowed.
#
# $5 = CPU list for this worker (e.g. "0-3"). Without it the shard runs unpinned, which is what
# caused the problem in the first place, so it is required rather than optional.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
CPUSET=${5:?cpu list required, e.g. 0-3}
exec taskset -c "$CPUSET" .conda_env/bin/python v2_work/poisson_aug/make_poisson_variants.py \
    --domain "$1" --shard "$2/$3" --variants "${4:-0}"
