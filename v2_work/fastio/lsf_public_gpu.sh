#!/bin/bash
# Same FLAME training, submitted to a PUBLIC GPU queue as a real batch job.
# Rationale: the private p1i queue is a single node whose 2 H100s are occupied by another
# of our own jobs, so a shared allocation there runs ~5x slower than an uncontended GPU.
# Public queues are congested but dispatch eventually, accept -W, and do not need a live
# terminal (unlike p1i, which only takes interactive jobs).
#
# Usage: bash lsf_public_gpu.sh <tag> <batch_subjects> <lr> [queue]
set -euo pipefail
ROOT=/dtu/p1/leopam/WBES-FaceEmbedding
TAG=${1:?tag}; BS=${2:?bs}; LR=${3:?lr}; Q=${4:-gpua100}
bsub -q "$Q" -J "$TAG" -n 4 -R "span[hosts=1] rusage[mem=48GB]" \
     -gpu "num=1:mode=exclusive_process" -W 24:00 \
     -o "$ROOT/v2_work/logs/runs/${TAG}_%J.out" \
     -e "$ROOT/v2_work/logs/runs/${TAG}_%J.err" \
     "bash $ROOT/v2_work/fastio/_node_${TAG}.sh"
