#!/bin/bash
cd /dtu/p1/leopam/WBES-FaceEmbedding
for t in 1 4 8 16 32; do
  .conda_env/bin/python v2_work/pointnet/thread_scaling.py "$t" 2>&1 | grep -v Warning
done
