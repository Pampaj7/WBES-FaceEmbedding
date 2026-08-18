#!/usr/bin/env bash
# Launch one arm of the potential A/B on the private node, asking for an EXCLUSIVE GPU so
# LSF hands us the idle one rather than sharing the busy one.
set -u
ROOT=/dtu/p1/leopam/WBES-FaceEmbedding
ARM=${1:?well|plain}
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=64GB]" \
     -gpu "num=1:mode=shared" -W 720 -J "pot_${ARM}" \
     bash "$ROOT/v2_work/potential/_node_pot_${ARM}.sh"
