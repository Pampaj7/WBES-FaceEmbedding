#!/usr/bin/env bash
#SBATCH --job-name=wbes-stream-tests
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#
# Correttezza (a) operatori/embedding e (b) GT al volo: aau/runs/evidence/stream/{ops_equiv,gt_check}.json
set -euo pipefail
export AAU_NV=
W="/tmp/stream_tests_${SLURM_JOB_ID:-manual}"
mkdir -p "$W"
trap 'rm -rf "$W"' EXIT
E=aau/runs/evidence/stream
aau/run.sh v3_work/stream/tests/test_ops_equiv.py --work "$W/eq" --out "$E/ops_equiv.json"
aau/run.sh v3_work/stream/tests/test_gt.py --out "$E/gt_check.json"
