#!/usr/bin/env bash
#SBATCH --job-name=wbes-massive-heldout
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#
# Held-out del monitoraggio del modello da rilasciare (aau/massive/heldout.py make, PLAN_MASSIVE sez. 23): shard in
# datasets/MASSIVE_HELDOUT/ring (fuori da git), dichiarazione in datasets/MASSIVE_HELDOUT/heldout.json, copiata in
# aau/massive/heldout.json (nel commit). Solo CPU.
set -euo pipefail
export AAU_NV=
OUT="${HELDOUT_OUT:-datasets/MASSIVE_HELDOUT}"
aau/run.sh aau/massive/heldout.py make --out "$OUT" --workers "${SLURM_CPUS_PER_TASK:-32}" ${HELDOUT_ARGS:-}
python3 - "$OUT/heldout.json" aau/massive/heldout.json <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
d.pop("sources", None)          # descrizioni lunghe: restano nel file completo accanto agli shard
open(sys.argv[2], "w").write(json.dumps(d, indent=1) + "\n")
PY
du -sh "$OUT"
