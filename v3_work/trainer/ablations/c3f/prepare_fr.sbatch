#!/usr/bin/env bash
#SBATCH --job-name=wbes-v3-c3f-factprep
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#
# Dati dei bracci C3F di factorized_protocol.md, appena E12 scrive datasets/CANONICAL_GT/train/READY:
#   gt_sr.npz    = GT-SR grezza di E12 ritagliata sui 6.993 soggetti di c3f/gt.npz (make_testdata.py --gt-only)
#   gt_frcal.npz = GT-FR TARATA di E12, stesso ritaglio
#   fr_params.json: kappa = media geometrica dei fattori di taratura per dominio di SR (un valore, protocollo)
#   spec_s2345.json: la spec con block_seed 2345 (secondo seme)
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export AAU_NV=
C="$AAU_RUNS/evidence/trainer_v3/ablations/c3f"
CGT="$WBES_ROOT/datasets/CANONICAL_GT/train"
until [[ -f "$CGT/READY" ]]; do sleep 60; done
echo "[factprep] $(date +%F_%T) READY: $(cat "$CGT/READY")"
"$AAU_DIR/run.sh" v3_work/trainer/tests/make_testdata.py --gt-only --out-dir "$C" \
  --gt-src "$CGT/gt_sr_bfm_ict_gnm.npz" --gt-name gt_sr
"$AAU_DIR/run.sh" v3_work/trainer/tests/make_testdata.py --gt-only --out-dir "$C" \
  --gt-src "$CGT/gt_fr_bfm_ict_gnm_calib.npz" --gt-name gt_frcal
python3 - "$CGT/gt_sr_bfm_ict_gnm_calib.json" "$C/fr_params.json" "$CGT/READY" <<'PY'
import json, math, sys
f = json.load(open(sys.argv[1]))["factor"]
k = math.exp(sum(math.log(f[d]) for d in ("bfm", "ict", "gnm")) / 3)
json.dump({"kappa": k, "factors_sr_calib": f, "rule": "media geometrica dei fattori per dominio (factorized_protocol.md)",
           "ready": open(sys.argv[3]).read().strip()}, open(sys.argv[2], "w"), indent=1)
print("kappa", k, f)
PY
python3 - "$C/spec.json" "$C/spec_s2345.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1])); s["block_seed"] = 2345
json.dump(s, open(sys.argv[2], "w"), indent=1)
PY
echo "[factprep] $(date +%F_%T) OK"
