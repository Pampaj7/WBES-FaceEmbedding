#!/usr/bin/env bash
#SBATCH --job-name=wbes-abl-F-smoke
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#
# Smoke della variante F (BOARD_DIARY, ipotesi H6): 2 epoche su 10 soggetti, con lo stesso sbatch del
# run vero (aau/train_remesh_frame.sbatch, frame current, cache in RAM). Dopo aau/models/view_F.sbatch.
#   1. vista di 10 soggetti di training del seed 1234 (id dei primi 10): i 60 npz standard + i crop
#      casuali dei soli soggetti che lo split a 10 (seed 1234) mette in training, cosi' gli held-out
#      dello smoke hanno le 6 topologie standard come nel run vero;
#   2. come il trainer vede view_F intera: etichette per soggetto e frequenza con cui
#      _sample_subject_mesh_entries pesca ciascun file `crop` (atteso 1/6: canonico + 5 casuali);
#   3. training di 2 epoche; controlli: eval online sulle 6 topologie standard, loss finite.
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

OUT=$AAU_DIR/scratch/abl_F/smoke_runs/${SLURM_JOB_ID:-manual}
VIEW=$AAU_DIR/scratch/abl_F/smoke_view
FULL=$WBES_ROOT/datasets/REMESH/view_F
mkdir -p "$OUT"
rm -rf "$VIEW"

echo "================ 1-2. vista dello smoke, etichette e campionamento"
SMOKE_VIEW="$VIEW" SMOKE_FULL="$FULL" AAU_NV= "$AAU_DIR/run.sh" - <<'PY'
import os, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, "aau/models")
from size_token import bfm_train_subjects
view, full = Path(os.environ["SMOKE_VIEW"]), Path(os.environ["SMOKE_FULL"])
dist = os.environ["WBES_DIST_NPZ"]
files = sorted(f for f in os.listdir(full) if f.endswith(".npz"))
std = [f[:-4] for f in files if "_crop_r" not in f]
train = bfm_train_subjects(std, dist, 1234)
from intrinsic_utils import SUBJECT_RE_ANY, build_subject_map
from robustness.data_utils import infer_topology_label_from_name, rebuild_subject_split
import robustness.train_runner as tr

ten = train[:10]
s_train, s_held = rebuild_subject_split(subjects=ten, eval_fraction=0.2, seed=1234, max_subjects=0)
view.mkdir(parents=True)
for f in files:
    sid = f.split("_")[0]
    if sid in ten and ("_crop_r" not in f or sid in s_train):
        os.symlink(os.path.realpath(full / f), view / f)
print(f"[smoke] vista {view}: {len(os.listdir(view))} npz; training {s_train}; held-out {s_held}")

class Names:  # basta la lista dei file: _build_subject_topology_map non apre gli npz
    def __init__(self, files): self.files = files
ds = Names(files)
subj_map = build_subject_map(files, subject_re=SUBJECT_RE_ANY)
topo = tr._build_subject_topology_map(dataset=ds, subj_map=subj_map)
labels = Counter(tuple(sorted((t, len(v)) for t, v in topo[s].items())) for s in topo)
for k, n in labels.items():
    print(f"[smoke] view_F: {n} soggetti con etichette {dict(k)}")
rng = np.random.default_rng(0)
picked = Counter()
for _ in range(20):  # 20 epoche simulate sui 400 soggetti di training, 6 mesh per soggetto
    for sid in train:
        for idx, t in tr._sample_subject_mesh_entries(sid=sid, subject_topology_map=topo, max_meshes=6, rng=rng):
            if t == "crop":
                picked[files[idx].split("GTready_")[1][:-4]] += 1
tot = sum(picked.values())
print("[smoke] file pescati per lo slot crop (20 epoche simulate): "
      + ", ".join(f"{k} {v / tot:.3f}" for k, v in sorted(picked.items())) + f" su {tot}")
PY

echo "================ 3. training di 2 epoche"
t0=$(date +%s)
WBES_DATA_DIR=$VIEW WBES_FRAME=current WBES_EPOCHS=2 WBES_RUNS_ROOT=$OUT/F bash aau/train_remesh_frame.sbatch
echo "[smoke] training 2 epoche: $(( $(date +%s) - t0 )) s"

RUN=$(ls -d "$OUT/F"/*/)
SMOKE_RUN="$RUN" AAU_NV= "$AAU_DIR/run.sh" - <<'PY'
import csv, json, math, os
from pathlib import Path
run = Path(os.environ["SMOKE_RUN"])
labels = sorted(json.loads((run / "online_mesh_eval_summary.json").read_text())["topology_labels"])
assert labels == sorted(["original", "remesh", "crop", "noisy", "down8k", "up60k"]), labels
rows = list(csv.DictReader(open(run / "train_log.csv")))
assert len(rows) == 2, len(rows)
vals = [float(v) for r in rows for k, v in r.items() if k != "epoch" and v not in ("", "nan")]
assert all(math.isfinite(v) for v in vals), rows
assert (run / "checkpoints" / "best_by_xtopo_mesh_clean.pth").is_file()
print(f"[smoke] eval online sulle etichette {labels}; train_log: {rows[-1]}")
PY
echo "[smoke] OK: $OUT"
